"""Explain the dashboard's flagged areas: which meteorological drivers push each area's bust risk up.

For every validation forecast (init, lead, direction), the cells the map colours (calibrated
probability at or above the display cut, as in web/src/lib/risk.ts) are grouped into connected
areas; the top 3 by expected busts are explained. Attribution is gradient x input on the
model's logit summed over the area: inputs are standardized, so 0 is the training-average
input and each channel's contribution is how far it moved this area's risk from that. Channel
contributions are summed over space and lead into plain-language drivers; each driver reports
its share of the positive contribution and its value over the area at that lead (physical units
where meaningful, otherwise standard deviations from typical). Multi-checkpoint hazards average
the attributions. Non-meteorological inputs (lead, land mask, day-of-year sin/cos) are not shown.

Writes outputs/dashboard/<hazard>/explain.json:
  {init: {lead: {"miss" | "false_alarm": [{lat, lon, cells, expected, box, drivers: [{name, share, value}]}]}}}

    python -m scripts.export_explanations            # rain, heatwave, lps
    python -m scripts.export_explanations --only lps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from scripts.export_dashboard_predictions import (
    DEV, LAT, LEADS, LON, LPS_CACHE, LPS_CKPTS, OUT, heatwave_checkpoints, lps_features, lps_inputs, rain_rows,
)
from src.models.graphnet.grid_graph import canonical_centres

SHOW_FROM = 0.5
TOP_AREAS, TOP_DRIVERS, MIN_CELLS = 3, 4, 3
SKIP = {"lead01", "land", "sin_doy", "cos_doy"}

mm = lambda v: f"{v:.0f} mm"
pct = lambda v: f"{100 * v:.0f}%"
degc = lambda v: f"{v:.1f} °C"
dep = lambda v: f"{v:+.1f} °C vs normal"

# driver name -> (channels, display channel, formatter for its physical value or None for sigma)
RAIN_GROUPS = {
    "Forecast rain amount": (["rainfall_mean_mm"], "rainfall_mean_mm", mm),
    "Members disagree on rain": (["rainfall_spread_mm"], "rainfall_spread_mm", mm),
    "GEFS heavy-rain probability": (["rain_exc_heavy", "rain_exc_very_heavy", "rain_exc_extreme"], "rain_exc_heavy", pct),
    "Heavy rain forecast nearby": (["nep645_r1", "nep645_r2", "p20", "p35"], "nep645_r2", pct),
    "Members place rain differently": (["dfss35_r1"], "dfss35_r1", None),
    "850 hPa moisture": (["q850_mean_kgkg", "q850_spread_kgkg", "moisture_flux_850"], "moisture_flux_850", None),
    "Low-level convergence": (["convergence_850"], "convergence_850", None),
    "Low-level spin (850 hPa vorticity)": (["vorticity_850"], "vorticity_850", None),
    "850 hPa winds": (["u850_mean_ms", "u850_spread_ms", "v850_mean_ms", "v850_spread_ms"], "u850_mean_ms", None),
    "Sea-level pressure": (["mslp_mean_pa", "mslp_spread_pa"], "mslp_mean_pa", None),
    "500 hPa pattern": (["gh500_mean_gpm", "gh500_spread_gpm", "gh500_grad"], "gh500_mean_gpm", None),
}
HEAT_GROUPS = {
    "Forecast Tmax": (["tmax_mean"], "tmax_mean", degc),
    "Forecast heat anomaly": (["tmax_departure"], "tmax_departure", dep),
    "Members disagree on Tmax": (["tmax_spread"], "tmax_spread", degc),
    "GEFS heatwave probability": (["p_heatwave", "p_ge40"], "p_heatwave", pct),
    "Observed heat, last 2 days": (["obs_departure_d-1", "obs_departure_d-2"], "obs_departure_d-1", dep),
    "Normal Tmax for the date": (["tmax_normal"], "tmax_normal", degc),
    "2 m temperature": (["tmp_2m_mean", "tmp_2m_spread"], "tmp_2m_mean", None),
    "850 hPa temperature": (["tmp_850_mean", "tmp_850_spread"], "tmp_850_mean", None),
    "Upper-level ridge (500/300 hPa)": (["hgt_500_mean", "hgt_500_spread", "hgt_300_mean", "hgt_300_spread"], "hgt_500_mean", None),
    "Near-surface humidity": (["spfh_2m_mean", "spfh_2m_spread"], "spfh_2m_mean", None),
    "Sea-level pressure": (["pres_msl_mean", "pres_msl_spread"], "pres_msl_mean", None),
}
LPS_GROUPS = {
    **{k: v for k, v in RAIN_GROUPS.items() if k not in ("Members place rain differently",)},
    "GEFS storm within 600 km": (["p_strike300", "p_strike600"], "p_strike600", pct),
    "Spin in GEFS members (850 hPa)": (["zeta850_mean", "zeta850_spread"], "zeta850_mean", None),
    "Pressure dip in GEFS members": (["msl_dip_mean"], "msl_dip_mean", None),
    "Systems GEFS sees in the domain": (["n_systems_domain"], "n_systems_domain", None),
    "MJO at issue time": (["rmm1", "rmm2", "rmm_amp"], "rmm_amp", None),
    "Seasonal storm climatology": (["clim_strike300"], "clim_strike300", pct),
}


def display_cut(t: dict) -> float:
    base = t["base_rate"]
    curve = sorted((c for c in t["calibration"]["curve"] if c["p"] > 0 and c["catch"] >= 0.01), key=lambda c: c["p"])
    cap = max(curve[-1]["p"] if curve else t["calibration"]["max"], base * 2)
    return base * np.exp(SHOW_FROM * np.log(cap / base))


def calibrated(p: np.ndarray, cal: dict, lead: int) -> np.ndarray:
    k = next((b for b in cal.get("bands", []) if lead in b["leads"]), cal)
    return np.interp(p, k["x"], k["y"])


def areas(hazard: str, index: dict) -> tuple[dict, list[str]]:
    """{(row, lead0, dir): [(cells mask, expected), ...]} for the top areas the map colours."""
    a = {q.stem: np.load(q, mmap_mode="r") for q in (OUT / hazard).glob("*.npy")}
    valid = np.asarray(a["obs_miss"]) >= 0
    out = {}
    for d, pk in (("miss", "p_miss"), ("false_alarm", "p_fa")):
        t = index[hazard]["tiers"][d]; cut = display_cut(t)
        p = np.asarray(a[pk], np.float32)
        for L in range(LEADS):
            cal = calibrated(p[:, L], t["calibration"], L + 1)
            flag = (cal >= cut) & valid[:, L]
            for i in range(len(p)):
                lab, n = ndimage.label(flag[i], structure=np.ones((3, 3)))
                if n == 0:
                    continue
                exp = ndimage.sum(cal[i], lab, index=np.arange(1, n + 1))
                size = ndimage.sum(np.ones_like(cal[i]), lab, index=np.arange(1, n + 1))
                keep = [k for k in np.argsort(-exp) if size[k] >= MIN_CELLS][:TOP_AREAS]
                if keep:
                    out[(i, L, d)] = [(lab == k + 1, float(exp[k])) for k in keep]
    return out, index[hazard]["inits"]


def attribute(forward_models, features, rows, names, groups, raw_of, found, n_rows, label: str) -> dict:
    """forward_models: list of (model, needs) where features(row, needs) -> [L,H,W,C] tensor input."""
    lats, lons = canonical_centres(0.5)
    ch = {c: i for i, c in enumerate(names)}
    gidx = {g: [ch[c] for c in cs if c in ch] for g, (cs, _, _) in groups.items()}
    gidx = {g: v for g, v in gidx.items() if v}
    res: dict = {}
    t0 = time.time()
    for s in range(0, n_rows, 8):
        batch = list(range(s, min(s + 8, n_rows)))
        todo = [(L, d, k) for L in range(LEADS) for d in ("miss", "false_alarm")
                for k in range(TOP_AREAS) if any(len(found.get((i, L, d), [])) > k for i in batch)]
        if not todo:
            continue
        attr = {}  # (i, L, d, k) -> [C]
        for model, needs in forward_models:
            x = torch.from_numpy(np.stack([features(rows[i], needs) for i in batch])).to(DEV).requires_grad_(True)
            logit = torch.logit(model(x).clamp(1e-6, 1 - 1e-6))
            for L, d, k in todo:
                mask = np.zeros((len(batch), LAT, LON), np.float32)
                for b, i in enumerate(batch):
                    ar = found.get((i, L, d), [])
                    if len(ar) > k:
                        mask[b] = ar[k][0]
                obj = (logit[:, L, :, :, 0 if d == "miss" else 1] * torch.from_numpy(mask).to(DEV)).sum()
                (g,) = torch.autograd.grad(obj, x, retain_graph=True)
                contrib = (g * x).sum(dim=(1, 2, 3)).detach().cpu().numpy()  # [B, C]
                for b, i in enumerate(batch):
                    if mask[b].any():
                        attr[(i, L, d, k)] = attr.get((i, L, d, k), 0) + contrib[b] / len(forward_models)
            del x, logit
        for (i, L, d, k), c in attr.items():
            cells, expected = found[(i, L, d)][k]
            raw = raw_of(rows[i], L, cells)
            score = {g: float(c[ix].sum()) for g, ix in gidx.items()}
            pos = {g: v for g, v in score.items() if v > 0}
            total = sum(pos.values()) or 1.0
            drivers = []
            for g, v in sorted(pos.items(), key=lambda kv: -kv[1])[:TOP_DRIVERS]:
                _, disp, fmt = groups[g]
                val = raw.get(disp)
                text = (fmt(val[0]) if fmt else f"{val[1]:+.1f}σ vs typical") if val else ""
                drivers.append({"name": g, "share": round(v / total, 3), "value": text})
            yy, xx = np.nonzero(cells)
            res.setdefault(label_of(i, label), {}).setdefault(str(L + 1), {}).setdefault(d, []).append({
                "lat": round(float(lats[yy].mean()), 2), "lon": round(float(lons[xx].mean()), 2),
                "cells": int(cells.sum()), "expected": round(expected, 2),
                "box": [float(lats[yy].min()), float(lats[yy].max()), float(lons[xx].min()), float(lons[xx].max())],
                "drivers": drivers})
        if s % 200 == 0:
            print(f"  {label}: {s}/{n_rows} | {time.time()-t0:.0f}s", flush=True)
    return res


_INITS: dict[str, list[str]] = {}


def label_of(i: int, label: str) -> str:
    return _INITS[label][i]


def raw_reader(names, mean, std, getter):
    """(row, lead, cells) -> {channel: (physical value, z)} averaged over the area at that lead."""
    def f(row, L, cells):
        z = getter(row)[L][cells].mean(0)
        return {c: (float(z[j] * std[j] + mean[j]), float(z[j])) for j, c in enumerate(names) if j < len(mean)}
    return f


def run_rain(index):
    from src.features.build_gridded_sequences_nbhd import NBHD_CHANNEL_NAMES
    from src.models.graphnet.grid_graph import build_grid_graph
    from src.models.graphnet.model import GridGraphNet
    cache = Path(r"D:\sih-data\caches\rain_20yr_24ch_nbhd")
    rows, n = rain_rows()
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 24))
    std = np.load(cache / "standardizer.npz")
    ck = torch.load(Path(__file__).resolve().parents[1] / "outputs" / "graphnet_20yr_directional_nbhd24.pt", map_location=DEV)
    m = GridGraphNet(build_grid_graph(), in_channels=24, hidden=ck["hidden"], out_channels=2).to(DEV)
    m.load_state_dict(ck["model_state"]); m.eval()
    found, inits = areas("rain", index); _INITS["rain"] = inits
    names = list(NBHD_CHANNEL_NAMES)
    return attribute([(m, None)], lambda r, _: np.asarray(X[r]), list(rows), names, RAIN_GROUPS,
                     raw_reader(names, std["mean"], std["std"], lambda r: np.asarray(X[r])), found, len(rows), "rain")


def run_heatwave(index):
    from scripts.train_heatwave import HeatwaveBustNet
    cache = Path(r"D:\sih-data\caches\heatwave_10yr")
    meta = json.loads((cache / "meta.json").read_text())
    n, C = meta["n"], len(meta["channels"])
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, C))
    std = np.load(cache / "standardizer.npz")
    va = np.where(~np.asarray(meta["is_train"]))[0]
    models = []
    for st in heatwave_checkpoints():
        m = HeatwaveBustNet(st["arch"], C, st["wide_idx"]).to(DEV); m.load_state_dict(st["model_state"]); m.eval()
        models.append((m, None))
    found, inits = areas("heatwave", index); _INITS["heatwave"] = inits
    return attribute(models, lambda r, _: np.asarray(X[r]), list(va), meta["channels"], HEAT_GROUPS,
                     raw_reader(meta["channels"], std["mean"], std["std"], lambda r: np.asarray(X[r])),
                     found, len(va), "heatwave")


def run_lps(index):
    from scripts.train_heatwave import HeatwaveBustNet
    from src.features.build_gridded_sequences_nbhd import NBHD_CHANNEL_NAMES
    meta, rich, Xl, va = lps_inputs()
    rain_std = np.load(Path(meta["rich_cache"]) / "standardizer.npz")
    lps_std = np.load(LPS_CACHE / "standardizer.npz")
    env_std = np.load(LPS_CACHE / "env_standardizer.npz")
    env_names = meta["env_channels"]
    names = list(NBHD_CHANNEL_NAMES) + meta["channels"] + env_names + ["lead01"]
    mean = np.r_[rain_std["mean"], lps_std["mean"], env_std["mean"]]
    sd = np.r_[rain_std["std"], lps_std["std"], env_std["std"]]
    models = []
    for path in LPS_CKPTS:
        st = torch.load(path, map_location=DEV)
        m = HeatwaveBustNet(st["arch"], st["in_channels"], st["wide_idx"]).to(DEV); m.load_state_dict(st["model_state"]); m.eval()
        models.append((m, bool(st.get("env"))))
    n_rain, n_lps, n_env = 24, len(meta["channels"]), len(env_names)
    full = lambda r: lps_features(rich, Xl, r, True)  # rain, lps, env, lead
    plain_cols = list(range(n_rain + n_lps)) + [n_rain + n_lps + n_env]

    def features(r, env):
        x = full(r)
        return x if env else x[..., plain_cols]

    # The plain model has no env channels: attribution comes back in its own channel layout, so map it
    # into the full layout by wrapping it to return full-width contributions.
    class Wrap(torch.nn.Module):
        def __init__(self, m, env):
            super().__init__(); self.m, self.env = m, env
            self.cols = torch.tensor(plain_cols, device=DEV)

        def forward(self, x):
            return self.m(x if self.env else x.index_select(-1, self.cols))

    wrapped = [(Wrap(m, env), True) for m, env in models]
    found, inits = areas("lps", index); _INITS["lps"] = inits
    return attribute(wrapped, features, list(va), names, LPS_GROUPS,
                     raw_reader(names, mean, sd, lambda r: full(r)), found, len(va), "lps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="rain,heatwave,lps")
    args = ap.parse_args()
    index = json.loads((OUT / "index.json").read_text())
    for h, fn in (("heatwave", run_heatwave), ("lps", run_lps), ("rain", run_rain)):
        if h in args.only.split(","):
            t0 = time.time()
            res = fn(index)
            (OUT / h / "explain.json").write_text(json.dumps(res, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
            n = sum(len(v) for per in res.values() for lv in per.values() for v in lv.values())
            print(f"{h}: {n} areas explained | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
