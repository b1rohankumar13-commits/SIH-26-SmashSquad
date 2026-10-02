"""Explain the active/break monsoon bust flags: which meteorological drivers push each one up.

Same method as scripts/export_explanations.py (gradient x input on the logit), adapted to the
regional monsoon model: the logit of one target (break/active x miss/false alarm) at one lead is
differentiated against the map inputs (65 channels, summed over the monsoon core zone, where the
model pools) and the ensemble regime signals the head reads directly. Contributions are grouped
into drivers; each reports its share of the positive push and its value (the core-zone mean in
standard deviations from typical, or the ensemble probability for regime signals).

Only flagged forecasts are explained: a target whose probability reaches its high alert tier.

Writes outputs/dashboard/monsoon/explain.json: {init: {lead: {target: [{name, share, value}]}}}
    python -m scripts.export_monsoon_explanations
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.export_dashboard_predictions import DEV, LAT, LEADS, LON, OUT
from scripts.train_active_break import AUX, CACHE, CH, LABELS, TARGETS, RegimeBustNet
from src.detection.monsoon_phase import mcz_mask
from src.features.build_gridded_sequences_rich import RICH_CHANNEL_NAMES
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_DRIVERS = 4


def mv(*v: str) -> list[str]:
    return [f"{a}_{s}" for a in v for s in ("mean", "spread")]


GROUPS = {
    "GEFS core-zone rain": mv("total_precipitation") + ["rain_exc_heavy", "rain_exc_very_heavy", "rain_exc_extreme"],
    "Ensemble regime signal": ["p_active", "p_break", "ens_mean_anom", "ens_spread"],
    "Low-level moisture": mv("spfh_850", "spfh_700", "spfh_2m") + ["moisture_flux_850"],
    "Monsoon flow (850/700 hPa)": mv("ugrd_850", "vgrd_850", "ugrd_700", "vgrd_700", "ugrd_10m", "vgrd_10m"),
    "Convection and cloud (OLR, CAPE)": mv("olr", "cape"),
    "Monsoon trough (pressure, spin, convergence)": mv("pres_msl", "pres_sfc") + ["vorticity_850", "convergence_850"],
    "Upper-level pattern": mv("hgt_850", "hgt_700", "hgt_500", "hgt_300", "ugrd_500", "vgrd_500", "ugrd_300", "vgrd_300")
                           + ["gh500_grad"],
    "Temperature profile": mv("tmp_2m", "tmp_850", "tmp_700", "tmp_500", "tmp_300"),
}
# display value per driver: (channel, "sigma" | "prob")
DISPLAY = {"GEFS core-zone rain": ("total_precipitation_mean", "sigma"), "Ensemble regime signal": (None, "prob"),
           "Low-level moisture": ("spfh_850_mean", "sigma"), "Monsoon flow (850/700 hPa)": ("ugrd_850_mean", "sigma"),
           "Convection and cloud (OLR, CAPE)": ("olr_mean", "sigma"),
           "Monsoon trough (pressure, spin, convergence)": ("pres_msl_mean", "sigma"),
           "Upper-level pattern": ("hgt_500_mean", "sigma"), "Temperature profile": ("tmp_850_mean", "sigma")}


def main() -> None:
    t0 = time.time()
    index = json.loads((OUT / "index.json").read_text())["monsoon"]
    lab = np.load(LABELS)
    n_cache = (CACHE / "X.dat").stat().st_size // (LEADS * LAT * LON * CH * 4)
    X = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n_cache, LEADS, LAT, LON, CH))
    rows = np.unique(lab["cache_row"]); pos = {r: i for i, r in enumerate(rows)}
    A = np.zeros((len(rows), LEADS, len(AUX)), np.float32); TR = np.zeros(len(rows), bool); init_of = [""] * len(rows)
    for k in range(len(lab["cache_row"])):
        i, L = pos[lab["cache_row"][k]], lab["lead"][k] - 1
        A[i, L] = np.nan_to_num([lab[a][k] for a in AUX]); TR[i] = lab["is_train"][k]; init_of[i] = str(lab["init"][k])
    va = np.where(~TR)[0]
    st = torch.load(PROJECT_ROOT / "outputs" / "active_break_gnn_wide.pt", map_location=DEV)
    model = RegimeBustNet(st["arch"], head=st["head"]).to(DEV); model.load_state_dict(st["model_state"]); model.eval()
    names = list(RICH_CHANNEL_NAMES)
    ch = {c: j for j, c in enumerate(names)}
    mask = torch.from_numpy(mcz_mask(*canonical_centres(0.5))).to(DEV)
    cut = {t: index["tiers"][t]["high"]["threshold"] for t in TARGETS if t in index["tiers"]}
    out: dict = {}
    n_expl = 0
    for i in va:
        x = torch.from_numpy(np.asarray(X[rows[i]])[None]).to(DEV).requires_grad_(True)
        aux = torch.from_numpy(A[i][None]).to(DEV).requires_grad_(True)
        p = model(x, aux)[0]  # [L, 4]
        logit = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        zmean = x[0][:, mask].mean(1).detach().cpu().numpy()  # [L, CH] core-zone mean (sigma units)
        for L in range(LEADS):
            for t_i, t in enumerate(TARGETS):
                if t not in cut or float(p[L, t_i]) < cut[t]:
                    continue
                gx, ga = torch.autograd.grad(logit[L, t_i], (x, aux), retain_graph=True)
                cx = (gx * x)[0][:, mask].sum((0, 1)).detach().cpu().numpy()   # [CH]
                ca = (ga * aux)[0].sum(0).detach().cpu().numpy()                # [AUX]
                score = {}
                for g, cs in GROUPS.items():
                    score[g] = float(sum(cx[ch[c]] for c in cs if c in ch) + sum(ca[AUX.index(c)] for c in cs if c in AUX))
                pos_ = {g: v for g, v in score.items() if v > 0}
                total = sum(pos_.values()) or 1.0
                drivers = []
                for g, v in sorted(pos_.items(), key=lambda kv: -kv[1])[:TOP_DRIVERS]:
                    disp, kind = DISPLAY[g]
                    if kind == "prob":
                        side = "break" if "break" in t else "active"
                        val = f"{100 * A[i, L, AUX.index('p_' + side)]:.0f}% of members in {side}"
                    else:
                        val = f"{zmean[L, ch[disp]]:+.1f}σ vs typical"
                    drivers.append({"name": g, "share": round(v / total, 3), "value": val})
                out.setdefault(init_of[i], {}).setdefault(str(L + 1), {})[t] = drivers
                n_expl += 1
        del x, aux, p, logit
    (OUT / "monsoon" / "explain.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"monsoon: {n_expl} flagged forecasts explained over {len(va)} inits | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
