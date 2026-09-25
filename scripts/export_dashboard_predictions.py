"""Export validation-period predictions for the multi-hazard dashboard.

For each hazard, runs the chosen model over the validation inits and writes compact
arrays to outputs/dashboard/<hazard>/ plus an index.json the dashboard reads:

  rain       24-ch directional GNN (graphnet_20yr_directional_nbhd24.pt), val 2016-2019
  heatwave   GNN+CNN 4-checkpoint average (heatwave_{gnn,cnn}_fair_s{0,1}.pt), val Mar-Jun 2008-2009
  monsoon    active/break GNN wide head (active_break_gnn_wide.pt), JJAS 2008-2009, MCZ-level

Per-cell hazards (rain, heatwave), arrays [n_init, lead, lat, lon]:
  p_miss.npy, p_fa.npy       float16 model probabilities
  obs_miss.npy, obs_fa.npy   int8 observed bust (1/0, -1 = no observation)
  ens_prob.npy               float16 ensemble event probability (what GEFS itself said)
Monsoon (regional), arrays [n_init, lead, 4]: p.npy, obs.npy (+ ensemble p_active/p_break).

Alert tiers per hazard/direction come from the validation set itself:
  "high" = F1-optimal threshold; "elevated" = threshold that catches 50% of busts, omitted
  (None) when the high tier already catches >= 75% (a second tier would add nothing).
Re-run with --tiers-only to recompute tiers from the saved arrays without re-running models.
Each tier stores catch rate, precision and lift over the base rate for plain-language legends.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.detection.heavy_rainfall import ensemble_exceedance_probability
from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres
from src.models.graphnet.model import GridGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "dashboard"
LEADS, LAT, LON = 9, 66, 70
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def calibration(y: np.ndarray, p: np.ndarray, n_cut: int = 120) -> dict:
    """Map raw scores to observed bust rates (isotonic, fitted on the validation set).

    The models are trained with a large positive-class weight, so raw scores are inflated
    (a 0.97 rain-miss score busts ~19% of the time). The dashboard shows and thresholds
    the calibrated value instead. Returns interpolation knots (x = raw score, y = bust
    rate) plus a curve of catch rate / precision / share flagged at calibrated cut-offs.
    """
    from sklearn.isotonic import IsotonicRegression
    y = y.astype(np.float64); p = p.astype(np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    xs = np.unique(np.r_[np.linspace(0, 1, 201), np.quantile(p, np.linspace(0, 1, 401)), p.max()])
    ys = iso.predict(xs)
    c = iso.predict(p)
    order = np.argsort(-c)
    cs, ys_sorted = c[order], y[order]
    hits = np.cumsum(ys_sorted)
    total = max(hits[-1], 1.0)
    levels = np.unique(np.round(cs, 4))[::-1]
    cuts = levels[np.linspace(0, len(levels) - 1, min(n_cut, len(levels))).astype(int)]
    curve = []
    for cut in cuts:
        k = int(np.searchsorted(-cs, -cut, side="right"))  # cells with calibrated >= cut
        if k == 0:
            continue
        curve.append({"p": float(cut), "catch": float(hits[k - 1] / total),
                      "precision": float(hits[k - 1] / k), "share": float(k / len(cs))})
    return {"x": np.round(xs, 5).tolist(), "y": np.round(ys, 5).tolist(), "max": float(c.max()), "curve": curve}


def tiers(y: np.ndarray, p: np.ndarray) -> dict:
    """Validation-derived alert thresholds with plain-language stats."""
    base = float(y.mean())
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    hi = int(np.argmax(f1))
    el = int(np.argmin(np.abs(rec - 0.5)))
    tier = lambda i: {"threshold": float(thr[i]), "catch_rate": float(rec[i]), "precision": float(prec[i]),
                      "lift": float(prec[i] / base) if base > 0 else None,
                      "flagged_share": float((p >= thr[i]).mean())}
    return {"base_rate": base, "pr_auc": float(average_precision_score(y, p)),
            "elevated": tier(el) if rec[hi] < 0.75 and thr[el] < thr[hi] else None, "high": tier(hi),
            "calibration": calibration(y, p)}


def save(dirname: str, **arrays):
    d = OUT / dirname
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)


def as_obs(y: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(y), y, -1).astype(np.int8)


def notable(obs_miss: np.ndarray, obs_fa: np.ndarray, inits: list[str], k: int = 12) -> list[str]:
    score = (obs_miss == 1).sum((1, 2, 3)) + (obs_fa == 1).sum((1, 2, 3))
    return [inits[i] for i in np.argsort(-score)[:k]]


# ----------------------------------------------------------------------------- rain
def export_rain() -> dict:
    cache = Path(r"D:\sih-data\caches\rain_20yr_24ch_nbhd")
    store = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
    files = sorted(glob.glob(str(store / "*.nc")))
    n, ch = len(files), 24
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, ch))
    Y = np.memmap(cache / "Y_dir.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 2))
    rows = np.arange(int(0.8 * n), n)
    inits = [Path(files[r]).stem[:8] for r in rows]

    ck = torch.load(PROJECT_ROOT / "outputs" / "graphnet_20yr_directional_nbhd24.pt", map_location=DEV)
    model = GridGraphNet(build_grid_graph(), in_channels=ch, hidden=ck["hidden"], out_channels=2).to(DEV)
    model.load_state_dict(ck["model_state"]); model.eval()
    P = np.empty((len(rows), LEADS, LAT, LON, 2), np.float16)
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            x = torch.from_numpy(np.asarray(X[rows[s:s + 8]])).to(DEV)
            P[s:s + 8] = model(x).cpu().numpy()
    Yv = np.asarray(Y[rows])
    ens = np.empty((len(rows), LEADS, LAT, LON), np.float16)
    for k, r in enumerate(rows):
        with xr.open_dataset(files[r]) as ds:
            tp = ds["total_precipitation"].isel(run=0)
            ens[k] = np.stack([ensemble_exceedance_probability(tp.sel(lead=L).values, 64.5)
                               for L in tp["lead"].values])
    obs_m, obs_f = as_obs(Yv[..., 0]), as_obs(Yv[..., 1])
    save("rain", p_miss=P[..., 0], p_fa=P[..., 1], obs_miss=obs_m, obs_fa=obs_f, ens_prob=ens)
    fin = obs_m >= 0
    return {"label": "Extreme rainfall", "kind": "grid", "inits": inits,
            "event": "24-h rain >= 64.5 mm (IMD heavy)",
            "model": "GNN (GraphSAGE+TCN), 24 channels incl. neighbourhood ensemble features",
            "truth": "IMD 0.25 deg gridded rainfall",
            "val_period": f"{inits[0]}..{inits[-1]}",
            "tiers": {"miss": tiers(obs_m[fin], P[..., 0][fin].astype(np.float32)),
                      "false_alarm": tiers(obs_f[fin], P[..., 1][fin].astype(np.float32))},
            "notable_inits": notable(obs_m, obs_f, inits)}


# ----------------------------------------------------------------------------- heatwave
def export_heatwave() -> dict:
    from scripts.train_heatwave import HeatwaveBustNet
    cache = Path(r"D:\sih-data\caches\heatwave_10yr")
    meta = json.loads((cache / "meta.json").read_text())
    n, C = meta["n"], len(meta["channels"])
    X = np.fromfile(cache / "X.dat", np.float32).reshape(n, LEADS, LAT, LON, C)
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, LEADS, LAT, LON, 2)
    Pens = np.fromfile(cache / "P.dat", np.float32).reshape(n, LEADS, LAT, LON)
    va = np.where(~np.asarray(meta["is_train"]))[0]
    inits = [meta["inits"][i][:8] for i in va]
    acc = np.zeros((len(va), LEADS, LAT, LON, 2), np.float32)
    ckpts = [PROJECT_ROOT / "outputs" / f"heatwave_{a}_fair_s{s}.pt" for a in ("gnn", "cnn") for s in (0, 1)]
    for path in ckpts:
        st = torch.load(path, map_location=DEV)
        m = HeatwaveBustNet(st["arch"], C, st["wide_idx"]).to(DEV)
        m.load_state_dict(st["model_state"]); m.eval()
        with torch.no_grad():
            for s in range(0, len(va), 8):
                acc[s:s + 8] += m(torch.from_numpy(X[va[s:s + 8]]).to(DEV)).cpu().numpy()
    P = (acc / len(ckpts)).astype(np.float16)
    obs_m, obs_f = as_obs(Y[va][..., 0]), as_obs(Y[va][..., 1])
    save("heatwave", p_miss=P[..., 0], p_fa=P[..., 1], obs_miss=obs_m, obs_fa=obs_f,
         ens_prob=Pens[va].astype(np.float16))
    fin = obs_m >= 0
    return {"label": "Heatwave", "kind": "grid", "inits": inits,
            "event": "Tmax >= 40 C and >= 4.5 C above normal (or >= 45 C)",
            "model": "GNN + CNN ensemble (4 checkpoints averaged)",
            "truth": "ERA5 daily maximum temperature",
            "val_period": f"{inits[0]}..{inits[-1]} (Mar-Jun)",
            "tiers": {"miss": tiers(obs_m[fin], P[..., 0][fin].astype(np.float32)),
                      "false_alarm": tiers(obs_f[fin], P[..., 1][fin].astype(np.float32))},
            "notable_inits": notable(obs_m, obs_f, inits)}


# ----------------------------------------------------------------------------- monsoon
def export_monsoon() -> dict:
    from scripts.train_active_break import AUX, CACHE, CH, LABELS, TARGETS, RegimeBustNet
    lab = np.load(LABELS)
    n_cache = (CACHE / "X.dat").stat().st_size // (LEADS * LAT * LON * CH * 4)
    X = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n_cache, LEADS, LAT, LON, CH))
    rows = np.unique(lab["cache_row"]); pos = {r: i for i, r in enumerate(rows)}
    Y = np.full((len(rows), LEADS, len(TARGETS)), -1, np.int8)
    A = np.zeros((len(rows), LEADS, len(AUX)), np.float32)
    TR = np.zeros(len(rows), bool); init_of = [""] * len(rows)
    for k in range(len(lab["cache_row"])):
        i, L = pos[lab["cache_row"][k]], lab["lead"][k] - 1
        Y[i, L] = [lab[t][k] for t in TARGETS]
        A[i, L] = np.nan_to_num([lab[a][k] for a in AUX])
        TR[i] = lab["is_train"][k]; init_of[i] = str(lab["init"][k])
    va = np.where(~TR)[0]
    st = torch.load(PROJECT_ROOT / "outputs" / "active_break_gnn_wide.pt", map_location=DEV)
    model = RegimeBustNet(st["arch"], head=st["head"]).to(DEV)
    model.load_state_dict(st["model_state"]); model.eval()
    P = np.empty((len(va), LEADS, len(TARGETS)), np.float32)
    with torch.no_grad():
        for s in range(0, len(va), 8):
            idx = va[s:s + 8]
            P[s:s + 8] = model(torch.from_numpy(np.asarray(X[rows[idx]])).to(DEV),
                               torch.from_numpy(A[idx]).to(DEV)).cpu().numpy()
    Yv = Y[va]
    save("monsoon", p=P.astype(np.float16), obs=Yv,
         p_active=A[va][..., AUX.index("p_active")].astype(np.float16),
         p_break=A[va][..., AUX.index("p_break")].astype(np.float16))
    t = {}
    for j, name in enumerate(TARGETS):
        fin = Yv[..., j] >= 0
        if Yv[..., j][fin].sum() >= 20:
            t[name] = tiers(Yv[..., j][fin], P[..., j][fin])
    return {"label": "Active / break monsoon", "kind": "regional", "inits": [init_of[i] for i in va],
            "targets": list(TARGETS), "region": "Monsoon core zone 18-28N, 73-86E",
            "event": "Active / break spell: MCZ rain index beyond +-1 sigma for >= 3 days",
            "model": "GNN (GraphSAGE+TCN) with frozen ensemble-baseline head",
            "truth": "IMD gridded rainfall over the monsoon core zone",
            "val_period": "JJAS 2008-2009", "tiers": t,
            "notable_inits": [init_of[va[i]] for i in np.argsort(-(Yv == 1).sum((1, 2)))[:12]]}


def recompute_tiers() -> None:
    index = json.loads((OUT / "index.json").read_text())
    for name, v in index.items():
        a = {q.stem: np.load(q, mmap_mode="r") for q in (OUT / name).glob("*.npy")}
        if v["kind"] == "grid":
            fin = np.asarray(a["obs_miss"]) >= 0
            v["tiers"] = {d: tiers(np.asarray(a[o])[fin], np.asarray(a[pk])[fin].astype(np.float32))
                          for d, pk, o in (("miss", "p_miss", "obs_miss"), ("false_alarm", "p_fa", "obs_fa"))}
        else:
            obs, p = np.asarray(a["obs"]), np.asarray(a["p"], np.float32)
            v["tiers"] = {t: tiers(obs[..., j][obs[..., j] >= 0], p[..., j][obs[..., j] >= 0])
                          for j, t in enumerate(v["targets"]) if obs[..., j][obs[..., j] >= 0].sum() >= 20}
            v["notable_inits"] = [v["inits"][i] for i in np.argsort(-(obs == 1).sum((1, 2)))[:12]]
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    report(index)


def report(index: dict) -> None:
    for name, v in index.items():
        for d, t in v["tiers"].items():
            el = t["elevated"]
            print(f"  {name}/{d}: base {100*t['base_rate']:.2f}% pr-auc {t['pr_auc']:.3f} | high catch "
                  f"{100*t['high']['catch_rate']:.0f}% prec {100*t['high']['precision']:.0f}% | elevated "
                  + (f"catch {100*el['catch_rate']:.0f}% prec {100*el['precision']:.0f}%" if el else "none"), flush=True)


def main():
    import sys
    if "--tiers-only" in sys.argv:
        recompute_tiers(); return
    t0 = time.time()
    index = {}
    for name, fn in (("heatwave", export_heatwave), ("monsoon", export_monsoon), ("rain", export_rain)):
        index[name] = fn()
        print(f"{name}: {len(index[name]['inits'])} inits | {time.time()-t0:.0f}s", flush=True)
        (OUT / "index.json").write_text(json.dumps(index, indent=1))
    lats, lons = canonical_centres(0.5)
    np.save(OUT / "lats.npy", lats); np.save(OUT / "lons.npy", lons)
    report(index)

if __name__ == "__main__":
    main()
