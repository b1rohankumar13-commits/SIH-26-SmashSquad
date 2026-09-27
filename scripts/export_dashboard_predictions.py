"""Export validation-period predictions for the multi-hazard dashboard.

For each hazard, runs the chosen model over the validation inits and writes compact
arrays to outputs/dashboard/<hazard>/ plus an index.json the dashboard reads:

  rain       24-ch directional GNN (graphnet_20yr_directional_nbhd24.pt), val 2016-2019
  heatwave   GNN+CNN 4-checkpoint average (heatwave_{gnn,cnn}_fair_s{0,1}.pt), val Mar-Jun 2008-2009
  monsoon    active/break GNN wide head (active_break_gnn_wide.pt), JJAS 2008-2009, MCZ-level
  lps        low-pressure-system strike GNNs, 20 years: average of lps_gnn20.pt and lps_gnn20env.pt
             (+ MJO/climatology channels), val 2016-2019, + ERA5 tracks.json and GEFS gefs_tracks.json

Per-cell hazards (rain, heatwave), arrays [n_init, lead, lat, lon]:
  p_miss.npy, p_fa.npy       float16 model probabilities
  obs_miss.npy, obs_fa.npy   int8 observed bust (1/0, -1 = no observation)
  ens_prob.npy               float16 ensemble event probability (what GEFS itself said)
Monsoon (regional), arrays [n_init, lead, 4]: p.npy, obs.npy (+ ensemble p_active/p_break).

Alert tiers per hazard/direction come from the validation set itself:
  "high" = F1-optimal threshold; "elevated" = threshold that catches 50% of busts, omitted
  (None) when the high tier already catches >= 75% (a second tier would add nothing).
Re-export a subset with --only lps[,rain...] (merged into the existing index).
Re-run with --tiers-only to recompute tiers from the saved arrays without re-running models.
Each tier stores catch rate, precision and lift over the base rate for plain-language legends.

Skill vs GEFS alone: every model is scored against an ensemble-only logistic baseline on
the same validation cells (b_miss/b_fa.npy, or b.npy for monsoon). For heatwave and
monsoon that baseline is the frozen wide path inside the model; rain has none, so the
equivalent logistic (local ensemble rain mean/spread/exceedance + lead) is fitted on the
training inits. Reports AUROC, PR-AUC and Brier skill vs climatology on the cells where
each bust is possible (miss: GEFS < 20%, false alarm: GEFS >= 50%); Brier uses isotonic
calibration fitted on one half of the validation period and scored on the other.
Re-run with --skill-only to rebuild the baselines and skill without re-running the models.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.detection.heavy_rainfall import ensemble_exceedance_probability
from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres
from src.models.graphnet.model import GridGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "dashboard"
LEADS, LAT, LON = 9, 66, 70
DEV = "cuda" if torch.cuda.is_available() else "cpu"


LEAD_BANDS = tuple((L,) for L in range(1, 10))  # one fit per lead day


def _knots(y: np.ndarray, p: np.ndarray) -> tuple[list, list, np.ndarray]:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    xs = np.unique(np.r_[np.linspace(0, 1, 201), np.quantile(p, np.linspace(0, 1, 401)), p.max()])
    return np.round(xs, 5).tolist(), np.round(iso.predict(xs), 5).tolist(), iso.predict(p)


def calibration(y: np.ndarray, p: np.ndarray, lead: np.ndarray | None = None, n_cut: int = 120) -> dict:
    """Map raw scores to observed bust rates (isotonic, fitted on the validation set).

    The models are trained with a large positive-class weight, so raw scores are inflated
    (a 0.97 rain-miss score busts ~19% of the time). The dashboard shows and thresholds
    the calibrated value instead. Returns interpolation knots (x = raw score, y = bust
    rate) plus a curve of catch rate / precision / share flagged at calibrated cut-offs.

    With `lead`, the map is fitted per lead day (in "bands"): skill
    falls with lead, so one pooled fit overstates long-range risk (~10% too many storm
    misses at day 9) and understates short-range risk. The curve then uses those values.
    """
    y = y.astype(np.float64); p = p.astype(np.float64)
    xs, ys, c = _knots(y, p)
    bands = None
    if lead is not None:
        bands = []
        for leads in LEAD_BANDS:
            sel = np.isin(lead, leads)
            bx, by, c[sel] = _knots(y[sel], p[sel])
            bands.append({"leads": list(leads), "x": bx, "y": by})
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
    out = {"x": xs, "y": ys, "max": float(c.max()), "curve": curve}
    if bands:
        out["bands"] = bands
    return out


def tiers(y: np.ndarray, p: np.ndarray, lead: np.ndarray | None = None) -> dict:
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
            "calibration": calibration(y, p, lead)}


def save(dirname: str, **arrays):
    d = OUT / dirname
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)


def lead_of(obs: np.ndarray) -> np.ndarray:
    """1-based lead for every element of an [init, lead, lat, lon] array."""
    return np.broadcast_to(np.arange(1, obs.shape[1] + 1)[None, :, None, None], obs.shape)


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
            "tiers": {"miss": tiers(obs_m[fin], P[..., 0][fin].astype(np.float32), lead_of(obs_m)[fin]),
                      "false_alarm": tiers(obs_f[fin], P[..., 1][fin].astype(np.float32), lead_of(obs_m)[fin])},
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
            "tiers": {"miss": tiers(obs_m[fin], P[..., 0][fin].astype(np.float32), lead_of(obs_m)[fin]),
                      "false_alarm": tiers(obs_f[fin], P[..., 1][fin].astype(np.float32), lead_of(obs_m)[fin])},
            "notable_inits": notable(obs_m, obs_f, inits)}


# ----------------------------------------------------------------------------- low-pressure systems
LPS_CACHE = Path(r"D:\sih-data\caches\lps_20yr")
# Averaged: the plain model is sharper at short leads, the environment model (MJO + seasonal
# climatology) at long leads; their mean beat both at every lead band on 2016-19 and 2022.
LPS_CKPTS = [PROJECT_ROOT / "outputs" / "lps_gnn20.pt", PROJECT_ROOT / "outputs" / "lps_gnn20env.pt"]
LPS_TRUTH = PROJECT_ROOT / "data" / "processed" / "lps_truth_20yr.npz"


def lps_inputs() -> tuple[dict, np.memmap, np.ndarray, np.ndarray]:
    """(meta, feature memmap, LPS channels, validation rows) for the storm model's input."""
    meta = json.loads((LPS_CACHE / "meta.json").read_text())
    fc = meta.get("feature_ch", 65)
    n = meta["n"]
    rich = np.memmap(Path(meta["rich_cache"]) / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, fc))
    Xl = np.memmap(LPS_CACHE / "X_lps.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, len(meta["channels"])))
    return meta, rich, Xl, np.where(~np.asarray(meta["is_train"]))[0]


def lps_features(rich: np.memmap | None, Xl: np.ndarray, i: int, env: bool = False) -> np.ndarray:
    """One init's model input: rain features, tracker channels, [environment channels], lead.
    rich=None zero-fills the rain block (the frozen baseline never reads it)."""
    meta = json.loads((LPS_CACHE / "meta.json").read_text()) if env or rich is None else None
    fc = meta.get("feature_ch", 65) if meta else 0
    parts = [np.asarray(rich[i]) if rich is not None else np.zeros((LEADS, LAT, LON, fc), np.float32),
             np.asarray(Xl[i])]
    if env:
        Xe = np.memmap(LPS_CACHE / "X_env.dat", np.float32, "r",
                       shape=(meta["n"], LEADS, LAT, LON, len(meta["env_channels"])))
        parts.append(np.asarray(Xe[i]))
    return np.concatenate(parts + [LEAD01.astype(np.float32)], -1)


def export_lps_tracks(inits: list[str]) -> None:
    """ERA5 tracker fixes for the validation window (+ lead days), for the map overlay."""
    z = np.load(LPS_TRUTH)
    first, last = inits[0], (np.datetime64(f"{inits[-1][:4]}-{inits[-1][4:6]}-{inits[-1][6:]}") + np.timedelta64(LEADS, "D"))
    last = str(last).replace("-", "")
    dates = z["dates"][z["t"]]
    keep = (dates >= first) & (dates <= last)
    tracks: dict[str, list] = {}
    for d, tr, la, lo in zip(dates[keep], z["track"][keep], z["lat"][keep], z["lon"][keep]):
        tracks.setdefault(str(int(tr)), []).append([str(d), float(la), float(lo)])
    (OUT / "lps").mkdir(parents=True, exist_ok=True)
    (OUT / "lps" / "tracks.json").write_text(json.dumps({k: sorted(v) for k, v in tracks.items()}))


def export_gefs_tracks(inits: list[str], dets: Path = PROJECT_ROOT / "data" / "processed" / "gefs_lps_dets_20yr.npz") -> None:
    """Each GEFS member's forecast storm tracks per init (tracker detections linked across lead days),
    for the map: {init: [{member, points: [[lead, lat, lon], ...]}]}. From extract_gefs_lps.py."""
    from src.detection.lps_tracker import Detection, link_tracks
    if not dets.exists():
        print(f"  no {dets.name}; skipping GEFS tracks", flush=True)
        return
    z = np.load(dets)
    want = set(inits)
    by_init: dict[str, dict[int, list]] = {}
    for i, m, L, la, lo, ze in zip(z["init"], z["member"], z["lead"], z["lat"], z["lon"], z["zeta"]):
        if i in want:
            by_init.setdefault(str(i), {}).setdefault(int(m), []).append(Detection(int(L), float(la), float(lo), float(ze), 0.0))
    out = {init: [{"member": m, "points": [[d.t, d.lat, d.lon] for d in tr]}
                  for m, ds in members.items() for tr in link_tracks(ds)]
           for init, members in by_init.items()}
    (OUT / "lps").mkdir(parents=True, exist_ok=True)
    (OUT / "lps" / "gefs_tracks.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  GEFS tracks for {len(out)} inits", flush=True)


def export_lps() -> dict:
    from scripts.train_heatwave import HeatwaveBustNet
    meta, rich, Xl, va = lps_inputs()
    n = meta["n"]
    Y = np.fromfile(LPS_CACHE / "Y_dir.dat", np.float32).reshape(n, LEADS, LAT, LON, 2)
    Pens = np.fromfile(LPS_CACHE / "P.dat", np.float32).reshape(n, LEADS, LAT, LON)
    acc = np.zeros((len(va), LEADS, LAT, LON, 2), np.float32)
    for path in LPS_CKPTS:
        st = torch.load(path, map_location=DEV)
        m = HeatwaveBustNet(st["arch"], st["in_channels"], st["wide_idx"]).to(DEV)
        m.load_state_dict(st["model_state"]); m.eval()
        with torch.no_grad():
            for s in range(0, len(va), 8):
                x = np.stack([lps_features(rich, Xl, i, st.get("env", False)) for i in va[s:s + 8]])
                acc[s:s + 8] += m(torch.from_numpy(x).to(DEV)).cpu().numpy()
    P = (acc / len(LPS_CKPTS)).astype(np.float16)
    inits = [meta["inits"][i] for i in va]
    obs_m, obs_f = as_obs(Y[va][..., 0]), as_obs(Y[va][..., 1])
    save("lps", p_miss=P[..., 0], p_fa=P[..., 1], obs_miss=obs_m, obs_fa=obs_f, ens_prob=Pens[va].astype(np.float16))
    export_lps_tracks(inits)
    export_gefs_tracks(inits)
    fin = obs_m >= 0
    return {"label": "Low-pressure systems", "kind": "grid", "inits": inits,
            "event": "a low-pressure system (depression or stronger) passing within 300 km",
            "model": "Two GNNs averaged (one adds MJO + seasonal storm climatology), trained on 2000-2015",
            "truth": "ERA5 850 hPa vorticity + MSLP tracker (finds 97% of IBTrACS storms over India)",
            "val_period": f"{inits[0]}..{inits[-1]}",
            # Storm positions are 00 UTC snapshots: lead L verifies at init + L days,
            # where daily rain/Tmax totals verify on init + L - 1.
            "valid_shift": 0,
            "tiers": {"miss": tiers(obs_m[fin], P[..., 0][fin].astype(np.float32), lead_of(obs_m)[fin]),
                      "false_alarm": tiers(obs_f[fin], P[..., 1][fin].astype(np.float32), lead_of(obs_m)[fin])},
            "notable_inits": notable(obs_m, obs_f, inits)}


def baseline_lps() -> None:
    meta, rich, Xl, va = lps_inputs()
    # The wide paths only read tracker channels + lead, so the rain block is zero-filled.
    B = np.zeros((len(va), LEADS, LAT, LON, 2), np.float32)
    for path in LPS_CKPTS:
        st = torch.load(path, map_location="cpu")
        assert min(st["wide_idx"]) >= meta.get("feature_ch", 65), "wide path expected to skip rain features"
        B += np.stack([heatwave_baseline(lps_features(None, Xl, i, st.get("env", False)), [st]) for i in va])
    B /= len(LPS_CKPTS)
    save("lps", b_miss=B[..., 0].astype(np.float16), b_fa=B[..., 1].astype(np.float16))


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


# ----------------------------------------------------------------------------- baselines + skill
RAIN_WIDE = [0, 1, 12, 13, 14]  # rainfall mean, spread, P(>=64.5 / 115.6 / 204.5 mm)


def rain_rows() -> tuple[np.ndarray, int]:
    store = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
    n = len(glob.glob(str(store / "*.nc")))
    return np.arange(int(0.8 * n), n), n


LEAD01 = np.broadcast_to(np.linspace(0, 1, LEADS)[:, None, None, None], (LEADS, LAT, LON, 1))


def rain_baseline_features(x: np.ndarray) -> np.ndarray:
    """[lead, lat, lon, 24] standardized rain features -> ensemble-only baseline inputs."""
    return np.concatenate([x[..., RAIN_WIDE], LEAD01], -1)


def fit_rain_baseline(n_fit: int = 400, n_neg: int = 400_000) -> list[tuple[np.ndarray, float]]:
    """Ensemble-only logistic per direction, fitted on training inits (rain has no wide path)."""
    from sklearn.linear_model import LogisticRegression
    cache = Path(r"D:\sih-data\caches\rain_20yr_24ch_nbhd")
    rows, n = rain_rows()
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 24))
    Y = np.memmap(cache / "Y_dir.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 2))
    rng = np.random.default_rng(0)
    fit = np.sort(rng.choice(rows[0], n_fit, replace=False))
    F = np.stack([rain_baseline_features(np.asarray(X[r])) for r in fit]); Yf = np.asarray(Y[fit])
    coefs = []
    for j in range(2):
        ok = np.isfinite(Yf[..., j]); f, y = F[ok], Yf[..., j][ok]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = np.r_[pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(f[keep], y[keep])
        coefs.append((lr.coef_[0], float(lr.intercept_[0])))
    return coefs


def apply_rain_baseline(x: np.ndarray, coefs) -> np.ndarray:
    f = rain_baseline_features(x)
    return np.stack([1 / (1 + np.exp(-(f @ w + b))) for w, b in coefs], -1)


def baseline_rain() -> None:
    cache = Path(r"D:\sih-data\caches\rain_20yr_24ch_nbhd")
    rows, n = rain_rows()
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 24))
    coefs = fit_rain_baseline()
    B = np.stack([apply_rain_baseline(np.asarray(X[r]), coefs) for r in rows]).astype(np.float16)
    save("rain", b_miss=B[..., 0], b_fa=B[..., 1])


def baseline_heatwave() -> None:
    """The frozen wide path (ensemble-only logistic) of the exported checkpoints, averaged."""
    cache = Path(r"D:\sih-data\caches\heatwave_10yr")
    meta = json.loads((cache / "meta.json").read_text())
    n, C = meta["n"], len(meta["channels"])
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, C))
    va = np.where(~np.asarray(meta["is_train"]))[0]
    ckpts = heatwave_checkpoints("cpu")
    B = np.stack([heatwave_baseline(np.asarray(X[i]), ckpts) for i in va])
    save("heatwave", b_miss=B[..., 0].astype(np.float16), b_fa=B[..., 1].astype(np.float16))


def heatwave_checkpoints(device: str = DEV) -> list[dict]:
    return [torch.load(PROJECT_ROOT / "outputs" / f"heatwave_{a}_fair_s{s}.pt", map_location=device)
            for a in ("gnn", "cnn") for s in (0, 1)]


def heatwave_baseline(x: np.ndarray, ckpts: list[dict]) -> np.ndarray:
    """Average of the checkpoints' frozen wide paths (ensemble-only logistic) for one init."""
    out = np.zeros((*x.shape[:3], 2), np.float32)
    for st in ckpts:
        w = st["model_state"]["wide.weight"].cpu().numpy(); b = st["model_state"]["wide.bias"].cpu().numpy()
        out += 1 / (1 + np.exp(-np.clip(x[..., list(st["wide_idx"])] @ w.T + b, -60, 60)))
    return out / len(ckpts)


def baseline_monsoon() -> None:
    from scripts.train_active_break import AUX, LABELS, TARGETS
    lab = np.load(LABELS)
    rows = np.unique(lab["cache_row"]); pos = {r: i for i, r in enumerate(rows)}
    A = np.zeros((len(rows), LEADS, len(AUX)), np.float32); TR = np.zeros(len(rows), bool)
    for k in range(len(lab["cache_row"])):
        i, L = pos[lab["cache_row"][k]], lab["lead"][k] - 1
        A[i, L] = np.nan_to_num([lab[a][k] for a in AUX]); TR[i] = lab["is_train"][k]
    st = torch.load(PROJECT_ROOT / "outputs" / "active_break_gnn_wide.pt", map_location="cpu")
    w, b = st["model_state"]["wide.weight"].numpy(), st["model_state"]["wide.bias"].numpy()
    lead = np.broadcast_to(np.linspace(0, 1, LEADS)[None, :, None], (len(rows), LEADS, 1))
    B = 1 / (1 + np.exp(-(np.concatenate([A, lead], -1) @ w.T + b)))
    save("monsoon", b=B[~TR].astype(np.float16))
    assert len(TARGETS) == B.shape[-1]


def cross_calibrated(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Isotonic fitted on the first half of the validation inits, applied to the second, and vice versa."""
    from sklearn.isotonic import IsotonicRegression
    late = groups >= np.median(groups)
    out = np.empty(len(y))
    for te in (late, ~late):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s[~te], y[~te])
        out[te] = iso.predict(s[te])
    return out


def skill(y: np.ndarray, p: np.ndarray, b: np.ndarray, groups: np.ndarray) -> dict:
    y = y.astype(np.float64)
    clim = y.mean() * (1 - y.mean())

    def score(s):
        s = s.astype(np.float64)
        c = cross_calibrated(y, s, groups)
        return {"auroc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                "bss": float(1 - np.mean((c - y) ** 2) / clim)}
    return {"model": score(p), "baseline": score(b), "n": int(len(y)), "busts": int(y.sum())}


def add_skill(index: dict) -> None:
    for name, v in index.items():
        a = {q.stem: np.load(q, mmap_mode="r") for q in (OUT / name).glob("*.npy")}
        if v["kind"] == "grid":
            # Score each direction only where that bust is possible: a miss needs GEFS < 20%,
            # a false alarm needs GEFS >= 50%. Elsewhere both scores are trivially right,
            # which would push false-alarm AUROC to 1.0 for any model.
            obs_m, ens = np.asarray(a["obs_miss"]), np.asarray(a["ens_prob"], np.float32)
            g_all = np.broadcast_to(np.arange(len(obs_m))[:, None, None, None], obs_m.shape)
            v["skill"] = {}
            for d, pk, bk, o, ok in (("miss", "p_miss", "b_miss", "obs_miss", ens < 0.2),
                                     ("false_alarm", "p_fa", "b_fa", "obs_fa", ens >= 0.5)):
                fin = (obs_m >= 0) & ok
                v["skill"][d] = skill(np.asarray(a[o])[fin], np.asarray(a[pk])[fin], np.asarray(a[bk])[fin], g_all[fin])
        else:
            obs, p, b = np.asarray(a["obs"]), np.asarray(a["p"]), np.asarray(a["b"])
            g = np.broadcast_to(np.arange(len(obs))[:, None], obs.shape[:2])
            v["skill"] = {}
            for j, t in enumerate(v["targets"]):
                fin = obs[..., j] >= 0
                if obs[..., j][fin].sum() >= 20:
                    v["skill"][t] = skill(obs[..., j][fin], p[..., j][fin], b[..., j][fin], g[fin])


BASELINES = {"heatwave": lambda: baseline_heatwave(), "monsoon": lambda: baseline_monsoon(),
             "lps": lambda: baseline_lps(), "rain": lambda: baseline_rain()}


def build_skill(only: list[str] | None = None) -> None:
    index = json.loads((OUT / "index.json").read_text())
    for name, fn in BASELINES.items():
        if name in index and (only is None or name in only):
            t0 = time.time(); fn(); print(f"{name} baseline | {time.time()-t0:.0f}s", flush=True)
    add_skill(index)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    report(index)


def recompute_tiers() -> None:
    index = json.loads((OUT / "index.json").read_text())
    for name, v in index.items():
        a = {q.stem: np.load(q, mmap_mode="r") for q in (OUT / name).glob("*.npy")}
        if v["kind"] == "grid":
            fin = np.asarray(a["obs_miss"]) >= 0
            v["tiers"] = {d: tiers(np.asarray(a[o])[fin], np.asarray(a[pk])[fin].astype(np.float32),
                                   lead_of(np.asarray(a["obs_miss"]))[fin])
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
        for d, s in v.get("skill", {}).items():
            m, b = s["model"], s["baseline"]
            print(f"  {name}/{d} skill: auroc {m['auroc']:.3f} vs {b['auroc']:.3f} | pr-auc {m['pr_auc']:.3f} vs "
                  f"{b['pr_auc']:.3f} | bss {m['bss']:.3f} vs {b['bss']:.3f}  (model vs GEFS-only)", flush=True)


EXPORTS = {"heatwave": lambda: export_heatwave(), "lps": lambda: export_lps(),
           "monsoon": lambda: export_monsoon(), "rain": lambda: export_rain()}


def main():
    import sys
    if "--tiers-only" in sys.argv:
        recompute_tiers(); return
    if "--skill-only" in sys.argv:
        build_skill(); return
    if "--gefs-tracks-only" in sys.argv:
        export_gefs_tracks(json.loads((OUT / "index.json").read_text())["lps"]["inits"]); return
    # --only lps,heatwave re-exports just those hazards into the existing index
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    t0 = time.time()
    index = json.loads((OUT / "index.json").read_text()) if only and (OUT / "index.json").exists() else {}
    for name, fn in EXPORTS.items():
        if only and name not in only:
            continue
        index[name] = fn()
        print(f"{name}: {len(index[name]['inits'])} inits | {time.time()-t0:.0f}s", flush=True)
        (OUT / "index.json").write_text(json.dumps(index, indent=1))
    lats, lons = canonical_centres(0.5)
    np.save(OUT / "lats.npy", lats); np.save(OUT / "lons.npy", lons)
    build_skill(only)

if __name__ == "__main__":
    main()
