"""Build engineered tabular features, train + save XGBoost, fuse with the GNN, evaluate."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import xgboost
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.features.build_gridded_sequences import N_CHANNELS
from src.features.build_tabular_dataset import FEATURE_NAMES, tabular_features
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.model import GridGraphNet

ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
CACHE = Path(os.environ.get("SIH_CACHE_DIR") or ROOT / "data" / "interim" / "_graphnet_cache")
F = len(FEATURE_NAMES)


def parse_years(spec: str) -> range:
    start, end = (spec.split("-") + [spec])[:2]
    return range(int(start), int(end) + 1)


def sorted_files(years):
    files = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    return sorted(files)


def build_tabular_cache(files):
    path = CACHE / "T.dat"
    expected = len(files) * 9 * 66 * 70 * F * 4
    if path.exists() and path.stat().st_size == expected:
        print("reusing tabular cache", flush=True)
        return np.memmap(path, np.float32, "r", shape=(len(files), 9, 66, 70, F))
    print(f"building tabular cache ({len(files)} dates, {F} features) ...", flush=True)
    table = np.memmap(path, np.float32, "w+", shape=(len(files), 9, 66, 70, F))
    for i, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            table[i] = tabular_features(ds)
        if (i + 1) % 200 == 0:
            table.flush()
            print(f"tabular {i + 1}/{len(files)}", flush=True)
    table.flush()
    return table


def gnn_predict(model, X, idx, device, batch=8):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            x = torch.from_numpy(np.asarray(X[idx[s:s + batch]])).to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def score(y, p):
    if y.min() == y.max():
        return float("nan"), float("nan")
    return average_precision_score(y, p), brier_score_loss(y, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-ckpt", type=str, default="outputs/graphnet_pilot_2000_2004_reg.pt")
    parser.add_argument("--xgb-model", type=str, default="outputs/xgboost_2000_2004.json")
    parser.add_argument("--reuse-xgb", action="store_true", help="Load the saved XGBoost model instead of retraining.")
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--xgb-depth", type=int, default=6)
    parser.add_argument("--neg-ratio", type=int, default=15, help="Negatives per positive when subsampling.")
    parser.add_argument("--fusion", choices=["global", "interact", "lead"], default="global",
                        help="global: one plain blender (best on pooled PR-AUC); interact: one blender "
                             "with lead interactions; lead: a separate blender per lead.")
    parser.add_argument("--years", type=str, default="2000-2004", help="Year range, e.g. 2000-2009.")
    args = parser.parse_args()

    files = sorted_files(parse_years(args.years))
    table = build_tabular_cache(files)
    n = len(files)
    labels = np.memmap(CACHE / "Y.dat", np.float32, "r", shape=(n, 9, 66, 70, 1))
    grids = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n, 9, 66, 70, N_CHANNELS))

    ckpt = torch.load(ROOT / args.gnn_ckpt, map_location="cpu", weights_only=False)
    n_train = int(ckpt["n_train"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GridGraphNet(build_grid_graph(), in_channels=int(ckpt.get("in_channels", N_CHANNELS)),
                         hidden=int(ckpt["hidden"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    # XGBoost on engineered features (train dates); reuse a saved model or train + save.
    xgb_path = ROOT / args.xgb_model
    if args.reuse_xgb and xgb_path.exists():
        clf = xgboost.XGBClassifier()
        clf.load_model(str(xgb_path))
        print(f"reusing XGBoost model {args.xgb_model}", flush=True)
    else:
        x_tr = np.asarray(table[:n_train]).reshape(-1, F)
        y_tr = np.asarray(labels[:n_train]).reshape(-1)
        valid = np.isfinite(y_tr)
        x_tr, y_tr = x_tr[valid], y_tr[valid]
        pos = np.where(y_tr == 1)[0]
        rng = np.random.default_rng(0)
        neg = rng.choice(np.where(y_tr == 0)[0],
                         size=min((y_tr == 0).sum(), len(pos) * args.neg_ratio), replace=False)
        sel = np.concatenate([pos, neg])
        clf = xgboost.XGBClassifier(n_estimators=args.xgb_estimators, max_depth=args.xgb_depth,
                                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="aucpr",
                                    scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1), n_jobs=-1)
        clf.fit(x_tr[sel], y_tr[sel])
        xgb_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(str(xgb_path))
        top = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda kv: -kv[1])[:8]
        print("top XGB features:", [(n_, round(float(v), 3)) for n_, v in top], flush=True)

    # Validation predictions from both branches.
    val = np.arange(n_train, n)
    gnn = gnn_predict(model, grids, val, device)
    y_val = np.asarray(labels[n_train:])
    xgb_val = clf.predict_proba(np.asarray(table[n_train:]).reshape(-1, F))[:, 1].reshape(y_val.shape)

    mask = np.isfinite(y_val).reshape(-1)
    g = gnn.reshape(-1)[mask]
    xb = xgb_val.reshape(-1)[mask]
    y = y_val.reshape(-1)[mask]
    lead = np.broadcast_to(np.arange(1, 10)[None, :, None, None, None], y_val.shape).reshape(-1)[mask]

    order = np.random.default_rng(0).permutation(len(y))
    fit, ev = order[:len(y) // 2], order[len(y) // 2:]
    ell = (lead - 5.0) / 4.0  # lead centred to ~[-1, 1]

    # A plain global blender applies one GNN weight to every lead, so it cannot discount the
    # GNN where its skill has decayed. The lead interactions (g*ell, xb*ell) let one blender
    # vary that weight with lead while keeping a single, cross-lead-comparable score.
    fused = np.empty(len(y))
    if args.fusion == "lead":
        base = np.column_stack([g, xb, np.abs(g - xb)])
        for L in range(1, 10):
            here = lead == L
            fit_L = fit[lead[fit] == L]
            if len(fit_L) and y[fit_L].max() > 0:
                blender = LogisticRegression(max_iter=1000).fit(base[fit_L], y[fit_L])
                fused[here] = blender.predict_proba(base[here])[:, 1]
            else:
                fused[here] = xb[here]  # too few positives to blend; fall back to XGBoost
    else:
        if args.fusion == "interact":
            feats = np.column_stack([g, xb, np.abs(g - xb), ell, g * ell, xb * ell])
        else:
            feats = np.column_stack([g, xb, np.abs(g - xb)])
        fused = LogisticRegression(max_iter=1000).fit(feats[fit], y[fit]).predict_proba(feats)[:, 1]
    fused_cal = IsotonicRegression(out_of_bounds="clip").fit(fused[fit], y[fit]).predict(fused)

    print("\n=== held-out val metrics (PR-AUC / Brier) ===", flush=True)
    for name, p in [("GNN (spatial)", g), ("XGBoost (engineered)", xb),
                    ("Fused", fused), ("Fused + calibrated", fused_cal)]:
        ap, br = score(y[ev], p[ev])
        print(f"  {name:22s}  PR-AUC {ap:.4f}   Brier {br:.5f}", flush=True)

    print("\n=== PR-AUC by lead day (GNN / XGB / fused) ===", flush=True)
    for L in range(1, 10):
        m = lead[ev] == L
        if m.sum() and y[ev][m].max() > 0:
            print(f"  lead {L}: {score(y[ev][m], g[ev][m])[0]:.3f} / "
                  f"{score(y[ev][m], xb[ev][m])[0]:.3f} / {score(y[ev][m], fused[ev][m])[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
