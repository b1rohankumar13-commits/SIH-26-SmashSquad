"""Evaluate the spatial model, train an XGBoost tabular branch, fuse, and calibrate.

Runs on the held-out validation dates in the feature cache. The GNN and XGBoost never
see the validation dates, so their val predictions are out-of-sample; the fusion and
calibration are fit on one half of the val cells and scored on the other half.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xgboost
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres
from src.models.graphnet.model import GridGraphNet

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "interim" / "_graphnet_cache"


def load_cache():
    x_path, y_path = CACHE / "X.dat", CACHE / "Y.dat"
    n = x_path.stat().st_size // (9 * 66 * 70 * 12 * 4)
    features = np.memmap(x_path, np.float32, "r", shape=(n, 9, 66, 70, 12))
    labels = np.memmap(y_path, np.float32, "r", shape=(n, 9, 66, 70, 1))
    return features, labels, n


def gnn_predict(model, features, idx, device, batch=8):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            b = idx[start:start + batch]
            x = torch.from_numpy(np.asarray(features[b])).to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def score(y, p):
    if y.min() == y.max():
        return float("nan"), float("nan")
    return average_precision_score(y, p), brier_score_loss(y, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="outputs/graphnet_pilot_2000_2004_reg.pt")
    args = parser.parse_args()

    ckpt = torch.load(ROOT / args.model, map_location="cpu", weights_only=False)
    n_train = int(ckpt["n_train"])
    features, labels, n = load_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GridGraphNet(build_grid_graph(), in_channels=12, hidden=int(ckpt["hidden"])).to(device)
    model.load_state_dict(ckpt["model_state"])

    val_idx = np.arange(n_train, n)
    gnn = gnn_predict(model, features, val_idx, device)
    y_val = np.asarray(labels[n_train:])
    print(f"val dates {len(val_idx)} | GNN checkpoint val PR-AUC {ckpt.get('val_pr_auc'):.4f}", flush=True)

    # XGBoost on per-cell features (train dates), negatives subsampled 15:1.
    x_tr = np.asarray(features[:n_train]).reshape(-1, 12)
    y_tr = np.asarray(labels[:n_train]).reshape(-1)
    valid = np.isfinite(y_tr)
    x_tr, y_tr = x_tr[valid], y_tr[valid]
    pos = np.where(y_tr == 1)[0]
    rng = np.random.default_rng(0)
    neg = rng.choice(np.where(y_tr == 0)[0], size=min((y_tr == 0).sum(), len(pos) * 15), replace=False)
    sel = np.concatenate([pos, neg])
    clf = xgboost.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                                scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1), n_jobs=-1)
    clf.fit(x_tr[sel], y_tr[sel])
    xgb_val = clf.predict_proba(np.asarray(features[n_train:]).reshape(-1, 12))[:, 1].reshape(y_val.shape)

    # Flatten val cells and keep the verified ones.
    mask = np.isfinite(y_val).reshape(-1)
    g = gnn.reshape(-1)[mask]
    xb = xgb_val.reshape(-1)[mask]
    y = y_val.reshape(-1)[mask]
    lead = np.broadcast_to(np.arange(1, 10)[None, :, None, None, None], y_val.shape).reshape(-1)[mask]
    lats, _ = canonical_centres()
    lat = np.broadcast_to(lats[None, None, :, None, None], y_val.shape).reshape(-1)[mask]

    # Fit fusion + calibration on half the val cells, score on the other half.
    order = np.random.default_rng(0).permutation(len(y))
    fit, ev = order[:len(y) // 2], order[len(y) // 2:]

    stack = np.column_stack([g, xb, np.abs(g - xb)])
    fusion = LogisticRegression(max_iter=1000).fit(stack[fit], y[fit])
    fused = fusion.predict_proba(stack)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(fused[fit], y[fit])
    fused_cal = iso.predict(fused)

    print("\n=== held-out val metrics (PR-AUC / Brier) ===")
    for name, p in [("GNN (spatial)", g), ("XGBoost (tabular)", xb),
                    ("Fused", fused), ("Fused + calibrated", fused_cal)]:
        ap, br = score(y[ev], p[ev])
        print(f"  {name:20s}  PR-AUC {ap:.4f}   Brier {br:.5f}")

    print("\n=== PR-AUC by lead day ===")
    for L in range(1, 10):
        m = lead[ev] == L
        if m.sum() and y[ev][m].max() > 0:
            print(f"  lead {L}: GNN {score(y[ev][m], g[ev][m])[0]:.3f}  fused {score(y[ev][m], fused[ev][m])[0]:.3f}")

    print("\n=== PR-AUC by latitude band ===")
    for name, lo, hi in [("South (<15N)", 0, 15), ("Central (15-25N)", 15, 25), ("North (>25N)", 25, 90)]:
        m = (lat[ev] >= lo) & (lat[ev] < hi)
        if m.sum() and y[ev][m].max() > 0:
            print(f"  {name:16s}: GNN {score(y[ev][m], g[ev][m])[0]:.3f}  fused {score(y[ev][m], fused[ev][m])[0]:.3f} "
                  f" (cells {int(m.sum())}, busts {int(y[ev][m].sum())})")


if __name__ == "__main__":
    main()
