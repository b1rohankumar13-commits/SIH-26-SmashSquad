"""Evaluate several GNN checkpoints on one common validation window.

Each checkpoint is scored with its OWN standardizer: the shared cache stores features
standardized with the cache's statistics, so this recovers the raw features and
re-standardizes per model. That isolates the effect of training data / model from both
the evaluation set and the normalization, making PR-AUC comparable across checkpoints
(e.g. a 5-year vs a 10-year model on the same 2008-2009 dates).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.features.build_gridded_sequences import N_CHANNELS
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.model import GridGraphNet

ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
CACHE = Path(os.environ.get("SIH_CACHE_DIR") or ROOT / "data" / "interim" / "_graphnet_cache")


def sorted_files(years):
    files = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    return sorted(files)


def pr_auc(y, p):
    return average_precision_score(y, p) if y.min() != y.max() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=str, default="2000-2009", help="Years the shared cache was built on.")
    parser.add_argument("--val-min-year", type=int, default=2008,
                        help="Evaluate on init dates in this year or later (a window both models can be scored on).")
    parser.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths to compare.")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    start, end = (args.years.split("-") + [args.years])[:2]
    files = sorted_files(range(int(start), int(end) + 1))
    n = len(files)
    years = np.array([int(Path(f).stem[:4]) for f in files])
    val = np.where(years >= args.val_min_year)[0]

    grids = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n, 9, 66, 70, N_CHANNELS))
    labels = np.memmap(CACHE / "Y.dat", np.float32, "r", shape=(n, 9, 66, 70, 1))
    z = np.load(CACHE / "standardizer.npz")
    cache_mean = z["mean"].astype(np.float32).reshape(1, 1, 1, 1, N_CHANNELS)
    cache_std = z["std"].astype(np.float32).reshape(1, 1, 1, 1, N_CHANNELS)

    y_val = np.asarray(labels[val])
    mask = np.isfinite(y_val).reshape(-1)
    y = y_val.reshape(-1)[mask]
    lead = np.broadcast_to(np.arange(1, 10)[None, :, None, None, None], y_val.shape).reshape(-1)[mask]
    print(f"common window: init year >= {args.val_min_year} | {len(val)} dates | "
          f"{int(y.sum())} bust cells / {len(y)} ({100 * y.mean():.3f}%)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt_path in args.ckpts:
        ckpt = torch.load(ROOT / ckpt_path, map_location="cpu", weights_only=False)
        cc = int(ckpt.get("in_channels", N_CHANNELS))
        model = GridGraphNet(build_grid_graph(), in_channels=cc, hidden=int(ckpt["hidden"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        cmean = np.asarray(ckpt["standardizer_mean"], np.float32).reshape(1, 1, 1, 1, cc)
        cstd = np.asarray(ckpt["standardizer_std"], np.float32).reshape(1, 1, 1, 1, cc)

        preds = []
        with torch.no_grad():
            for s in range(0, len(val), args.batch):
                idx = val[s:s + args.batch]
                raw = np.asarray(grids[idx]) * cache_std + cache_mean       # undo cache standardizer
                x = torch.from_numpy(((raw - cmean) / cstd).astype(np.float32)).to(device)  # this model's
                preds.append(model(x).cpu().numpy())
        p = np.concatenate(preds).reshape(-1)[mask]

        overall = pr_auc(y, p)
        per_lead = [pr_auc(y[lead == L], p[lead == L]) for L in range(1, 10)]
        trained = f"{ckpt.get('years', ['?'])[0]}-{ckpt.get('years', ['?'])[-1]}"
        print(f"\n{ckpt_path}  (trained {trained})", flush=True)
        print(f"  overall PR-AUC {overall:.4f}", flush=True)
        print("  by lead: " + " ".join(f"{L}:{v:.3f}" for L, v in zip(range(1, 10), per_lead)), flush=True)


if __name__ == "__main__":
    main()
