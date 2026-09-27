"""Score LPS strike-bust checkpoints on a window of years (PR-AUC per direction and lead band).

Used to separate "more training years" from "different input features": the 2000-2007 and
2000-2015 models trained on the same 20-year cache are both scored on 2016-2019, which
neither saw. The frozen wide path of each checkpoint gives its GEFS-only baseline.

    python -m scripts.eval_lps_window outputs/lps_gnn20.pt outputs/lps_gnn20_8yr.pt --years 2016-2019

--average also scores the mean of all checkpoints' probabilities (a simple model ensemble).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from scripts.train_heatwave import TARGETS, HeatwaveBustNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = {"D1-3": (0, 3), "D4-6": (3, 6), "D7-9": (6, 9)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\lps_20yr")
    ap.add_argument("--years", default="2016-2019")
    ap.add_argument("--average", action="store_true", help="Also score the mean of the checkpoints.")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W, fc = meta["n"], meta["leads"], meta["lat"], meta["lon"], meta.get("feature_ch", 65)
    years = np.array([int(s[:4]) for s in meta["inits"]])
    y0, y1 = (int(y) for y in args.years.split("-"))
    idx = np.where((years >= y0) & (years <= y1))[0]
    rich = np.memmap(Path(meta["rich_cache"]) / "X.dat", np.float32, "r", shape=(n, L, H, W, fc))
    Xl = np.memmap(cache / "X_lps.dat", np.float32, "r", shape=(n, L, H, W, len(meta["channels"])))
    lag_names = meta.get("lag_channels", [])
    Xg = np.memmap(cache / "X_lag.dat", np.float32, "r", shape=(n, L, H, W, len(lag_names))) if lag_names else None
    env_names = meta.get("env_channels", [])
    Xe = np.memmap(cache / "X_env.dat", np.float32, "r", shape=(n, L, H, W, len(env_names))) if env_names else None
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)[idx]
    lead01 = np.broadcast_to(np.linspace(0, 1, L, dtype=np.float32)[:, None, None, None], (L, H, W, 1))
    print(f"{len(idx)} inits {args.years}", flush=True)

    def report(P: np.ndarray, B: np.ndarray) -> None:
        for j, t in enumerate(TARGETS):
            row = []
            for band, (a, z) in {"all": (0, L), **BANDS}.items():
                y = Y[:, a:z, ..., j]; ok = np.isfinite(y)
                row.append(f"{band} {average_precision_score(y[ok], P[:, a:z, ..., j][ok]):.4f}"
                           f" ({average_precision_score(y[ok], B[:, a:z, ..., j][ok]):.4f})")
            print(f"  {t:11s} PR-AUC model (GEFS-only): " + " | ".join(row), flush=True)

    acc = None
    for path in args.checkpoints:
        st = torch.load(path, map_location=DEV)
        m = HeatwaveBustNet(st["arch"], st["in_channels"], st["wide_idx"]).to(DEV)
        m.load_state_dict(st["model_state"]); m.eval()
        P = np.empty((len(idx), L, H, W, 2), np.float32); B = np.empty_like(P)
        w = m.wide.weight.detach().cpu().numpy(); b = m.wide.bias.detach().cpu().numpy()
        with torch.no_grad():
            for s in range(0, len(idx), 8):
                rows = idx[s:s + 8]
                parts = [np.asarray(rich[rows]), np.asarray(Xl[rows])]
                if st.get("lag"):
                    parts.append(np.asarray(Xg[rows]))
                if st.get("env"):
                    parts.append(np.asarray(Xe[rows]))
                x = np.concatenate(parts + [np.broadcast_to(lead01, (len(rows), L, H, W, 1))], -1)
                P[s:s + 8] = m(torch.from_numpy(x).to(DEV)).cpu().numpy()
                B[s:s + 8] = 1 / (1 + np.exp(-np.clip(x[..., list(st["wide_idx"])] @ w.T + b, -60, 60)))
        print(f"\n{path.name} (trained to {st.get('train_last', '?')}, best epoch {st['epoch']}, "
              f"lag {bool(st.get('lag'))}, env {bool(st.get('env'))})", flush=True)
        report(P, B)
        acc = P if acc is None else acc + P
    if args.average and len(args.checkpoints) > 1:
        print(f"\naverage of {len(args.checkpoints)} checkpoints", flush=True)
        report(acc / len(args.checkpoints), B)


if __name__ == "__main__":
    main()
