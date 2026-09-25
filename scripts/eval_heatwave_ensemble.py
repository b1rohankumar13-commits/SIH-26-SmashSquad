"""Combine heatwave-bust checkpoints (GNN + CNN) on the validation split.

No training: loads the fair-rerun checkpoints, predicts P(miss), P(false alarm) on val,
and scores single models, same-arch seed averages, GNN+CNN probability averages, and a
per-head pick (CNN miss head + GNN false-alarm head).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from scripts.train_heatwave import HeatwaveBustNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def predict(ckpt: Path, X: np.ndarray, idx: np.ndarray, C: int, dev: str, batch: int = 8) -> np.ndarray:
    state = torch.load(ckpt, map_location=dev)
    model = HeatwaveBustNet(state["arch"], C, state["wide_idx"]).to(dev)
    model.load_state_dict(state["model_state"]); model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            out.append(model(torch.from_numpy(X[idx[s:s + batch]]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\heatwave_10yr")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W, C = meta["n"], meta["leads"], meta["lat"], meta["lon"], len(meta["channels"])
    X = np.fromfile(cache / "X.dat", np.float32).reshape(n, L, H, W, C)
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)
    va = np.where(~np.asarray(meta["is_train"]))[0]
    yv = Y[va]; fin = np.isfinite(yv[..., 0])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    preds = {}
    for arch in ("gnn", "cnn"):
        for s in seeds:
            preds[(arch, s)] = predict(PROJECT_ROOT / "outputs" / f"heatwave_{arch}_fair_s{s}.pt", X, va, C, dev)
            print(f"predicted {arch} s{s}", flush=True)

    def score(p):
        a = [float(average_precision_score(yv[..., j][fin], p[..., j][fin])) for j in range(2)]
        return {"miss": a[0], "false_alarm": a[1], "mean": (a[0] + a[1]) / 2}

    rows = {}
    for s in seeds:
        rows[f"gnn s{s}"] = score(preds[("gnn", s)])
        rows[f"cnn s{s}"] = score(preds[("cnn", s)])
        rows[f"gnn+cnn avg s{s}"] = score((preds[("gnn", s)] + preds[("cnn", s)]) / 2)
        pick = np.stack([preds[("cnn", s)][..., 0], preds[("gnn", s)][..., 1]], -1)
        rows[f"pick cnn-miss/gnn-fa s{s}"] = score(pick)
    g = np.mean([preds[("gnn", s)] for s in seeds], 0); c = np.mean([preds[("cnn", s)] for s in seeds], 0)
    rows["gnn seed-avg"] = score(g)
    rows["cnn seed-avg"] = score(c)
    rows["gnn+cnn all-avg"] = score((g + c) / 2)
    rows["pick seed-avg (cnn miss, gnn fa)"] = score(np.stack([c[..., 0], g[..., 1]], -1))

    print(f"\n{'combination':34s} {'miss':>7s} {'fa':>7s} {'mean':>7s}")
    for k, v in rows.items():
        print(f"{k:34s} {v['miss']:7.4f} {v['false_alarm']:7.4f} {v['mean']:7.4f}")
    (PROJECT_ROOT / "outputs" / "heatwave_ensemble_eval.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
