"""Train the directional rain-bust GNN: two heads, P(miss) and P(false-alarm).

Reuses the 20yr feature cache (X.dat, 19 channels) and the 2-channel Y_dir.dat.
Shared GraphSAGE+TCN encoder, 2-output head. Reports per-direction PR-AUC.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.model import GridGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAT, LON, LEADS, CH = 66, 70, 9, 19


def to_batch(X, Y, idx, device):
    x = torch.from_numpy(np.asarray(X[idx])).to(device)
    yr = torch.from_numpy(np.asarray(Y[idx]))
    mask = torch.isfinite(yr).float().to(device)
    y = torch.nan_to_num(yr, nan=0.0).to(device)
    return x, y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("SIH_CACHE_DIR"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="outputs/graphnet_20yr_directional.pt")
    ap.add_argument("--in-channels", type=int, default=CH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=["miss", "mean"], default="miss",
                    help="Checkpoint/early-stop metric: miss PR-AUC (default) or mean(miss, fa).")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    ch = args.in_channels
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    xb = (cache / "X.dat").stat().st_size
    n = xb // (LEADS * LAT * LON * ch * 4)
    X = np.memmap(cache / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, ch))
    Y = np.memmap(cache / "Y_dir.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 2))
    n_train = int(0.8 * n)
    print(f"n={n} | train {n_train} val {n-n_train} | 2 heads (miss, false_alarm)", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GridGraphNet(build_grid_graph(), in_channels=ch, hidden=args.hidden, out_channels=2).to(dev)

    ytr = np.asarray(Y[:n_train])
    finite = np.isfinite(ytr)
    pos = np.nansum(ytr == 1, axis=(0, 1, 2, 3))          # [miss, fa]
    valid = finite.sum(axis=(0, 1, 2, 3))
    pw = torch.tensor([(valid[c] - pos[c]) / max(pos[c], 1) for c in range(2)], device=dev).float()
    print(f"miss base {100*pos[0]/valid[0]:.3f}% (pw {pw[0]:.0f}) | "
          f"fa base {100*pos[1]/valid[1]:.3f}% (pw {pw[1]:.0f})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_idx, va_idx = np.arange(n_train), np.arange(n_train, n)
    best, best_ep, since = -1.0, 0, 0
    hist = {"config": vars(args), "n": int(n), "n_train": n_train, "epochs": []}
    hist_path = Path(args.out).with_suffix(".history.json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t = time.time(); model.train()
        for s in range(0, n_train, args.batch):
            idx = np.sort(tr_idx[np.random.permutation(n_train)][s:s + args.batch])
            x, y, m = to_batch(X, Y, idx, dev)
            opt.zero_grad()
            bce = F.binary_cross_entropy(model(x), y, reduction="none")
            w = torch.where(y > 0.5, pw, torch.ones_like(y))
            loss = (bce * w * m).sum() / m.sum().clamp(min=1.0)
            loss.backward(); opt.step()

        model.eval()
        preds = [[], []]; tgts = [[], []]
        with torch.no_grad():
            for s in range(0, len(va_idx), args.batch):
                idx = va_idx[s:s + args.batch]
                x, y, m = to_batch(X, Y, idx, dev)
                out = model(x).cpu().numpy()
                yv = y.cpu().numpy(); mv = m.cpu().numpy().astype(bool)
                for c in range(2):
                    keep = mv[..., c].reshape(-1)
                    preds[c].append(out[..., c].reshape(-1)[keep])
                    tgts[c].append(yv[..., c].reshape(-1)[keep])
        aucs = []
        for c in range(2):
            p, tt = np.concatenate(preds[c]), np.concatenate(tgts[c])
            aucs.append(average_precision_score(tt, p) if tt.min() != tt.max() else float("nan"))
        score = aucs[0] if args.select == "miss" else np.nanmean(aucs)
        print(f"epoch {ep:2d} | {time.time()-t:5.1f}s | PR-AUC miss {aucs[0]:.4f} | fa {aucs[1]:.4f} "
              f"| mean {score:.4f}", flush=True)

        hist["epochs"].append({"epoch": ep, "seconds": round(time.time() - t, 1),
                              "pr_auc_miss": float(aucs[0]), "pr_auc_fa": float(aucs[1])})
        hist_path.write_text(json.dumps(hist, indent=2))
        if np.isfinite(score) and score > best:
            best, best_ep, since = score, ep, 0
            torch.save({"model_state": model.state_dict(), "in_channels": ch, "hidden": args.hidden,
                        "out_channels": 2, "heads": ["miss", "false_alarm"],
                        "pr_auc_miss": aucs[0], "pr_auc_fa": aucs[1], "epoch": ep}, args.out)
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at {ep}", flush=True); break
    print(f"\nBest mean PR-AUC {best:.4f} @ epoch {best_ep} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
