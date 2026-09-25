"""Active/break regime-bust bake-off: GNN vs CNN on the 29-variable (65-channel) cache.

Same labels (data/processed/active_break_labels.npz), same inputs, same head, same
training loop - only the spatial encoder differs:
  gnn: GraphSAGE + TCN over the grid graph (GridGraphNet encoder)
  cnn: 2-D conv stack per lead + the same temporal convolution
Both pool the encoder output over the monsoon core zone (mean + max), then a head
that also sees the ensemble's own regime signal (p_active, p_break, mean anomaly,
spread) predicts 4 bust types per lead: miss/false-alarm x break/active.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn

from src.detection.monsoon_phase import mcz_mask
from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres
from src.models.graphnet.model import GridGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(r"D:\sih-data\caches\rain_rich_10yr")
LABELS = PROJECT_ROOT / "data" / "processed" / "active_break_labels.npz"
TARGETS = ("miss_break", "fa_break", "miss_active", "fa_active")
AUX = ("p_active", "p_break", "ens_mean_anom", "ens_spread")
LEADS, LAT, LON, CH = 9, 66, 70, 65


class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.tcn = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2))

    def forward(self, x):  # [B, L, H, W, C] -> [B, L, H, W, hidden]
        b, l, h, w, c = x.shape
        z = self.spatial(x.reshape(b * l, h, w, c).permute(0, 3, 1, 2))  # [B*L, hid, H, W]
        hid = z.shape[1]
        t = z.reshape(b, l, hid, h * w).permute(0, 3, 2, 1).reshape(b * h * w, hid, l)
        for conv in self.tcn:
            t = t + torch.relu(conv(t))
        return t.reshape(b, h, w, hid, l).permute(0, 4, 1, 2, 3)


class RegimeBustNet(nn.Module):
    """head='plain': MLP([pooled map features, aux]).
    head='wide': wide-and-deep - a direct linear path from the ensemble signals (+ lead)
    to the output, plus a map-feature MLP whose last layer starts at zero, so the model
    starts as the ensemble-only logistic baseline and the encoder learns a correction."""

    def __init__(self, arch: str, hidden: int = 64, head: str = "plain"):
        super().__init__()
        self.head_type = head
        if arch == "gnn":
            self.gnn = GridGraphNet(build_grid_graph(), in_channels=CH, hidden=hidden)
            self.encode = self.gnn.features
        else:
            self.cnn = CNNEncoder(CH, hidden)
            self.encode = self.cnn
        lats, lons = canonical_centres(0.5)
        self.register_buffer("mask", torch.from_numpy(mcz_mask(lats, lons)))
        if head == "plain":
            self.head = nn.Sequential(nn.Linear(2 * hidden + len(AUX), 64), nn.ReLU(), nn.Linear(64, len(TARGETS)))
        else:
            self.wide = nn.Linear(len(AUX) + 1, len(TARGETS))  # ensemble signals + lead
            self.deep = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, len(TARGETS)))
            nn.init.zeros_(self.deep[-1].weight); nn.init.zeros_(self.deep[-1].bias)

    def forward(self, x, aux):  # aux [B, L, len(AUX)] -> probs [B, L, 4]
        z = self.encode(x)[:, :, self.mask]  # [B, L, n_mcz, hidden]
        pooled = torch.cat([z.mean(2), z.amax(2)], dim=-1)
        if self.head_type == "plain":
            return torch.sigmoid(self.head(torch.cat([pooled, aux], dim=-1)))
        lead = torch.linspace(0, 1, aux.shape[1], device=aux.device)[None, :, None].expand(aux.shape[0], -1, 1)
        return torch.sigmoid(self.wide(torch.cat([aux, lead], dim=-1)) + self.deep(pooled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gnn", "cnn"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--head", choices=["plain", "wide"], default="plain")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    lab = np.load(LABELS)
    n_cache = (CACHE / "X.dat").stat().st_size // (LEADS * LAT * LON * CH * 4)
    X = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n_cache, LEADS, LAT, LON, CH))

    # regroup (init, lead) rows into per-init tensors [L, ...] with a lead mask
    rows = np.unique(lab["cache_row"])
    pos = {r: i for i, r in enumerate(rows)}
    Y = np.zeros((len(rows), LEADS, len(TARGETS)), np.float32)
    A = np.zeros((len(rows), LEADS, len(AUX)), np.float32)
    M = np.zeros((len(rows), LEADS), np.float32)
    TR = np.zeros(len(rows), bool)
    for k in range(len(lab["cache_row"])):
        i, L = pos[lab["cache_row"][k]], lab["lead"][k] - 1
        Y[i, L] = [lab[t][k] for t in TARGETS]
        A[i, L] = np.nan_to_num([lab[a][k] for a in AUX])
        M[i, L] = 1.0
        TR[i] = lab["is_train"][k]
    tr_idx, va_idx = np.where(TR)[0], np.where(~TR)[0]
    print(f"[{args.arch}] inits train {len(tr_idx)} val {len(va_idx)} | "
          f"positives train {Y[tr_idx].sum((0, 1)).astype(int).tolist()} "
          f"val {Y[va_idx].sum((0, 1)).astype(int).tolist()} {list(TARGETS)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegimeBustNet(args.arch, head=args.head).to(dev)
    if args.head == "wide":
        # Initialise the wide path as the ensemble-only logistic baseline (same inputs:
        # aux + lead scaled 0-1), fitted per target on training rows, then freeze it.
        # Epoch 1 then scores the baseline; the map encoder can only add a correction.
        from sklearn.linear_model import LogisticRegression
        lead01 = np.broadcast_to(np.linspace(0, 1, LEADS)[None, :, None], (len(rows), LEADS, 1))
        feats = np.concatenate([A, lead01], -1)[tr_idx][M[tr_idx] > 0]
        with torch.no_grad():
            for j, name in enumerate(TARGETS):
                yj = Y[tr_idx][..., j][M[tr_idx] > 0]
                lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats, yj)
                model.wide.weight[j] = torch.tensor(lr.coef_[0], dtype=torch.float32, device=model.wide.weight.device)
                model.wide.bias[j] = float(lr.intercept_[0])
        for prm in model.wide.parameters():
            prm.requires_grad = False
        print("wide path initialised from the logistic baseline and frozen", flush=True)
    pos_n = (Y[tr_idx] * M[tr_idx, :, None]).sum((0, 1))
    neg_n = M[tr_idx].sum() - pos_n
    pw = torch.tensor(neg_n / np.maximum(pos_n, 1), device=dev, dtype=torch.float32)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=1e-3)
    print(f"params {sum(p.numel() for p in model.parameters()):,} | pos_weight {pw.round().tolist()}", flush=True)

    def batch(idx):
        x = torch.from_numpy(np.asarray(X[rows[idx]])).to(dev)
        return (x, torch.from_numpy(A[idx]).to(dev), torch.from_numpy(Y[idx]).to(dev),
                torch.from_numpy(M[idx]).to(dev))

    tag = args.arch if args.head == "plain" else f"{args.arch}_wide"
    out_path = PROJECT_ROOT / "outputs" / f"active_break_{tag}.pt"
    history = {"arch": args.arch, "config": vars(args), "cache": str(CACHE), "labels": str(LABELS),
               "n_train_inits": int(len(tr_idx)), "n_val_inits": int(len(va_idx)),
               "val_positives": dict(zip(TARGETS, Y[va_idx].sum((0, 1)).astype(int).tolist())),
               "params": int(sum(p.numel() for p in model.parameters())), "epochs": []}
    hist_path = PROJECT_ROOT / "outputs" / f"active_break_{tag}_history.json"
    best, best_ep, since = -1.0, 0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        order = np.random.permutation(tr_idx)
        for s in range(0, len(order), args.batch):
            x, a, y, m = batch(np.sort(order[s:s + args.batch]))
            p = model(x, a)
            w = torch.where(y > 0.5, pw, torch.ones_like(y))
            loss = (F.binary_cross_entropy(p, y, reduction="none") * w * m[..., None]).sum() / (
                m.sum() * len(TARGETS)).clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); P, T, MM = [], [], []
        with torch.no_grad():
            for s in range(0, len(va_idx), args.batch):
                x, a, y, m = batch(va_idx[s:s + args.batch])
                P.append(model(x, a).cpu().numpy()); T.append(y.cpu().numpy()); MM.append(m.cpu().numpy())
        P, T, MM = np.concatenate(P), np.concatenate(T), np.concatenate(MM).astype(bool)
        aucs = {}
        for j, name in enumerate(TARGETS):
            t, p = T[..., j][MM], P[..., j][MM]
            aucs[name] = average_precision_score(t, p) if 0 < t.sum() < t.size else float("nan")
        tb = T.max(-1)[MM]; pb = 1 - np.prod(1 - P, -1)[MM]
        aucs["bust"] = average_precision_score(tb, pb)
        score = aucs["bust"]
        print(f"epoch {ep:2d} | {time.time()-t0:5.1f}s | " +
              " | ".join(f"{k} {v:.3f}" for k, v in aucs.items()), flush=True)
        history["epochs"].append({"epoch": ep, "seconds": round(time.time() - t0, 1),
                                  "val_pr_auc": {k: float(v) for k, v in aucs.items()}})
        if score > best:
            best, best_ep, since = score, ep, 0
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "head": args.head, "aucs": aucs, "epoch": ep}, out_path)
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at {ep}", flush=True); break
        history.update(best_epoch=best_ep, best_val_union_pr_auc=float(best))
        hist_path.write_text(json.dumps(history, indent=2))
    history.update(best_epoch=best_ep, best_val_union_pr_auc=float(best),
                   best_val_pr_auc=history["epochs"][best_ep - 1]["val_pr_auc"])
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\n[{args.arch}] best union-bust PR-AUC {best:.4f} @ epoch {best_ep} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
