"""Heatwave-bust bake-off: GNN (flat / multi-mesh) vs ConvLSTM vs CNN on the heatwave cache.

Same cache, labels, head and loop; only the spatial(-temporal) encoder differs:
  gnn:      GraphSAGE + TCN (GridGraphNet encoder), flat 0.5 deg 8-neighbour graph
  multimesh: GraphSAGE over stacked 0.5/1/2/4 deg grids with up/down links + TCN
  convlstm: 2-layer ConvLSTM (16, 32) forward over leads - PyTorch port of src/models/convlstm.py
  cnn:      2 conv layers per lead + TCN (same as the active/break CNN)
Per-cell, per-lead, two outputs: P(miss), P(false alarm).

Head = wide-and-deep: a logistic baseline on the ensemble-only channels (+ lead) is
fitted on training cells, copied into a frozen linear path; the encoder adds a
correction through a zero-initialised deep path. Epoch 0 therefore scores the baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn

from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.mesh import build_multi_mesh
from src.models.graphnet.multimesh_model import MultiMeshGraphNet
from src.models.graphnet.model import GridGraphNet
from scripts.train_active_break import CNNEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("miss", "false_alarm")


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: tuple[int, int] = (16, 32), k: int = 3, dropout: float = 0.15):
        super().__init__()
        self.hidden = hidden
        dims = [in_ch, *hidden]
        self.gates = nn.ModuleList(nn.Conv2d(dims[i] + dims[i + 1], 4 * dims[i + 1], k, padding=k // 2)
                                   for i in range(2))
        self.norms = nn.ModuleList(nn.GroupNorm(1, h) for h in hidden)  # LayerNorm over channels
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden[-1]

    def forward(self, x):  # [B, L, H, W, C] -> [B, L, H, W, hidden]
        b, l, h, w, _ = x.shape
        seq = x.permute(0, 1, 4, 2, 3)
        for gate, norm, hid in zip(self.gates, self.norms, self.hidden):
            hs = x.new_zeros(b, hid, h, w); cs = torch.zeros_like(hs); outs = []
            for t in range(l):
                i, f, g, o = gate(torch.cat([self.drop(seq[:, t]), hs], 1)).chunk(4, 1)
                cs = torch.sigmoid(f) * cs + torch.sigmoid(i) * torch.tanh(g)
                hs = torch.sigmoid(o) * torch.tanh(cs)
                outs.append(norm(hs))
            seq = torch.stack(outs, 1)
        return seq.permute(0, 1, 3, 4, 2)


class HeatwaveBustNet(nn.Module):
    def __init__(self, arch: str, in_ch: int, wide_idx: list[int], hidden: int = 64):
        super().__init__()
        if arch == "gnn":
            self.gnn = GridGraphNet(build_grid_graph(), in_channels=in_ch, hidden=hidden)
            self.encode, dim = self.gnn.features, hidden
        elif arch == "multimesh":
            self.mm = MultiMeshGraphNet(build_multi_mesh(n_levels=4), in_channels=in_ch, hidden=hidden)
            self.encode, dim = self.mm.features, hidden
        elif arch == "cnn":
            self.encode = CNNEncoder(in_ch, hidden); dim = hidden
        else:
            self.encode = ConvLSTMEncoder(in_ch); dim = self.encode.out_dim
        self.register_buffer("wide_idx", torch.tensor(wide_idx))
        self.wide = nn.Linear(len(wide_idx), len(TARGETS))
        self.deep = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, len(TARGETS)))
        nn.init.zeros_(self.deep[-1].weight); nn.init.zeros_(self.deep[-1].bias)

    def forward(self, x):
        return torch.sigmoid(self.wide(x[..., self.wide_idx]) + self.deep(self.encode(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gnn", "multimesh", "convlstm", "cnn"], required=True)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\heatwave_10yr")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=["miss", "mean"], default="mean",
                    help="checkpoint/early-stop metric: miss PR-AUC or mean(miss, false_alarm)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _s1")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W, C = meta["n"], meta["leads"], meta["lat"], meta["lon"], len(meta["channels"])
    X = np.fromfile(cache / "X.dat", np.float32).reshape(n, L, H, W, C)  # ~5 GB, fits in RAM
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)
    P = np.fromfile(cache / "P.dat", np.float32).reshape(n, L, H, W)
    is_tr = np.asarray(meta["is_train"])
    tr_idx, va_idx = np.where(is_tr)[0], np.where(~is_tr)[0]
    wide_idx = meta["wide_channels"]
    print(f"[{args.arch}] n {n} train {len(tr_idx)} val {len(va_idx)} | {C} ch | wide {meta['wide_names']}", flush=True)

    # --- frozen logistic baseline on training cells (all positives + sampled negatives, balanced)
    rng = np.random.default_rng(args.seed)
    ytr = Y[tr_idx]; xw = X[tr_idx][..., wide_idx]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeatwaveBustNet(args.arch, C, wide_idx).to(dev)
    pw = []
    with torch.no_grad():
        for j, name in enumerate(TARGETS):
            yj = ytr[..., j]; fin = np.isfinite(yj)
            pos = np.argwhere(fin & (yj == 1)); neg = np.argwhere(fin & (yj == 0))
            pw.append((len(neg)) / max(len(pos), 1))
            neg = neg[rng.choice(len(neg), min(len(neg), 20 * max(len(pos), 1000)), replace=False)]
            pick = np.concatenate([pos, neg]); ys = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            feats = xw[tuple(pick.T)]
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats, ys)
            model.wide.weight[j] = torch.tensor(lr.coef_[0], dtype=torch.float32)
            model.wide.bias[j] = float(lr.intercept_[0])
            print(f"  {name}: train positives {len(pos):,} base {100*len(pos)/fin.sum():.3f}% "
                  f"(pos_weight {pw[-1]:.0f})", flush=True)
    for prm in model.wide.parameters():
        prm.requires_grad = False
    del ytr, xw
    pwt = torch.tensor(pw, device=dev, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {n_params:,}", flush=True)

    # trivial same-task rules on val: miss = rank by bias-corrected mean departure where P<0.2; fa = ensemble P
    yv = Y[va_idx]; fin_v = np.isfinite(yv[..., 0])
    dep_ch = meta["channels"].index("tmax_departure")
    rules = {"miss": average_precision_score(yv[..., 0][fin_v], X[va_idx][..., dep_ch][fin_v]),
             "false_alarm": average_precision_score(yv[..., 1][fin_v], P[va_idx][fin_v])}
    base_rate = {t: float(yv[..., j][fin_v].mean()) for j, t in enumerate(TARGETS)}
    print(f"val base rate miss {100*base_rate['miss']:.3f}% fa {100*base_rate['false_alarm']:.3f}% | "
          f"rule PR-AUC miss {rules['miss']:.4f} (departure) fa {rules['false_alarm']:.4f} (ensemble P)", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=1e-4)

    def to_dev(idx):
        yb = torch.from_numpy(Y[idx]).to(dev)
        m = torch.isfinite(yb).float()
        return torch.from_numpy(X[idx]).to(dev), torch.nan_to_num(yb), m

    def evaluate():
        model.eval(); ps = []
        with torch.no_grad():
            for s in range(0, len(va_idx), args.batch):
                ps.append(model(to_dev(va_idx[s:s + args.batch])[0]).cpu().numpy())
        pr = np.concatenate(ps)
        return {t: float(average_precision_score(yv[..., j][fin_v], pr[..., j][fin_v])) for j, t in enumerate(TARGETS)}

    out_path = PROJECT_ROOT / "outputs" / f"heatwave_{args.arch}{args.tag}.pt"
    hist_path = PROJECT_ROOT / "outputs" / f"heatwave_{args.arch}{args.tag}_history.json"
    hist = {"arch": args.arch, "config": vars(args), "cache": str(cache), "params_trainable": n_params,
            "val_base_rate": base_rate, "rule_pr_auc": rules, "epochs": []}
    a0 = evaluate()
    hist["baseline_logistic_pr_auc"] = a0
    print(f"epoch  0 | baseline (frozen logistic) | miss {a0['miss']:.4f} | fa {a0['false_alarm']:.4f}", flush=True)

    sel = (lambda a: a["miss"]) if args.select == "miss" else (lambda a: (a["miss"] + a["false_alarm"]) / 2)
    best, best_ep, since = sel(a0), 0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        order = rng.permutation(tr_idx)
        for s in range(0, len(order), args.batch):
            x, y, m = to_dev(np.sort(order[s:s + args.batch]))
            p = model(x)
            w = torch.where(y > 0.5, pwt, torch.ones_like(y))
            loss = (F.binary_cross_entropy(p, y, reduction="none") * w * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        a = evaluate()
        print(f"epoch {ep:2d} | {time.time()-t0:5.1f}s | miss {a['miss']:.4f} | fa {a['false_alarm']:.4f}", flush=True)
        hist["epochs"].append({"epoch": ep, "seconds": round(time.time() - t0, 1), "val_pr_auc": a})
        if sel(a) > best:
            best, best_ep, since = sel(a), ep, 0
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "channels": meta["channels"],
                        "wide_idx": wide_idx, "val_pr_auc": a, "epoch": ep}, out_path)
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at {ep}", flush=True); break
        hist.update(best_epoch=best_ep, best_val_select=best)
        hist_path.write_text(json.dumps(hist, indent=2))
    best_a = hist["epochs"][best_ep - 1]["val_pr_auc"] if best_ep else a0
    hist.update(best_epoch=best_ep, best_val_select=best, best_val_pr_auc=best_a)
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"\n[{args.arch}{args.tag}] best {args.select} {best:.4f} @ epoch {best_ep}: miss {best_a['miss']:.4f} "
          f"fa {best_a['false_alarm']:.4f} (baseline miss {a0['miss']:.4f} fa {a0['false_alarm']:.4f})", flush=True)


if __name__ == "__main__":
    main()
