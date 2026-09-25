"""Low-pressure-system strike-bust bake-off: GNN (flat / multi-mesh) vs CNN vs ConvLSTM.

Inputs per cell/lead: the 65-ch rich cache (rain_rich_10yr) + 6 LPS channels (lps_10yr/X_lps.dat)
+ lead (0-1) = 72 channels. Targets: P(miss), P(false alarm) of a system passing within
300 km (labels from build_lps_cache.py). Same wide-and-deep head as the heatwave model:
a logistic baseline on the ensemble LPS signals (+ lead), frozen; the encoder adds a
correction. Epoch 0 scores that baseline. Train 2000-2007, validate 2008-2009.
The rich cache (39 GB) is read per batch from its memmap; the LPS channels fit in RAM.
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

from scripts.train_heatwave import TARGETS, HeatwaveBustNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RICH_CH = 65
WIDE_NAMES = ["p_strike300", "p_strike600", "zeta850_mean", "zeta850_spread", "msl_dip_mean", "lead01"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gnn", "multimesh", "convlstm", "cnn"], required=True)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\lps_10yr")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W = meta["n"], meta["leads"], meta["lat"], meta["lon"]
    lps_names = meta["channels"]
    rich = np.memmap(Path(meta["rich_cache"]) / "X.dat", np.float32, "r", shape=(n, L, H, W, RICH_CH))
    Xl = np.fromfile(cache / "X_lps.dat", np.float32).reshape(n, L, H, W, len(lps_names))
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)
    P = np.fromfile(cache / "P.dat", np.float32).reshape(n, L, H, W)
    lead01 = np.broadcast_to(np.linspace(0, 1, L, dtype=np.float32)[:, None, None, None], (L, H, W, 1))
    C = RICH_CH + len(lps_names) + 1
    names = [f"rich{i}" for i in range(RICH_CH)] + lps_names + ["lead01"]
    wide_idx = [names.index(w) for w in WIDE_NAMES]
    is_tr = np.asarray(meta["is_train"])
    tr_idx, va_idx = np.where(is_tr)[0], np.where(~is_tr)[0]
    print(f"[{args.arch}] n {n} train {len(tr_idx)} val {len(va_idx)} | {C} ch | wide {WIDE_NAMES}", flush=True)

    def features(idx):
        x = np.concatenate([np.asarray(rich[idx]), Xl[idx], np.broadcast_to(lead01, (len(idx), L, H, W, 1))], -1)
        return x

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeatwaveBustNet(args.arch, C, wide_idx).to(dev)
    rng = np.random.default_rng(args.seed)
    # frozen logistic baseline on wide features (LPS channels + lead only - no rich read needed)
    wide_feats = np.concatenate([Xl[tr_idx], np.broadcast_to(lead01, (len(tr_idx), L, H, W, 1))], -1)
    wide_feats = wide_feats[..., [(lps_names + ["lead01"]).index(w) for w in WIDE_NAMES]]
    ytr = Y[tr_idx]; pw = []
    with torch.no_grad():
        for j, name in enumerate(TARGETS):
            yj = ytr[..., j]; fin = np.isfinite(yj)
            pos = np.argwhere(fin & (yj == 1)); neg = np.argwhere(fin & (yj == 0))
            pw.append(len(neg) / max(len(pos), 1))
            neg = neg[rng.choice(len(neg), min(len(neg), 20 * max(len(pos), 1000)), replace=False)]
            pick = np.concatenate([pos, neg]); ys = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(wide_feats[tuple(pick.T)], ys)
            model.wide.weight[j] = torch.tensor(lr.coef_[0], dtype=torch.float32)
            model.wide.bias[j] = float(lr.intercept_[0])
            print(f"  {name}: train positives {len(pos):,} base {100*len(pos)/fin.sum():.3f}% (pos_weight {pw[-1]:.0f})",
                  flush=True)
    for prm in model.wide.parameters():
        prm.requires_grad = False
    del wide_feats, ytr
    pwt = torch.tensor(pw, device=dev, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    yv = Y[va_idx]; fin_v = np.isfinite(yv[..., 0])
    rules = {"miss": float(average_precision_score(yv[..., 0][fin_v], Xl[va_idx][..., lps_names.index("p_strike600")][fin_v])),
             "false_alarm": float(average_precision_score(yv[..., 1][fin_v], P[va_idx][fin_v]))}
    base_rate = {t: float(yv[..., j][fin_v].mean()) for j, t in enumerate(TARGETS)}
    print(f"trainable params {n_params:,} | val base miss {100*base_rate['miss']:.3f}% fa {100*base_rate['false_alarm']:.3f}% "
          f"| rule miss {rules['miss']:.4f} (p_strike600) fa {rules['false_alarm']:.4f} (ensemble P)", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=1e-4)

    def to_dev(idx):
        yb = torch.from_numpy(Y[idx]).to(dev)
        return torch.from_numpy(features(idx)).to(dev), torch.nan_to_num(yb), torch.isfinite(yb).float()

    def evaluate():
        model.eval(); ps = []
        with torch.no_grad():
            for s in range(0, len(va_idx), args.batch):
                ps.append(model(to_dev(va_idx[s:s + args.batch])[0]).cpu().numpy())
        pr = np.concatenate(ps)
        return {t: float(average_precision_score(yv[..., j][fin_v], pr[..., j][fin_v])) for j, t in enumerate(TARGETS)}

    tag = f"{args.arch}{args.tag}"
    out_path = PROJECT_ROOT / "outputs" / f"lps_{tag}.pt"
    hist_path = PROJECT_ROOT / "outputs" / f"lps_{tag}_history.json"
    hist = {"arch": args.arch, "config": vars(args), "params_trainable": n_params, "val_base_rate": base_rate,
            "rule_pr_auc": rules, "tracker": meta["tracker"], "epochs": []}
    a0 = evaluate(); hist["baseline_logistic_pr_auc"] = a0
    print(f"epoch  0 | baseline (frozen logistic) | miss {a0['miss']:.4f} | fa {a0['false_alarm']:.4f}", flush=True)
    sel = lambda a: (a["miss"] + a["false_alarm"]) / 2
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
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "wide_idx": wide_idx,
                        "in_channels": C, "val_pr_auc": a, "epoch": ep}, out_path)
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at {ep}", flush=True); break
        hist.update(best_epoch=best_ep, best_val_select=best)
        hist_path.write_text(json.dumps(hist, indent=2))
    best_a = hist["epochs"][best_ep - 1]["val_pr_auc"] if best_ep else a0
    hist.update(best_epoch=best_ep, best_val_select=best, best_val_pr_auc=best_a)
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"\n[{tag}] best mean {best:.4f} @ epoch {best_ep}: miss {best_a['miss']:.4f} fa {best_a['false_alarm']:.4f} "
          f"(baseline miss {a0['miss']:.4f} fa {a0['false_alarm']:.4f})", flush=True)


if __name__ == "__main__":
    main()
