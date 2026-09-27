"""Low-pressure-system strike-bust bake-off: GNN (flat / multi-mesh) vs CNN vs ConvLSTM.

Inputs per cell/lead: the 65-ch rich cache (rain_rich_10yr) + 6 LPS channels (lps_10yr/X_lps.dat)
+ lead (0-1) = 72 channels. Targets: P(miss), P(false alarm) of a system passing within
300 km (labels from build_lps_cache.py). Same wide-and-deep head as the heatwave model:
a logistic baseline on the ensemble LPS signals (+ lead), frozen; the encoder adds a
correction. Epoch 0 scores that baseline. Train 2000-2007, validate 2008-2009.

The 20-year cache (lps_20yr, 24-ch rain features, "feature_ch" in meta.json) trains on
2000-2015 / validates on 2016-2019 by default; --train-last/--val-years override the split,
e.g. --train-last 2007 --val-years 2008-2009 for the equal-years ablation.
--lag adds the lagged-ensemble channels (X_lag.dat from build_lps_lag.py: the runs issued
1 and 2 days earlier at the same valid time); their pooled strike probability also joins the
frozen GEFS-only baseline, so "GEFS alone" gets the extra runs too.
--env adds storm-environment channels (X_env.dat from build_lps_env.py: MJO at issue time,
seasonal storm climatology, day of year). They are not GEFS output, so the baseline ignores them.
LPS channels are read from memmaps per batch (the 20-year cache does not fit in RAM).
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
    ap.add_argument("--train-last", type=int, default=None, help="Last training year (default: meta is_train).")
    ap.add_argument("--val-years", default=None, help="e.g. 2008-2009 (default: every non-training init).")
    ap.add_argument("--lag", action="store_true", help="Add lagged-ensemble channels (X_lag.dat).")
    ap.add_argument("--env", action="store_true", help="Add storm-environment channels (X_env.dat).")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W = meta["n"], meta["leads"], meta["lat"], meta["lon"]
    lps_names = meta["channels"]
    RICH_CH = meta.get("feature_ch", 65)
    rich = np.memmap(Path(meta["rich_cache"]) / "X.dat", np.float32, "r", shape=(n, L, H, W, RICH_CH))
    Xl = np.memmap(cache / "X_lps.dat", np.float32, "r", shape=(n, L, H, W, len(lps_names)))
    lag_names = meta["lag_channels"] if args.lag else []
    Xg = np.memmap(cache / "X_lag.dat", np.float32, "r", shape=(n, L, H, W, len(lag_names))) if args.lag else None
    env_names = meta["env_channels"] if args.env else []
    Xe = np.memmap(cache / "X_env.dat", np.float32, "r", shape=(n, L, H, W, len(env_names))) if args.env else None
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)
    P = np.fromfile(cache / "P.dat", np.float32).reshape(n, L, H, W)
    lead01 = np.broadcast_to(np.linspace(0, 1, L, dtype=np.float32)[:, None, None, None], (L, H, W, 1))
    C = RICH_CH + len(lps_names) + len(lag_names) + len(env_names) + 1
    names = [f"rich{i}" for i in range(RICH_CH)] + lps_names + lag_names + env_names + ["lead01"]
    wide_names = WIDE_NAMES + (["p_strike300_lagged"] if args.lag else [])
    wide_idx = [names.index(w) for w in wide_names]
    years = np.array([int(s[:4]) for s in meta["inits"]])
    is_tr = years <= args.train_last if args.train_last else np.asarray(meta["is_train"])
    if args.val_years:
        v0, v1 = (int(y) for y in args.val_years.split("-"))
        is_va = (years >= v0) & (years <= v1) & ~is_tr
    else:
        is_va = ~is_tr
    tr_idx, va_idx = np.where(is_tr)[0], np.where(is_va)[0]
    print(f"[{args.arch}] n {n} train {len(tr_idx)} val {len(va_idx)} | {C} ch | wide {wide_names}", flush=True)

    def extra(idx):  # LPS (+ lagged) channels + lead, i.e. everything after the rain/rich block
        parts = ([np.asarray(Xl[idx])] + ([np.asarray(Xg[idx])] if args.lag else [])
                 + ([np.asarray(Xe[idx])] if args.env else []))
        return np.concatenate(parts + [np.broadcast_to(lead01, (len(idx), L, H, W, 1))], -1)

    def features(idx):
        return np.concatenate([np.asarray(rich[idx]), extra(idx)], -1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HeatwaveBustNet(args.arch, C, wide_idx).to(dev)
    rng = np.random.default_rng(args.seed)
    # frozen logistic baseline on wide features (LPS/lag channels + lead only - no rich read needed).
    # Cells are sampled by flat index and gathered row by row, so nothing train-sized is materialised.
    wide_cols = [(lps_names + lag_names + env_names + ["lead01"]).index(w) for w in wide_names]
    ytr = Y[tr_idx]; pw = []
    cell = L * H * W

    def gather(flat):
        rows, rest = np.divmod(np.sort(flat), cell)
        out = np.empty((len(flat), len(wide_cols)), np.float32); k = 0
        for r in np.unique(rows):
            sel = rest[rows == r]
            out[k:k + len(sel)] = extra(tr_idx[[r]])[0].reshape(cell, -1)[sel][:, wide_cols]; k += len(sel)
        return out, np.sort(flat)

    with torch.no_grad():
        for j, name in enumerate(TARGETS):
            yj = ytr[..., j].reshape(-1); fin = np.isfinite(yj)
            pos = np.flatnonzero(fin & (yj == 1)); n_neg = int(fin.sum()) - len(pos)
            pw.append(n_neg / max(len(pos), 1))
            want = min(n_neg, 20 * max(len(pos), 1000))
            cand = np.unique(rng.integers(0, len(yj), int(want * 1.3) + 1000))
            neg = cand[fin[cand] & (yj[cand] == 0)][:want]
            feats, flat = gather(np.concatenate([pos, neg]))
            ys = (yj[flat] == 1).astype(np.float64)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats, ys)
            model.wide.weight[j] = torch.tensor(lr.coef_[0], dtype=torch.float32)
            model.wide.bias[j] = float(lr.intercept_[0])
            print(f"  {name}: train positives {len(pos):,} base {100*len(pos)/fin.sum():.3f}% (pos_weight {pw[-1]:.0f})",
                  flush=True)
    for prm in model.wide.parameters():
        prm.requires_grad = False
    del ytr
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
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "wide_idx": wide_idx, "lag": args.lag, "env": args.env,
                        "in_channels": C, "val_pr_auc": a, "epoch": ep, "cache": str(cache),
                        "train_last": int(years[tr_idx].max()), "val_years": [int(years[va_idx].min()), int(years[va_idx].max())]},
                       out_path)
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
