"""Score western-disturbance checkpoints (averaged) against GEFS alone on a WD cache.

Uses the cache's non-training rows: 2016-2019 in wd_20yr, all of 2022 in wd_test_2022.
GEFS-only = the frozen ensemble-only logistic inside each checkpoint (the wide path).
Scored on the cells where each bust is possible (miss: GEFS P < 0.2, false alarm: P >= 0.5).

  python -m scripts.eval_wd --cache-dir D:/sih-data/caches/wd_test_2022 --ckpts wd_gnn_s0 wd_gnn_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.train_heatwave import HeatwaveBustNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE = {"miss": lambda p: p < 0.2, "false_alarm": lambda p: p >= 0.5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\wd_test_2022")
    ap.add_argument("--ckpts", nargs="+", default=["wd_gnn_s0", "wd_gnn_s1"])
    ap.add_argument("--out", default=None, help="write the scores as JSON")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W, C = meta["n"], meta["leads"], meta["lat"], meta["lon"], len(meta["channels"])
    X = np.memmap(cache / "X.dat", np.float16, "r", shape=(n, L, H, W, C))
    Y = np.fromfile(cache / "Y_dir.dat", np.float32).reshape(n, L, H, W, 2)
    P = np.fromfile(cache / "P.dat", np.float32).reshape(n, L, H, W)
    rows = np.where(~np.asarray(meta["is_train"]))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model_p = np.zeros((len(rows), L, H, W, 2), np.float32)
    base_p = np.zeros_like(model_p)
    for name in args.ckpts:
        st = torch.load(PROJECT_ROOT / "outputs" / f"{name}.pt", map_location=dev)
        net = HeatwaveBustNet(st["arch"], C, st["wide_idx"]).to(dev)
        net.load_state_dict(st["model_state"]); net.eval()
        with torch.no_grad():
            for s in range(0, len(rows), 8):
                x = torch.from_numpy(np.asarray(X[rows[s:s + 8]], np.float32)).to(dev)
                model_p[s:s + 8] += net(x).cpu().numpy()
                base_p[s:s + 8] += torch.sigmoid(net.wide(x[..., net.wide_idx])).cpu().numpy()
    model_p /= len(args.ckpts); base_p /= len(args.ckpts)

    y, p = Y[rows], P[rows]
    lead = np.broadcast_to(np.arange(1, L + 1)[None, :, None, None], p.shape)
    print(f"{cache.name}: {len(rows)} inits {meta['inits'][rows[0]][:8]}..{meta['inits'][rows[-1]][:8]} | "
          f"ckpts {', '.join(args.ckpts)}")
    out = {"cache": str(cache), "ckpts": args.ckpts, "n_inits": int(len(rows))}
    for j, d in enumerate(("miss", "false_alarm")):
        ok = np.isfinite(y[..., j]) & ELIGIBLE[d](p)
        yt, m, b = y[..., j][ok], model_p[..., j][ok], base_p[..., j][ok]
        res = {"positives": int(yt.sum()), "base_rate": float(yt.mean()),
               "auroc": [roc_auc_score(yt, m), roc_auc_score(yt, b)],
               "pr_auc": [average_precision_score(yt, m), average_precision_score(yt, b)], "by_lead": {}}
        print(f"  {d}: {res['positives']} busts, base {100 * res['base_rate']:.3f}% | "
              f"AUROC {res['auroc'][0]:.3f} vs {res['auroc'][1]:.3f} | "
              f"PR-AUC {res['pr_auc'][0]:.3f} vs {res['pr_auc'][1]:.3f}  (model vs GEFS-only)")
        for band in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
            sel = np.isin(lead[ok], band)
            if yt[sel].sum() == 0:
                continue
            pr = [average_precision_score(yt[sel], m[sel]), average_precision_score(yt[sel], b[sel])]
            res["by_lead"][f"{band[0]}-{band[-1]}"] = pr
            print(f"    days {band[0]}-{band[-1]}: {int(yt[sel].sum())} busts | PR-AUC {pr[0]:.3f} vs {pr[1]:.3f}")
        out[d] = res
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
