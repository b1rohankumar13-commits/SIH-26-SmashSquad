"""What the live pipeline's real-time stand-ins cost, over the 2022 test year.

Heatwave and WD models were trained with observations as inputs (ERA5 Tmax on the two days
before issue; yesterday's IMD rain and the ERA5 upstream trough). In real time those do not
exist yet and scripts/live_forecast.py substitutes the forecast system's own earlier runs.
This runs every in-season 2022 GEFS forecast both ways - observed inputs vs --realtime
stand-ins - and scores both against the same truth, on the cells where each bust can happen
(miss: GEFS P < 0.2, false alarm: P >= 0.5), next to the GEFS-only wide path.

    python -m scripts.eval_live_realtime            # heatwave (Mar-Jun) and wd (Dec-Mar)
    python -m scripts.eval_live_realtime --hazards wd
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts import live_forecast as lf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE = {"miss": lambda p: p < 0.2, "false_alarm": lambda p: p >= 0.5}


def wide_only(models: list, x: np.ndarray) -> np.ndarray:
    """GEFS-only baseline: the checkpoints' frozen wide paths, averaged."""
    acc = np.zeros((lf.LEADS, lf.LAT, lf.LON, 2), np.float32)
    with torch.no_grad():
        for m, inp in models:
            v = torch.from_numpy(inp(x)[None]).to(lf.DEV)
            acc += torch.sigmoid(m.wide(v[..., m.wide_idx]))[0].cpu().numpy()
    return acc / len(models)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hazards", default="heatwave,wd")
    ap.add_argument("--year", default="2022")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "live_realtime_2022.json"))
    args = ap.parse_args()
    inits = sorted(Path(f).stem[:8] for f in glob.glob(str(lf.STORES["gefs"] / "rain" / f"{args.year}*.nc")))
    builders = {"heatwave": lf.heatwave_inputs, "wd": lf.wd_inputs}
    report = {}
    for h in args.hazards.split(","):
        rows = [i for i in inits if lf.day(i).month in lf.SEASON[h]
                and all(lf.store_file("gefs", k, i).exists() for k in ({"heatwave": ("tmax", "mv")}.get(h, ("rain",))))]
        acc = {m: {"y": [], "p": [], "b": [], "ens": [], "lead": []} for m in ("observed", "realtime")}
        t0 = time.time()
        for n, init in enumerate(rows):
            for mode in ("observed", "realtime"):
                inp = builders[h]("gefs", init, mode == "realtime")
                if inp["obs"] is None:
                    break
                p = lf.predict(inp["models"], inp["x"])
                a = acc[mode]
                a["y"].append(inp["obs"]); a["p"].append(p); a["b"].append(wide_only(inp["models"], inp["x"]))
                a["ens"].append(inp["ens"])
            if n % 20 == 0:
                print(f"  {h} {n}/{len(rows)} | {time.time() - t0:.0f}s", flush=True)
        report[h] = {"inits": len(acc["observed"]["y"])}
        for mode, a in acc.items():
            y, p, b, e = (np.stack(a[k]) for k in ("y", "p", "b", "ens"))
            report[h][mode] = {}
            for j, d in enumerate(("miss", "false_alarm")):
                ok = np.isfinite(y[..., j]) & ELIGIBLE[d](e)
                yt = y[..., j][ok]
                if yt.sum() == 0:
                    continue
                report[h][mode][d] = {"busts": int(yt.sum()),
                                      "auroc": float(roc_auc_score(yt, p[..., j][ok])),
                                      "pr_auc": float(average_precision_score(yt, p[..., j][ok])),
                                      "gefs_only_pr_auc": float(average_precision_score(yt, b[..., j][ok]))}
        for d in ("miss", "false_alarm"):
            o, r = report[h]["observed"].get(d), report[h]["realtime"].get(d)
            if o and r:
                print(f"{h}/{d}: {o['busts']} busts, {report[h]['inits']} forecasts | PR-AUC observed {o['pr_auc']:.3f} "
                      f"vs real-time {r['pr_auc']:.3f} (GEFS-only {o['gefs_only_pr_auc']:.3f}) | AUROC {o['auroc']:.3f} "
                      f"vs {r['auroc']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
