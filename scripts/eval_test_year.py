"""Score the frozen models once on the held-out test year (default 2022, operational GEFS).

Everything that could be tuned was fixed before this year was touched: model weights,
the ensemble-only baselines (fitted on training inits), the probability calibration
(isotonic, fitted on the validation years) and the dashboard's display cut-off. Each
direction is scored only where that bust can happen (miss: GEFS < 20%, false alarm:
GEFS >= 50%), as in the dashboard's skill card.

Reports, model vs GEFS-only baseline: AUROC, PR-AUC, Brier skill (reference = the
validation-period bust rate, a forecast that could have been issued in advance), and the
bust rate inside vs outside the area the dashboard would colour.

    python -m scripts.eval_test_year            # writes outputs/test_2022.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.export_dashboard_predictions import (
    DEV, LAT, LEADS, LON, OUT, apply_rain_baseline, fit_rain_baseline, heatwave_baseline,
    heatwave_checkpoints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHES = Path(r"D:\sih-data\caches")
SHOW_FROM = 0.5  # web/src/lib/risk.ts
DIRS = (("miss", 0, "p_miss", "b_miss", "obs_miss"), ("false_alarm", 1, "p_fa", "b_fa", "obs_fa"))


def eligible(direction: str, ens: np.ndarray) -> np.ndarray:
    return ens < 0.2 if direction == "miss" else ens >= 0.5


def display_cut(tiers: dict) -> float:
    """The calibrated probability the dashboard starts colouring at (atIntensity(SHOW_FROM))."""
    base = tiers["base_rate"]
    curve = sorted((c for c in tiers["calibration"]["curve"] if c["p"] > 0 and c["catch"] >= 0.01),
                   key=lambda c: c["p"])
    cap = max(curve[-1]["p"] if curve else tiers["calibration"]["max"], base * 2)
    return base * np.exp(SHOW_FROM * np.log(cap / base))


def calibrated(p: np.ndarray, cal: dict) -> np.ndarray:
    """[init, lead, ...] raw scores -> the dashboard's calibrated values (per lead day where fitted)."""
    out = np.empty(p.shape, np.float64)
    for L in range(p.shape[1]):
        k = next((b for b in cal.get("bands", []) if L + 1 in b["leads"]), cal)
        out[:, L] = np.interp(p[:, L], k["x"], k["y"])
    return out


def predict_rain(X: np.memmap) -> np.ndarray:
    from src.models.graphnet.grid_graph import build_grid_graph
    from src.models.graphnet.model import GridGraphNet
    ck = torch.load(PROJECT_ROOT / "outputs" / "graphnet_20yr_directional_nbhd24.pt", map_location=DEV)
    model = GridGraphNet(build_grid_graph(), in_channels=24, hidden=ck["hidden"], out_channels=2).to(DEV)
    model.load_state_dict(ck["model_state"]); model.eval()
    P = np.empty((len(X), LEADS, LAT, LON, 2), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 8):
            P[s:s + 8] = model(torch.from_numpy(np.asarray(X[s:s + 8])).to(DEV)).cpu().numpy()
    return P


def predict_heatwave(X: np.memmap, C: int) -> tuple[np.ndarray, np.ndarray]:
    from scripts.train_heatwave import HeatwaveBustNet
    ckpts = heatwave_checkpoints()
    P = np.zeros((len(X), LEADS, LAT, LON, 2), np.float32)
    for st in ckpts:
        m = HeatwaveBustNet(st["arch"], C, st["wide_idx"]).to(DEV)
        m.load_state_dict(st["model_state"]); m.eval()
        with torch.no_grad():
            for s in range(0, len(X), 8):
                P[s:s + 8] += m(torch.from_numpy(np.asarray(X[s:s + 8])).to(DEV)).cpu().numpy()
    B = np.stack([heatwave_baseline(np.asarray(x), ckpts) for x in X])
    return P / len(ckpts), B


def score(hazard: str, P: np.ndarray, B: np.ndarray, Y: np.ndarray, ens: np.ndarray, index: dict) -> dict:
    val = {q.stem: np.load(q, mmap_mode="r") for q in (OUT / hazard).glob("*.npy")}
    v_ens = np.asarray(val["ens_prob"], np.float32)
    out = {}
    for d, j, pk, bk, ok in DIRS:
        y_all = Y[..., j]; fin_all = np.isfinite(y_all)
        fin = fin_all & eligible(d, ens)
        y = y_all[fin].astype(np.float64)
        v_fin = (np.asarray(val[ok]) >= 0) & eligible(d, v_ens)
        vy = np.asarray(val[ok])[v_fin].astype(np.float64)
        ref = vy.mean()
        res = {"n": int(fin.sum()), "busts": int(y.sum()), "bust_rate": float(y.mean()), "val_bust_rate": float(ref)}
        for name, S, vk in (("model", P, pk), ("baseline", B, bk)):
            s = S[..., j][fin].astype(np.float64)
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                np.asarray(val[vk])[v_fin].astype(np.float64), vy)
            bs, bs_ref = np.mean((iso.predict(s) - y) ** 2), np.mean((ref - y) ** 2)
            res[name] = {"auroc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                         "bss": float(1 - bs / bs_ref)}
        # dashboard view: calibrated with the dashboard's own (validation) curve, all observed cells
        t = index[hazard]["tiers"][d]
        cal = calibrated(P[..., j], t["calibration"])[fin_all]
        flag, yy = cal >= display_cut(t), y_all[fin_all]
        res["display"] = {"cut": float(display_cut(t)), "share": float(flag.mean()),
                          "inside": float(yy[flag].mean()) if flag.any() else None,
                          "outside": float(yy[~flag].mean()),
                          "catch": float(yy[flag].sum() / max(yy.sum(), 1))}
        out[d] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()
    index = json.loads((OUT / "index.json").read_text())
    report = {"year": args.year, "source": "operational GEFSv12, frozen models", "hazards": {}}

    rc = CACHES / f"rain_test_{args.year}"
    if (rc / "meta.json").exists():
        n = json.loads((rc / "meta.json").read_text())["n"]
        X = np.memmap(rc / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 24))
        Y = np.asarray(np.memmap(rc / "Y_dir.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, 2)))
        ens = np.fromfile(rc / "P.dat", np.float32).reshape(n, LEADS, LAT, LON)
        coefs = fit_rain_baseline()
        B = np.stack([apply_rain_baseline(np.asarray(x), coefs) for x in X])
        report["hazards"]["rain"] = {"n_inits": n, **score("rain", predict_rain(X), B, Y, ens, index)}

    hc = CACHES / f"heatwave_test_{args.year}"
    if (hc / "meta.json").exists():
        meta = json.loads((hc / "meta.json").read_text())
        n, C = meta["n"], len(meta["channels"])
        X = np.memmap(hc / "X.dat", np.float32, "r", shape=(n, LEADS, LAT, LON, C))
        Y = np.fromfile(hc / "Y_dir.dat", np.float32).reshape(n, LEADS, LAT, LON, 2)
        ens = np.fromfile(hc / "P.dat", np.float32).reshape(n, LEADS, LAT, LON)
        P, B = predict_heatwave(X, C)
        report["hazards"]["heatwave"] = {"n_inits": n, **score("heatwave", P, B, Y, ens, index)}

    path = PROJECT_ROOT / "outputs" / f"test_{args.year}.json"
    path.write_text(json.dumps(report, indent=1))
    for h, r in report["hazards"].items():
        for d in ("miss", "false_alarm"):
            s, v = r[d], index[h].get("skill", {}).get(d)
            m, b, disp = s["model"], s["baseline"], s["display"]
            print(f"{h:8s} {d:11s} {args.year}: AUROC {m['auroc']:.3f} vs {b['auroc']:.3f} | PR-AUC {m['pr_auc']:.3f} "
                  f"vs {b['pr_auc']:.3f} (rate {s['bust_rate']:.4f}) | BSS {m['bss']:+.3f} vs {b['bss']:+.3f}"
                  + (f" || val AUROC {v['model']['auroc']:.3f} vs {v['baseline']['auroc']:.3f}" if v else ""))
            inside = f"{disp['inside']:.3f}" if disp["inside"] is not None else "-"
            print(f"{'':21s}coloured {disp['share']:.2%} of cells: bust rate {inside} inside vs {disp['outside']:.4f} "
                  f"outside, catches {disp['catch']:.0%}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
