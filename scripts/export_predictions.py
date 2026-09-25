"""Run the GNN + XGBoost fusion for one forecast init and publish records to BustSentinel.

Emits one record per grid cell and lead day in the API's schema
({grid_id, lead_day, latitude, longitude, gnn_probability, overall_bust_probability})
and POSTs them to /gnn/predictions. The fusion and calibration are fit on the held-out
validation cells (excluding the exported init) and applied to the exported run.
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import torch
import xgboost
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.features.build_gridded_sequences import N_CHANNELS
from src.features.build_tabular_dataset import FEATURE_NAMES
from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres
from src.models.graphnet.model import GridGraphNet

ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
CACHE = Path(os.environ.get("SIH_CACHE_DIR") or ROOT / "data" / "interim" / "_graphnet_cache")
F = len(FEATURE_NAMES)


def sorted_files(years):
    files = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    return sorted(files)


def gnn_predict(model, grids, idx, device, batch=8):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            x = torch.from_numpy(np.asarray(grids[idx[s:s + batch]])).to(device)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=str, default="2000-2009")
    parser.add_argument("--gnn-ckpt", type=str, default="outputs/graphnet_10yr_19ch_baseline.pt")
    parser.add_argument("--xgb-model", type=str, default="outputs/xgb_10yr.json")
    parser.add_argument("--run-date", type=str, default=None,
                        help="Init date YYYYMMDD to export; default is the validation date with most busts.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", type=str, default="gnn-10yr-19ch")
    parser.add_argument("--region", type=str, default="India", help="region_id tag for every cell.")
    args = parser.parse_args()

    start, end = (args.years.split("-") + [args.years])[:2]
    files = sorted_files(range(int(start), int(end) + 1))
    n = len(files)
    grids = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n, 9, 66, 70, N_CHANNELS))
    table = np.memmap(CACHE / "T.dat", np.float32, "r", shape=(n, 9, 66, 70, F))
    labels = np.memmap(CACHE / "Y.dat", np.float32, "r", shape=(n, 9, 66, 70, 1))

    ckpt = torch.load(ROOT / args.gnn_ckpt, map_location="cpu", weights_only=False)
    n_train = int(ckpt["n_train"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GridGraphNet(build_grid_graph(), in_channels=int(ckpt.get("in_channels", N_CHANNELS)),
                         hidden=int(ckpt["hidden"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    clf = xgboost.XGBClassifier()
    clf.load_model(str(ROOT / args.xgb_model))

    run_ids = [Path(f).stem for f in files]
    val = np.arange(n_train, n)
    if args.run_date:
        run_idx = run_ids.index(next(r for r in run_ids if r.startswith(args.run_date)))
    else:
        val_busts = [int((np.asarray(labels[i]) == 1).sum()) for i in val]
        run_idx = int(val[int(np.argmax(val_busts))])
    init = run_ids[run_idx]
    print(f"exporting init {init} (index {run_idx})", flush=True)

    # Fit fusion + calibration on validation cells, excluding the exported init.
    fit_idx = val[val != run_idx]
    n_fit = len(fit_idx)
    g_fit = gnn_predict(model, grids, fit_idx, device).reshape(n_fit, 9, 66, 70)
    xb_fit = clf.predict_proba(np.asarray(table[fit_idx]).reshape(-1, F))[:, 1].reshape(n_fit, 9, 66, 70)
    y_fit = np.asarray(labels[fit_idx]).reshape(n_fit, 9, 66, 70)
    lead_fit = np.broadcast_to(np.arange(9)[None, :, None, None], (n_fit, 9, 66, 70))

    gf, xf, yf, lf = (a.reshape(-1) for a in (g_fit, xb_fit, y_fit, lead_fit))
    keep = np.isfinite(yf)
    gf, xf, yf, lf = gf[keep], xf[keep], yf[keep], lf[keep]
    stack_fit = np.column_stack([gf, xf, np.abs(gf - xf)])
    fusion = LogisticRegression(max_iter=1000).fit(stack_fit, yf)
    fused_fit = fusion.predict_proba(stack_fit)[:, 1]
    # Calibrate per lead day: each lead has its own reliability, and a single
    # pooled isotonic map hedges longer leads toward climatology (understating
    # their busts) even though the true bust rate is roughly flat across leads.
    calibrators = {
        lead: IsotonicRegression(out_of_bounds="clip").fit(fused_fit[lf == lead], yf[lf == lead])
        for lead in range(9)
    }

    # Predict the exported run and apply the same fusion + per-lead calibration.
    g = gnn_predict(model, grids, [run_idx], device).reshape(9, 66, 70)
    xb = clf.predict_proba(np.asarray(table[run_idx]).reshape(-1, F))[:, 1].reshape(9, 66, 70)
    fused = fusion.predict_proba(
        np.column_stack([g.reshape(-1), xb.reshape(-1), np.abs(g - xb).reshape(-1)]))[:, 1].reshape(9, 66, 70)
    overall = np.stack([calibrators[lead].predict(fused[lead].reshape(-1)).reshape(66, 70)
                        for lead in range(9)])

    lats, lons = canonical_centres()
    init_dt = datetime.strptime(init[:8], "%Y%m%d")
    records = []
    for lead in range(1, 10):
        valid = (init_dt + timedelta(days=lead - 1)).strftime("%Y-%m-%dT00:00:00Z")
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                records.append({
                    "grid_id": f"{lat:.6f}_{lon:.6f}",
                    "lead_day": lead,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "gnn_probability": float(g[lead - 1, i, j]),
                    "overall_bust_probability": float(overall[lead - 1, i, j]),
                    "region_id": args.region,
                    "valid_time": valid,
                })
    print(f"built {len(records)} records; posting to {args.api_url}/gnn/predictions", flush=True)

    run_id = f"{init[:8]}_{init[8:10] or '00'}"
    response = requests.post(
        f"{args.api_url}/gnn/predictions",
        json={"run_id": run_id, "model_id": args.model_id, "records": records},
        timeout=120,
    )
    print(response.status_code, response.text[:400], flush=True)
    response.raise_for_status()


if __name__ == "__main__":
    main()
