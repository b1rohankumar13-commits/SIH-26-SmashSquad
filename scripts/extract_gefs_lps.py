"""Extract every GEFS member's low-pressure-system detections (same tracker, mask and per-lead
thresholds as the strike cache) into one file, for map tracks and position-error analysis.

    python -m scripts.extract_gefs_lps --store rain --years 2000-2019 --out data/processed/gefs_lps_dets_20yr.npz
    python -m scripts.extract_gefs_lps --store ops_rain --years 2022-2022 --out data/processed/gefs_lps_dets_2022.npz

Output arrays, one row per detection: init (YYYYMMDD), member (0-4), lead (1-9; valid init + lead
days, 00 UTC), lat, lon, zeta (s^-1), deficit (hPa).
"""

from __future__ import annotations

import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.build_lps_cache import GEFS_ZETA_BY_LEAD, STORES
from src.detection.lps_tracker import detect
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUTH = PROJECT_ROOT / "data" / "processed" / "lps_truth_20yr.npz"


def process(args) -> list[tuple]:
    path, store, mask, deficit_min = args
    _, get_u, get_v, get_msl, _, _ = STORES[store]
    lats, lons = canonical_centres(0.5)
    init = Path(path).stem[:8]
    out = []
    with xr.open_dataset(path) as ds:
        get = lambda da: da.isel(run=0).transpose("member", "lead", "latitude", "longitude").values
        u, v, msl = get(get_u(ds)), get(get_v(ds)), get(get_msl(ds)) / 100.0
    for m in range(u.shape[0]):
        for L in range(u.shape[1]):
            for d in detect(u[m, L], v[m, L], msl[m, L], lats, lons, L + 1, mask=mask,
                            zeta_min=GEFS_ZETA_BY_LEAD[L], deficit_min=deficit_min):
                out.append((init, m, L + 1, d.lat, d.lon, d.zeta, d.deficit))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", choices=list(STORES), default="rain")
    ap.add_argument("--years", default="2000-2019")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    truth = np.load(TRUTH)
    import json
    deficit_min = json.loads(str(truth["params"]))["deficit_min"]
    root = STORES[args.store][0]
    y0, y1 = (int(y) for y in args.years.split("-"))
    files = sorted(f for y in range(y0, y1 + 1) for f in glob.glob(str(root / f"{y}*.nc")))
    t0, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, res in enumerate(pool.map(process, [(f, args.store, truth["mask"], deficit_min) for f in files],
                                         chunksize=8)):
            rows += res
            if k % 500 == 0:
                print(f"  {k}/{len(files)} | {len(rows)} detections | {time.time()-t0:.0f}s", flush=True)
    cols = list(zip(*rows)) if rows else [[]] * 7
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    np.savez(out, init=np.asarray(cols[0], dtype="U8"), member=np.asarray(cols[1], np.int8),
             lead=np.asarray(cols[2], np.int8), lat=np.asarray(cols[3], np.float32),
             lon=np.asarray(cols[4], np.float32), zeta=np.asarray(cols[5], np.float32),
             deficit=np.asarray(cols[6], np.float32))
    print(f"{len(files)} inits, {len(rows)} detections -> {out} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
