"""Build 2-channel directional bust labels (miss, false-alarm) for the 20yr cache.

Reuses the existing X.dat in the cache dir (features are unchanged); writes a new
Y_dir.dat [n, 9, 66, 70, 2] in the same date order as X.dat. Parallel, page-aligned.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

from src.features.rainfall_labels import build_directional_bust_labels, regrid_obs_to_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
OBS_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
LEADS, LAT, LON = 9, 66, 70
Y_SHAPE = (LEADS, LAT, LON, 2)


def load_obs(year: int) -> xr.DataArray:
    return regrid_obs_to_grid(xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))


def shard_bounds(n: int, workers: int) -> list[int]:
    bounds = [0]
    for k in range(1, workers):
        bounds.append(min(max(round(k * n / workers / 256) * 256, bounds[-1]), n))
    bounds.append(n)
    return bounds


def build_shard(job):
    start, files, cache_dir, total = job
    labels = np.memmap(Path(cache_dir) / "Y_dir.dat", np.float32, "r+", shape=(total, *Y_SHAPE))
    obs_cache = {}
    for offset, path in enumerate(files):
        year = int(Path(path).stem[:4])
        if year not in obs_cache:
            obs_cache[year] = load_obs(year)
        with xr.open_dataset(path) as ds:
            labels[start + offset] = build_directional_bust_labels(ds, obs_cache[year])
    labels.flush()
    return start, len(files), os.getpid()


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2000-2019")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default=os.environ.get("SIH_CACHE_DIR"))
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    a, b = (int(y) for y in args.years.split("-"))
    files = sorted(f for y in range(a, b + 1) for f in glob.glob(str(FORECAST_ROOT / f"{y}*.nc")))
    n = len(files)
    print(f"{n} dates {args.years} -> Y_dir.dat [n,9,66,70,2] | {args.workers} workers", flush=True)

    np.memmap(cache_dir / "Y_dir.dat", np.float32, "w+", shape=(n, *Y_SHAPE)).flush()
    bounds = shard_bounds(n, args.workers)
    jobs = [(bounds[k], files[bounds[k]:bounds[k + 1]], str(cache_dir), n) for k in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for start, count, pid in pool.imap_unordered(build_shard, jobs):
            print(f"  shard @{start} done ({count} dates, pid {pid}) t+{time.time()-t0:.0f}s", flush=True)
    print(f"Y_dir built: {n} dates in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
