"""Build the gridded feature/label cache with a pool of decode workers.

The serial cache build in run_graphnet_pilot streams one netCDF at a time; the
decode dominates and the disk has headroom, so this fans the dates out across a
few worker processes. Each worker writes a contiguous, page-aligned slice of the
shared memmaps, so there is no concatenation pass and no duplicated storage. The
standardizer is fit once, serially, on the training split before the fan-out so
every worker transforms with identical statistics.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

from src.features.build_gridded_sequences import ChannelStandardizer, N_CHANNELS, build_gridded_sequences
from src.features.rainfall_labels import build_bust_labels, regrid_obs_to_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
OBS_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"

LEADS, LAT, LON = 9, 66, 70
X_SHAPE = (LEADS, LAT, LON, N_CHANNELS)
Y_SHAPE = (LEADS, LAT, LON, 1)
# Row byte sizes; 256-row boundaries are a whole number of 4 KiB pages for both
# (verified: 3_160_080 * 256 and 166_320 * 256 are multiples of 4096), so shards
# never share a page and can be written concurrently without a race.
PAGE_ALIGN_ROWS = 256


def load_obs(year: int) -> xr.DataArray:
    return regrid_obs_to_grid(xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))


def fit_standardizer(train_files: list[str], sample_size: int = 300) -> ChannelStandardizer:
    stride = max(1, len(train_files) // sample_size)
    seqs = []
    for path in train_files[::stride][:sample_size]:
        with xr.open_dataset(path) as ds:
            seqs.append(build_gridded_sequences(ds, sequence_length=int(ds.sizes["lead"])).load())
    sample = xr.concat(seqs, dim="run")
    return ChannelStandardizer.fit(sample, list(sample["run"].values))


def shard_bounds(n: int, workers: int) -> list[int]:
    """Split [0, n) into `workers` ranges whose starts are 256-row aligned."""
    bounds = [0]
    for k in range(1, workers):
        cut = round(k * n / workers / PAGE_ALIGN_ROWS) * PAGE_ALIGN_ROWS
        bounds.append(min(max(cut, bounds[-1]), n))
    bounds.append(n)
    return bounds


def build_shard(job: tuple) -> tuple[int, int, int]:
    """Decode one contiguous slice of dates into the shared memmaps."""
    start, files, mean, std, cache_dir, total = job
    standardizer = ChannelStandardizer(mean=mean, std=std)
    features = np.memmap(Path(cache_dir) / "X.dat", np.float32, "r+", shape=(total, *X_SHAPE))
    labels = np.memmap(Path(cache_dir) / "Y.dat", np.float32, "r+", shape=(total, *Y_SHAPE))
    obs_cache: dict[int, xr.DataArray] = {}
    for offset, path in enumerate(files):
        year = int(Path(path).stem[:4])
        if year not in obs_cache:
            obs_cache[year] = load_obs(year)
        with xr.open_dataset(path) as ds:
            seq = build_gridded_sequences(ds, sequence_length=int(ds.sizes["lead"]))
            features[start + offset] = standardizer.transform(seq)[0]
            labels[start + offset] = build_bust_labels(ds, obs_cache[year])[..., None]
    features.flush()
    labels.flush()
    return start, len(files), os.getpid()


def main() -> None:
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=str, default="2000-2019")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("SIH_CACHE_DIR"))
    args = parser.parse_args()
    if not args.cache_dir:
        raise SystemExit("Set --cache-dir or SIH_CACHE_DIR")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_year, end_year = (int(y) for y in args.years.split("-"))
    files = sorted(
        f for year in range(start_year, end_year + 1)
        for f in glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    )
    n = len(files)
    n_train = max(1, min(int(0.8 * n), n - 1))
    print(f"{n} dates {args.years} | train {n_train} val {n - n_train} | {args.workers} workers", flush=True)

    t0 = time.time()
    standardizer = fit_standardizer(files[:n_train])
    print(f"standardizer fitted in {time.time() - t0:.0f}s; allocating memmaps ...", flush=True)

    # Pre-create the shared memmaps at full size so workers can open them "r+".
    np.memmap(cache_dir / "X.dat", np.float32, "w+", shape=(n, *X_SHAPE)).flush()
    np.memmap(cache_dir / "Y.dat", np.float32, "w+", shape=(n, *Y_SHAPE)).flush()

    bounds = shard_bounds(n, args.workers)
    jobs = [
        (bounds[k], files[bounds[k]:bounds[k + 1]], standardizer.mean, standardizer.std, str(cache_dir), n)
        for k in range(args.workers)
    ]
    print(f"shard rows: {[ (bounds[k], bounds[k+1]) for k in range(args.workers) ]}", flush=True)

    t1 = time.time()
    with mp.Pool(args.workers) as pool:
        for start, count, pid in pool.imap_unordered(build_shard, jobs):
            print(f"  shard @{start} done ({count} dates, pid {pid}) t+{time.time() - t1:.0f}s", flush=True)
    build_s = time.time() - t1
    np.savez(cache_dir / "standardizer.npz", mean=standardizer.mean, std=standardizer.std)
    print(f"cache built: {n} dates in {build_s:.0f}s ({n / build_s:.2f} dates/s)", flush=True)


if __name__ == "__main__":
    main()
