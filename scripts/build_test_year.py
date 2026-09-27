"""Build held-out test caches from operational GEFS (default 2022), frozen to training.

Nothing here is fitted on the test year: features use the training caches' standardizer,
heatwave members use the training bias correction and day-of-year normal, and labels use
the same rules as training. The test year was never used for training, early stopping,
calibration or any threshold.

  rain      -> D:\\sih-data\\caches\\rain_test_2022      X.dat (24ch), Y_dir.dat, P.dat, meta.json
  heatwave  -> D:\\sih-data\\caches\\heatwave_test_2022  X.dat (24ch), Y_dir.dat, P.dat, meta.json

    python -m scripts.build_test_year rain
    python -m scripts.build_test_year heatwave
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.detection.heatwave import detect_heatwave_bust, ensemble_heatwave_probability, heatwave_day
from src.detection.heavy_rainfall import ensemble_exceedance_probability
from src.features.build_gridded_sequences_nbhd import build_nbhd_sequences, standardize_nbhd
from src.features.rainfall_labels import build_directional_bust_labels, regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPS = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "operational"
IMD = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
ERA5 = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "tmax"
CACHES = Path(r"D:\sih-data\caches")
LEADS, LAT, LON = 9, 66, 70


def imd_years(*years: int) -> xr.DataArray:
    return xr.concat([regrid_obs_to_grid(xr.open_dataset(IMD / str(y) / f"RF25_ind{y}_rfp25.nc")) for y in years],
                     dim="time")


def build_rain(year: int) -> None:
    out = CACHES / f"rain_test_{year}"; out.mkdir(parents=True, exist_ok=True)
    std = np.load(CACHES / "rain_20yr_24ch_nbhd" / "standardizer.npz")
    files = sorted(glob.glob(str(OPS / "rain" / f"{year}*.nc")))
    n = len(files)
    obs = imd_years(year, year + 1).load()
    X = np.memmap(out / "X.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, 24))
    Y = np.memmap(out / "Y_dir.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, 2))
    P = np.empty((n, LEADS, LAT, LON), np.float32)
    t0 = time.time()
    for k, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            X[k] = standardize_nbhd(build_nbhd_sequences(ds, sequence_length=LEADS), std["mean"], std["std"])[0]
            Y[k] = build_directional_bust_labels(ds, obs)
            tp = ds["total_precipitation"].isel(run=0)
            P[k] = np.stack([ensemble_exceedance_probability(tp.sel(lead=L).values, 64.5) for L in tp["lead"].values])
        if k % 50 == 0:
            print(f"  rain {k}/{n} | {time.time()-t0:.0f}s", flush=True)
    X.flush(); Y.flush(); P.tofile(out / "P.dat")
    y = np.asarray(Y)
    meta = {"n": n, "inits": [Path(f).stem for f in files], "source": "operational GEFSv12 (noaa-gefs-pds)",
            "truth": "IMD 0.25 deg gridded rainfall", "standardizer": "rain_20yr_24ch_nbhd (training)",
            "miss_busts": int(np.nansum(y[..., 0])), "fa_busts": int(np.nansum(y[..., 1]))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"rain {year}: {n} inits, {meta['miss_busts']} miss / {meta['fa_busts']} FA busts -> {out}", flush=True)


def build_heatwave(year: int) -> None:
    from scripts.build_heatwave_cache import CHANNELS, MV_VARS, SEASON_MONTHS, WIDE
    train = CACHES / "heatwave_10yr"
    out = CACHES / f"heatwave_test_{year}"; out.mkdir(parents=True, exist_ok=True)
    normal = np.load(train / "normal.npy"); bias = np.load(train / "bias.npz")["bias"]
    std = np.load(train / "standardizer.npz")
    have_mv = {Path(f).stem for f in glob.glob(str(OPS / "mv" / f"{year}*.nc"))}
    inits = sorted(Path(f).stem for f in glob.glob(str(OPS / "tmax" / f"{year}*.nc"))
                   if int(Path(f).stem[4:6]) in SEASON_MONTHS and Path(f).stem in have_mv)
    n = len(inits)
    init_dates = pd.DatetimeIndex([pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}") for s in inits])

    ds = xr.open_mfdataset(sorted(glob.glob(str(ERA5 / f"era5_tmax_{year}*.nc"))), combine="by_coords")
    t = ds["t2m"].rename({"valid_time": "time"}).sortby("latitude").sortby("longitude")
    lats, lons = canonical_centres(0.5)
    truth = (t.interp(latitude=lats, longitude=lons, method="linear") - 273.15).load()
    truth["time"] = pd.DatetimeIndex(truth["time"].values).normalize()
    tidx, tvals = pd.DatetimeIndex(truth["time"].values), truth.values
    land = np.isfinite(regrid_obs_to_grid(xr.open_dataset(IMD / "2005" / "RF25_ind2005_rfp25.nc")).isel(time=0).values)

    valid = init_dates.values[:, None] + (np.arange(LEADS) * np.timedelta64(1, "D"))[None, :]
    vpos = tidx.get_indexer(pd.DatetimeIndex(valid.ravel())).reshape(n, LEADS)
    assert (vpos >= 0).all(), "ERA5 truth missing for some valid dates"
    obs = tvals[vpos]
    vdoy = pd.DatetimeIndex(valid.ravel()).dayofyear.values.reshape(n, LEADS)
    vmonth = pd.DatetimeIndex(valid.ravel()).month.values.reshape(n, LEADS)
    norm_v = normal[vdoy - 1]

    T = np.empty((n, 5, LEADS, LAT, LON), np.float32)
    for k, s in enumerate(inits):
        with xr.open_dataset(OPS / "tmax" / f"{s}.nc") as d:
            T[k] = d["tmax_2m"].isel(run=0).transpose("member", "lead", "latitude", "longitude").values - 273.15
    Tc = T - bias[np.arange(LEADS)[None, :], vmonth][:, None]  # training bias correction
    P = np.stack([ensemble_heatwave_probability(Tc[k], norm_v[k]) for k in range(n)])
    obs_hw = heatwave_day(obs, norm_v)
    Y = detect_heatwave_bust(P, obs_hw, np.broadcast_to(land, obs.shape) & np.isfinite(obs))

    def obs_dep(days_back: int) -> np.ndarray:
        d = init_dates - pd.Timedelta(days=days_back)
        pos = tidx.get_indexer(d)
        dep = np.full((n, LAT, LON), np.nan, np.float32)
        ok = pos >= 0
        dep[ok] = tvals[pos[ok]] - normal[d[ok].dayofyear.values - 1]
        return dep

    dep1, dep2 = obs_dep(1), obs_dep(2)
    mean_c = Tc.mean(1)
    ang = 2 * np.pi * vdoy / 365.25
    X = np.memmap(out / "X.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, len(CHANNELS)))
    b = lambda a: np.broadcast_to(a, (LEADS, LAT, LON))
    for k, s in enumerate(inits):
        with xr.open_dataset(OPS / "mv" / f"{s}.nc") as d:
            mv = [d[v].isel(run=0).transpose("member", "lead", "latitude", "longitude").values for v in MV_VARS]
        chans = [mean_c[k], Tc[k].std(0), mean_c[k] - norm_v[k], P[k], (Tc[k] >= 40).mean(0), norm_v[k]]
        for f in mv:
            chans += [f.mean(0), f.std(0)]
        chans += [b(dep1[k]), b(dep2[k]), b(np.sin(ang[k])[:, None, None]), b(np.cos(ang[k])[:, None, None]),
                  b(np.linspace(0, 1, LEADS)[:, None, None]), b(land.astype(np.float32))]
        X[k] = np.nan_to_num((np.stack(chans, -1) - std["mean"]) / std["std"], nan=0.0, posinf=0.0, neginf=0.0)
    X.flush()
    Y.astype(np.float32).tofile(out / "Y_dir.dat"); P.astype(np.float32).tofile(out / "P.dat")
    raw_bias = float(np.nanmean((T.mean(1) - obs)[:, :, land]))
    meta = {"n": n, "channels": list(CHANNELS), "wide_channels": [list(CHANNELS).index(c) for c in WIDE],
            "inits": inits, "is_train": [False] * n, "source": "operational GEFSv12 (noaa-gefs-pds)",
            "truth": "ERA5 daily max 2m T (utc+05:30)", "frozen_from": str(train),
            "raw_bias_land_c": raw_bias, "miss_busts": int(np.nansum(Y[..., 0])), "fa_busts": int(np.nansum(Y[..., 1]))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"heatwave {year}: {n} inits, raw GEFS-ERA5 bias {raw_bias:+.2f} C, "
          f"{meta['miss_busts']} miss / {meta['fa_busts']} FA busts -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hazard", choices=["rain", "heatwave"])
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()
    (build_rain if args.hazard == "rain" else build_heatwave)(args.year)


if __name__ == "__main__":
    main()
