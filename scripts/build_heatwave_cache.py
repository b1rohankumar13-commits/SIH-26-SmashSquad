"""Build the heatwave-bust cache (GEFS Tmax + context vars -> ERA5 Tmax truth).

Season: inits 1 Mar - 30 Jun, 2000-2009 (valid days to 8 Jul). Train 2000-2007, val 2008-2009.
Truth: ERA5 daily Tmax (IST day), regridded to the 0.5 deg grid, India land mask from IMD.
Lead L verifies on init + (L-1) days (GEFS daily max over UTC day L).

Writes to --cache-dir:
  X.dat      float32 [n, lead, lat, lon, C]  standardized (train stats)
  Y_dir.dat  float32 [n, lead, lat, lon, 2]  (miss, false_alarm), NaN off-mask
  P.dat      float32 [n, lead, lat, lon]     ensemble heatwave probability (raw)
  meta.json  channels, dates, is_train, wide-channel indices, base rates
  standardizer.npz, bias.npz, normal.npy
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
from src.features.rainfall_labels import regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEFS = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs"
TMAX_ROOT = GEFS / "reforecast_2000_2019_tmax"
MV_ROOT = GEFS / "reforecast_2000_2019_mv"
ERA5_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "tmax"
IMD_RAIN = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall" / "2005" / "RF25_ind2005_rfp25.nc"

LEADS, LAT, LON = 9, 66, 70
TRAIN_LAST_YEAR = 2007
SEASON_MONTHS = (3, 4, 5, 6)
MV_VARS = ("tmp_2m", "tmp_850", "hgt_500", "hgt_300", "spfh_2m", "pres_msl")
CHANNELS = (
    ["tmax_mean", "tmax_spread", "tmax_departure", "p_heatwave", "p_ge40", "tmax_normal"]
    + [f"{v}_{s}" for v in MV_VARS for s in ("mean", "spread")]
    + ["obs_departure_d-1", "obs_departure_d-2", "sin_doy", "cos_doy", "lead01", "land"]
)
# Ensemble-only signals (+ lead) for the frozen logistic baseline / wide path.
WIDE = ["tmax_mean", "tmax_spread", "tmax_departure", "p_heatwave", "p_ge40", "lead01"]


def load_truth() -> xr.DataArray:
    ds = xr.open_mfdataset(sorted(glob.glob(str(ERA5_ROOT / "*.nc"))), combine="by_coords")
    t = ds["t2m"].rename({"valid_time": "time"}).sortby("latitude").sortby("longitude")
    lats, lons = canonical_centres(0.5)
    t = t.interp(latitude=lats, longitude=lons, method="linear") - 273.15
    t["time"] = pd.DatetimeIndex(t["time"].values).normalize()
    return t.load()


def doy_normal(truth: xr.DataArray, half_window: int = 15) -> np.ndarray:
    """[366, lat, lon] smoothed day-of-year normal from training years."""
    train = truth.sel(time=truth["time"].dt.year <= TRAIN_LAST_YEAR)
    raw = train.groupby("time.dayofyear").mean().reindex(dayofyear=np.arange(1, 367)).values
    raw = pd.DataFrame(raw.reshape(366, -1)).interpolate(limit_direction="both").values
    padded = np.concatenate([raw[-half_window:], raw, raw[:half_window]])
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    smooth = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, padded)
    return smooth.reshape(366, LAT, LON).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\heatwave_10yr")
    args = ap.parse_args()
    out = Path(args.cache_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mv_have = {Path(f).stem for f in glob.glob(str(MV_ROOT / "*.nc"))}
    inits = sorted(Path(f).stem for f in glob.glob(str(TMAX_ROOT / "*.nc"))
                   if int(Path(f).stem[4:6]) in SEASON_MONTHS and int(Path(f).stem[:4]) <= 2009)
    dropped = [i for i in inits if i not in mv_have]
    inits = [i for i in inits if i in mv_have]
    n = len(inits)
    init_dates = pd.DatetimeIndex([pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}") for s in inits])
    is_train = np.asarray(init_dates.year <= TRAIN_LAST_YEAR)
    print(f"{n} season inits (Mar-Jun 2000-2009), train {is_train.sum()} val {(~is_train).sum()} "
          f"| dropped {len(dropped)} missing from multivar store", flush=True)

    truth = load_truth()
    land = np.isfinite(regrid_obs_to_grid(xr.open_dataset(IMD_RAIN)).isel(time=0).values)
    normal = doy_normal(truth)
    tidx = pd.DatetimeIndex(truth["time"].values)
    tvals = truth.values
    print(f"truth {tvals.shape} {tidx[0].date()}..{tidx[-1].date()} | land cells {land.sum()} "
          f"| {time.time()-t0:.0f}s", flush=True)

    # valid dates [n, lead]; obs [n, lead, lat, lon]; normal at valid DOY
    valid = init_dates.values[:, None] + (np.arange(LEADS) * np.timedelta64(1, "D"))[None, :]
    vpos = tidx.get_indexer(pd.DatetimeIndex(valid.ravel())).reshape(n, LEADS)
    assert (vpos >= 0).all(), "truth missing for some valid dates"
    obs = tvals[vpos]
    vdoy = pd.DatetimeIndex(valid.ravel()).dayofyear.values.reshape(n, LEADS)
    vmonth = pd.DatetimeIndex(valid.ravel()).month.values.reshape(n, LEADS)
    norm_v = normal[vdoy - 1]

    # GEFS Tmax members [n, member, lead, lat, lon] in C
    T = np.empty((n, 5, LEADS, LAT, LON), np.float32)
    for k, s in enumerate(inits):
        with xr.open_dataset(TMAX_ROOT / f"{s}.nc") as ds:
            T[k] = ds["tmax_2m"].isel(run=0).transpose("member", "lead", "latitude", "longitude").values - 273.15
    print(f"GEFS tmax loaded {T.shape} | {time.time()-t0:.0f}s", flush=True)

    # mean bias per (lead, valid month, cell) from training inits, then remove it
    err = T.mean(1) - obs
    bias = np.zeros((LEADS, 13, LAT, LON), np.float32)
    for L in range(LEADS):
        for m in np.unique(vmonth[:, L]):
            sel = is_train & (vmonth[:, L] == m)
            bias[L, m] = np.nanmean(err[sel, L], axis=0)
    Tc = T - bias[np.arange(LEADS)[None, :], vmonth][:, None]
    raw_bias_land = float(np.nanmean(err[is_train][:, :, land]))
    print(f"GEFS-ERA5 Tmax mean bias over land (train) {raw_bias_land:+.2f} C -> removed per lead/month/cell",
          flush=True)

    P = np.stack([ensemble_heatwave_probability(Tc[k], norm_v[k]) for k in range(n)])
    obs_hw = heatwave_day(obs, norm_v)
    lab_valid = np.broadcast_to(land, obs.shape) & np.isfinite(obs)
    Y = detect_heatwave_bust(P, obs_hw, lab_valid)

    # observed persistence: departure on the two days before init (known at issue time)
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
    Xs = np.memmap(out / "X.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, len(CHANNELS)))
    for k, s in enumerate(inits):
        with xr.open_dataset(MV_ROOT / f"{s}.nc") as ds:
            mv = [ds[v].isel(run=0).transpose("member", "lead", "latitude", "longitude").values for v in MV_VARS]
        chans = [mean_c[k], Tc[k].std(0), mean_c[k] - norm_v[k], P[k], (Tc[k] >= 40).mean(0), norm_v[k]]
        for f in mv:
            chans += [f.mean(0), f.std(0)]
        b = lambda a: np.broadcast_to(a, (LEADS, LAT, LON))
        chans += [b(dep1[k]), b(dep2[k]),
                  b(np.sin(ang[k])[:, None, None]), b(np.cos(ang[k])[:, None, None]),
                  b(np.linspace(0, 1, LEADS)[:, None, None]), b(land.astype(np.float32))]
        Xs[k] = np.stack(chans, -1)
        if k % 200 == 0:
            print(f"  X {k}/{n} | {time.time()-t0:.0f}s", flush=True)

    # standardize with training-row stats over land cells
    tr = np.where(is_train)[0][::4]
    samp = np.asarray(Xs[tr])[:, :, land]
    mu = np.nanmean(samp, axis=(0, 1, 2)); sd = np.nanstd(samp, axis=(0, 1, 2))
    sd[sd < 1e-6] = 1.0
    for s0 in range(0, n, 64):
        blk = (np.asarray(Xs[s0:s0 + 64]) - mu) / sd
        Xs[s0:s0 + 64] = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
    Xs.flush(); del Xs

    Yd = np.memmap(out / "Y_dir.dat", np.float32, "w+", shape=Y.shape); Yd[:] = Y; Yd.flush()
    Pd = np.memmap(out / "P.dat", np.float32, "w+", shape=P.shape); Pd[:] = P; Pd.flush()
    np.savez(out / "standardizer.npz", mean=mu, std=sd)
    np.savez(out / "bias.npz", bias=bias)
    np.save(out / "normal.npy", normal)

    def rate(mask_rows, c):
        y = Y[mask_rows][..., c]
        return float(np.nansum(y) / np.isfinite(y).sum())

    obs_rate = float(obs_hw[:, :, land].mean())
    stats = {"obs_heatwave_cell_rate": obs_rate,
             "miss_rate_train": rate(is_train, 0), "fa_rate_train": rate(is_train, 1),
             "miss_rate_val": rate(~is_train, 0), "fa_rate_val": rate(~is_train, 1),
             "miss_pos_val": int(np.nansum(Y[~is_train][..., 0])), "fa_pos_val": int(np.nansum(Y[~is_train][..., 1])),
             "raw_bias_land_c": raw_bias_land}
    meta = {"n": n, "leads": LEADS, "lat": LAT, "lon": LON, "channels": CHANNELS,
            "wide_channels": [CHANNELS.index(c) for c in WIDE], "wide_names": WIDE,
            "inits": inits, "is_train": is_train.tolist(), "dropped_inits": dropped,
            "criterion": "Tmax>=40 & (dep>=4.5 | Tmax>=45); miss P<0.2, FA P>=0.5; bias-corrected members",
            "truth": "ERA5 daily max 2m T (utc+05:30), India land mask from IMD", "stats": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("stats " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
