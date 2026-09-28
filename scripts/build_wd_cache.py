"""Build the western-disturbance (WD) precipitation-bust cache.

Event: IMD 24-h precipitation >= 35.6 mm (IMD "rather heavy") over NW India (land, >= 24N,
<= 85E) on a day with an ERA5-tracked WD centre within GATE_KM (scripts/build_wd_truth.py).
Season: inits 1 Dec - 31 Mar. Lead L verifies on IMD day init + (L-1), as for rain.
  miss        : GEFS P(>= 35.6 mm) < 0.2 and a WD event happened (NaN for non-WD heavy rain)
  false alarm : GEFS P >= 0.5 and < 35.6 mm fell
Inputs come from the GEFS rain store (precip, MSLP, 850 hPa winds, q850, gh500), plus
GEFS WD centres per member (geostrophic 500 hPa vorticity), yesterday's IMD rain and the ERA5
analysis at issue time upstream of the model domain (40-65E, which the GEFS crop cannot see).

Writes to --cache-dir: X.dat float16 [n, lead, lat, lon, C] standardized, Y_dir.dat float32
[n, lead, lat, lon, 2] (NaN off-region), P.dat float32 ensemble event probability, meta.json,
standardizer.npz.
  training cache:  --years 2000-2019 (train <= 2015, val 2016-2019)
  2022 test:       --years 2022 --store ops --standardizer <train cache>/standardizer.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from src.detection import wd_tracker as wd
from src.detection.lps_tracker import strike_mask
from src.features.rainfall_labels import regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEFS = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs"
STORES = {"reforecast": GEFS / "reforecast_2000_2019", "ops": GEFS / "operational" / "rain"}
IMD = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
ERA5 = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "wd"
TRUTH = PROJECT_ROOT / "data" / "processed" / "wd_truth.npz"

LEADS, LAT, LON = 9, 66, 70
EVENT_MM, MODERATE_MM = 35.6, 15.6
MISS_BELOW, FA_AT = 0.2, 0.5
TRAIN_LAST = 2015
SEASON = (12, 1, 2, 3)
GEFS_WD_KM = 600.0
UPSTREAM = (25.0, 40.0, 40.0, 65.0)  # lat0, lat1, lon0, lon1
CHANNELS = [
    "rain_mean", "rain_spread", "p_event", "p_moderate", "p_event_nbhd", "rain_max",
    "mslp_mean", "mslp_spread", "u850_mean", "v850_mean", "q850_mean", "ug500_mean", "vg500_mean",
    "gh500_mean", "gh500_spread", "zeta500_mean", "zeta500_spread", "p_wd_near",
    "obs_rain_d-1", "upstream_zeta", "upstream_lon", "sin_doy", "cos_doy", "lead01", "region",
]
# Ensemble-only rain signals (+ lead) for the frozen logistic baseline / wide path.
WIDE = ["rain_mean", "rain_spread", "p_event", "p_moderate", "p_event_nbhd", "lead01"]

LATS, LONS = canonical_centres(0.5)


def members(ds: xr.Dataset, var: str, level: int | None = None) -> np.ndarray:
    a = ds[var] if level is None else ds[var].sel(level=level)
    return a.isel(run=0).transpose("member", "lead", "latitude", "longitude").values.astype(np.float32)


def features(job: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Raw channel stack [lead, lat, lon, C] and ensemble event probability for one init."""
    k, path, obs_prev, up_zeta, up_lon, doy, region = job
    with xr.open_dataset(path) as ds:
        rain = np.clip(members(ds, "total_precipitation"), 0, None)
        mslp = members(ds, "mean_sea_level_pressure") / 100.0
        u8, v8, q8 = members(ds, "u_wind", 850), members(ds, "v_wind", 850), members(ds, "specific_humidity", 850)
        gh5 = members(ds, "geopotential_height", 500)
    u5, v5 = wd.geostrophic_wind(gh5, LATS, LONS)  # the store keeps 500 hPa heights, not winds
    zeta = np.empty_like(u5)
    near = np.zeros((LEADS, LAT, LON), np.float32)
    for m in range(rain.shape[0]):
        for L in range(LEADS):
            zeta[m, L] = wd.smoothed_zeta(u5[m, L], v5[m, L], LATS, LONS)
            dets = wd.detect(u5[m, L], v5[m, L], LATS, LONS, zeta_min=wd.GEO_ZETA_MIN)
            near[L] += strike_mask([(d.lat, d.lon) for d in dets], LATS, LONS, GEFS_WD_KM)
    near /= rain.shape[0]
    tropics = LATS < 15  # geostrophic balance fails towards the equator
    for a in (u5, v5, zeta):
        a[..., tropics, :] = np.nan
    p_event = (rain >= EVENT_MM).mean(0)
    nbhd = np.stack([ndimage.maximum_filter(rain[m] >= EVENT_MM, size=(1, 3, 3)) for m in range(rain.shape[0])]).mean(0)
    ang = 2 * np.pi * (doy + np.arange(LEADS)) / 365.25
    b = lambda a: np.broadcast_to(np.asarray(a, np.float32), (LEADS, LAT, LON))
    chans = [np.log1p(rain.mean(0)), np.log1p(rain.std(0)), p_event, (rain >= MODERATE_MM).mean(0), nbhd,
             np.log1p(rain.max(0)), mslp.mean(0), mslp.std(0), u8.mean(0), v8.mean(0), q8.mean(0) * 1e3,
             u5.mean(0), v5.mean(0), gh5.mean(0), gh5.std(0), zeta.mean(0) * 1e5, zeta.std(0) * 1e5, near,
             b(np.log1p(obs_prev)), b(up_zeta * 1e5), b((up_lon - 52.5) / 12.5),
             b(np.sin(ang)[:, None, None]), b(np.cos(ang)[:, None, None]),
             b(np.linspace(0, 1, LEADS)[:, None, None]), b(region)]
    return k, np.stack(chans, -1).astype(np.float32), p_event.astype(np.float32)


def load_imd(years) -> dict[pd.Timestamp, np.ndarray]:
    out = {}
    for y in years:
        f = IMD / str(y) / f"RF25_ind{y}_rfp25.nc"
        if f.exists():
            o = regrid_obs_to_grid(xr.open_dataset(f))
            for t, field in zip(pd.DatetimeIndex(o["time"].values).normalize(), o.values.astype(np.float32)):
                out[t] = field
    return out


def load_upstream() -> dict[pd.Timestamp, tuple[float, float]]:
    """ERA5 at each day: max smoothed 500 hPa vorticity upstream of the domain and its longitude."""
    ds = xr.open_mfdataset(sorted(glob.glob(str(ERA5 / "era5_wd_*.nc"))), combine="by_coords")
    ds = ds.sel(pressure_level=500.0).sortby("latitude")
    lats, lons = ds["latitude"].values, ds["longitude"].values
    a0, a1, o0, o1 = UPSTREAM
    box = ((lats >= a0) & (lats <= a1))[:, None] & ((lons >= o0) & (lons <= o1))[None, :]
    u, v = ds["u"].values, ds["v"].values
    out = {}
    for t, uk, vk in zip(pd.DatetimeIndex(ds["valid_time"].values).normalize(), u, v):
        z = np.where(box, wd.smoothed_zeta(uk, vk, lats, lons), -np.inf)
        i, j = np.unravel_index(np.argmax(z), z.shape)
        out[t] = (float(z[i, j]), float(lons[j]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\wd_20yr")
    ap.add_argument("--years", default="2000-2019")
    ap.add_argument("--store", choices=list(STORES), default="reforecast")
    ap.add_argument("--standardizer", default=None, help="reuse a training cache's standardizer.npz (test builds)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.cache_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y0, _, y1 = args.years.partition("-")
    years = range(int(y0), int(y1 or y0) + 1)

    epoch = pd.Timestamp("2000-01-01")
    tr = np.load(TRUTH)
    centres: dict[int, list] = {}
    for d, la, lo in zip(tr["day"], tr["lat"], tr["lon"]):
        centres.setdefault(int(d), []).append((float(la), float(lo)))
    imd = load_imd(range(years[0] - 1, years[-1] + 2))
    upstream = load_upstream()
    era5_days = set(upstream)
    land = np.isfinite(next(iter(imd.values())))
    la2, lo2 = np.meshgrid(LATS, LONS, indexing="ij")
    region = land & (la2 >= 24) & (lo2 <= 85)
    print(f"truth {len(tr['day'])} WD centre-days | IMD {len(imd)} days | ERA5 {len(era5_days)} days | "
          f"region cells {region.sum()} | {time.time()-t0:.0f}s", flush=True)

    files, dropped = [], 0
    for f in sorted(glob.glob(str(STORES[args.store] / "*.nc"))):
        s = Path(f).stem
        init = pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
        if init.year not in years or init.month not in SEASON:
            continue
        valid = [init + pd.Timedelta(days=L) for L in range(LEADS)]
        need_era5 = [v + pd.Timedelta(days=o) for v in valid for o in (-1, 0, 1)] + [init]
        if all(v in imd for v in valid) and init - pd.Timedelta(days=1) in imd and all(d in era5_days for d in need_era5):
            files.append((f, init))
        else:
            dropped += 1
    n = len(files)
    inits = [Path(f).stem for f, _ in files]
    is_train = np.array([i.year <= TRAIN_LAST for _, i in files])
    print(f"{n} Dec-Mar inits, train {is_train.sum()} val {(~is_train).sum()} | dropped {dropped} "
          f"(truth or obs missing)", flush=True)

    # labels
    Y = np.full((n, LEADS, LAT, LON, 2), np.nan, np.float32)
    obs = np.empty((n, LEADS, LAT, LON), np.float32)
    gate_cache: dict[int, np.ndarray] = {}

    def gate(day: pd.Timestamp) -> np.ndarray:
        k = (day - epoch).days
        if k not in gate_cache:
            c = centres.get(k - 1, []) + centres.get(k, []) + centres.get(k + 1, [])
            gate_cache[k] = strike_mask(c, LATS, LONS, wd.GATE_KM)
        return gate_cache[k]

    jobs = []
    for k, (f, init) in enumerate(files):
        for L in range(LEADS):
            obs[k, L] = imd[init + pd.Timedelta(days=L)]
        up = upstream[init]
        jobs.append((k, f, imd[init - pd.Timedelta(days=1)], up[0], up[1], init.dayofyear, region.astype(np.float32)))

    C = len(CHANNELS)
    Xraw = np.lib.format.open_memmap(out / "X_raw.npy", "w+", np.float32, (n, LEADS, LAT, LON, C))
    P = np.empty((n, LEADS, LAT, LON), np.float32)
    with Pool(args.workers) as pool:
        for done, (k, x, p) in enumerate(pool.imap_unordered(features, jobs, chunksize=4)):
            Xraw[k], P[k] = x, p
            if done % 250 == 0:
                print(f"  X {done}/{n} | {time.time()-t0:.0f}s", flush=True)
    Xraw.flush()

    n_ev = n_wd = 0
    for k, (_, init) in enumerate(files):
        for L in range(LEADS):
            o = obs[k, L]; ok = region & np.isfinite(o)
            heavy = o >= EVENT_MM
            wd_ev = heavy & gate(init + pd.Timedelta(days=L))
            n_ev += int((heavy & ok).sum()); n_wd += int((wd_ev & ok).sum())
            miss = np.where(heavy & ~wd_ev, np.nan, (P[k, L] < MISS_BELOW) & wd_ev)
            fa = (P[k, L] >= FA_AT) & ~heavy
            Y[k, L, ..., 0] = np.where(ok, miss, np.nan)
            Y[k, L, ..., 1] = np.where(ok, fa, np.nan)

    if args.standardizer:
        s = np.load(args.standardizer); mu, sd = s["mean"], s["std"]
    else:
        samp = np.asarray(Xraw[np.where(is_train)[0][::8]]).reshape(-1, C)
        mu = np.nanmean(samp, 0); sd = np.nanstd(samp, 0); sd[sd < 1e-6] = 1.0
        del samp
    X = np.memmap(out / "X.dat", np.float16, "w+", shape=(n, LEADS, LAT, LON, C))
    for s0 in range(0, n, 64):
        X[s0:s0 + 64] = np.nan_to_num((np.asarray(Xraw[s0:s0 + 64]) - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    X.flush(); del X, Xraw
    (out / "X_raw.npy").unlink()

    Yd = np.memmap(out / "Y_dir.dat", np.float32, "w+", shape=Y.shape); Yd[:] = Y; Yd.flush()
    Pd = np.memmap(out / "P.dat", np.float32, "w+", shape=P.shape); Pd[:] = P; Pd.flush()
    np.savez(out / "standardizer.npz", mean=mu, std=sd)

    def rate(rows, c):
        y = Y[rows][..., c]
        return float(np.nansum(y) / max(np.isfinite(y).sum(), 1))

    stats = {"heavy_cells": n_ev, "wd_gated_share": n_wd / max(n_ev, 1),
             "miss_rate_train": rate(is_train, 0), "fa_rate_train": rate(is_train, 1),
             "miss_rate_val": rate(~is_train, 0), "fa_rate_val": rate(~is_train, 1),
             "miss_pos_train": int(np.nansum(Y[is_train][..., 0])), "fa_pos_train": int(np.nansum(Y[is_train][..., 1])),
             "miss_pos_val": int(np.nansum(Y[~is_train][..., 0])), "fa_pos_val": int(np.nansum(Y[~is_train][..., 1]))}
    meta = {"n": n, "leads": LEADS, "lat": LAT, "lon": LON, "channels": CHANNELS, "dtype": "float16",
            "wide_channels": [CHANNELS.index(c) for c in WIDE], "wide_names": WIDE, "rule_channel": "rain_max",
            "inits": inits, "is_train": is_train.tolist(), "store": args.store, "years": args.years,
            "criterion": f"IMD >= {EVENT_MM} mm over NW India with an ERA5 WD within {wd.GATE_KM:.0f} km; "
                         f"miss P<{MISS_BELOW}, FA P>={FA_AT}",
            "stats": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("stats " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
