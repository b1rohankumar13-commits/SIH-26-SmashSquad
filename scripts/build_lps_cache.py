"""Low-pressure-system strike-bust cache from the GEFS multivar store (2000-2009).

Per init (rows aligned with the rich 65-ch cache D:/sih-data/caches/rain_rich_10yr, i.e. the
sorted multivar files) and lead L (valid init + L days, 00 UTC):
  * run the LPS detector on each of the 5 members (u/v 850 hPa, MSLP) - same code, grid and
    mask as the ERA5 truth (data/processed/lps_truth.npz). The vorticity threshold is set
    per lead so GEFS detects systems as often as ERA5 does (0.436/day, 2000-2007): at the
    ERA5 threshold GEFS finds 50-75% more systems (logs/lps_gefs_frequency_check.log), which
    would otherwise show up as spurious false-alarm busts. Same idea as removing the
    heatwave Tmax bias.
  * ensemble strike probability = share of members with a centre within 300 km of the cell
  * truth strike = an ERA5 centre within 300 km on the valid date
Writes to --cache-dir:
  X_lps.dat  float32 [n, lead, lat, lon, 6] extra channels (standardized on train rows):
             p_strike300, p_strike600, zeta850 mean, zeta850 spread, msl dip mean, n_systems_domain
  Y_dir.dat  float32 [n, lead, lat, lon, 2] (miss, false_alarm)
  P.dat      float32 [n, lead, lat, lon]    ensemble strike probability (raw)
  meta.json  inits, is_train (2000-2007), channel names, climatology check
The model concatenates X_lps with the rich cache (65 + 6 = 71 channels).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from src.detection.lps_tracker import (SMOOTH_SIGMA_DEG, _annulus, _disk, detect, relative_vorticity,
                                       strike_bust_labels, strike_mask)
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MV_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019_mv"
TRUTH = PROJECT_ROOT / "data" / "processed" / "lps_truth.npz"
LEADS, LAT, LON = 9, 66, 70
# Per-lead GEFS vorticity thresholds matching ERA5 detection frequency (train years only).
GEFS_ZETA_BY_LEAD = [4.656e-05, 4.733e-05, 4.701e-05, 4.790e-05, 4.895e-05, 4.934e-05, 5.000e-05, 4.942e-05, 4.866e-05]
CHANNELS = ["p_strike300", "p_strike600", "zeta850_mean", "zeta850_spread", "msl_dip_mean", "n_systems_domain"]


def process_init(args) -> tuple[int, np.ndarray, np.ndarray, list]:
    k, path, mask, zeta_by_lead, deficit_min = args
    lats, lons = canonical_centres(0.5)
    step = 0.5
    ring = _annulus(5.0 / step, 8.0 / step); ring = ring / ring.sum()
    with xr.open_dataset(path) as ds:
        get = lambda v: ds[v].isel(run=0).transpose("member", "lead", "latitude", "longitude").values
        u, v, msl = get("ugrd_850"), get("vgrd_850"), get("pres_msl") / 100.0
    nm = u.shape[0]
    feats = np.zeros((LEADS, LAT, LON, len(CHANNELS)), np.float32)
    p300 = np.zeros((LEADS, LAT, LON), np.float32)
    ncount = []
    for L in range(LEADS):
        zs, dips, s300, s600, nsys = [], [], 0, 0, 0
        for m in range(nm):
            z = ndimage.gaussian_filter(relative_vorticity(u[m, L], v[m, L], lats, lons),
                                        SMOOTH_SIGMA_DEG / step, mode="nearest")
            zs.append(z)
            dips.append(ndimage.convolve(msl[m, L], ring, mode="nearest")
                        - ndimage.minimum_filter(msl[m, L], footprint=_disk(2.5 / step), mode="nearest"))
            c = [(d.lat, d.lon) for d in detect(u[m, L], v[m, L], msl[m, L], lats, lons, L, mask=mask,
                                                 zeta_min=zeta_by_lead[L], deficit_min=deficit_min)]
            nsys += len(c)
            s300 = s300 + strike_mask(c, lats, lons, 300.0)
            s600 = s600 + strike_mask(c, lats, lons, 600.0)
        zs = np.stack(zs)
        p300[L] = s300 / nm
        feats[L] = np.stack([s300 / nm, s600 / nm, zs.mean(0) * 1e5, zs.std(0) * 1e5,
                             np.mean(dips, 0), np.full((LAT, LON), nsys / nm)], -1)
        ncount.append(nsys / nm)
    return k, feats, p300, ncount


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=r"D:\sih-data\caches\lps_10yr")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.cache_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    truth = np.load(TRUTH)
    params = json.loads(str(truth["params"]))
    mask = truth["mask"]
    tdates = list(truth["dates"])
    lats, lons = canonical_centres(0.5)
    centres_by_date: dict[str, list] = {}
    for t, la, lo in zip(truth["t"], truth["lat"], truth["lon"]):
        centres_by_date.setdefault(tdates[t], []).append((float(la), float(lo)))

    files = sorted(f for y in range(2000, 2010) for f in glob.glob(str(MV_ROOT / f"{y}*.nc")))
    inits = [Path(f).stem[:8] for f in files]
    n = len(files)
    is_train = np.array([int(s[:4]) <= 2007 for s in inits])
    print(f"{n} inits | ERA5 zeta>={params['zeta_min']:.1e}, GEFS zeta by lead {GEFS_ZETA_BY_LEAD} "
          f"dip>={params['deficit_min']} | "
          f"{args.workers} workers", flush=True)

    X = np.memmap(out / "X_lps.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, len(CHANNELS)))
    P = np.memmap(out / "P.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON))
    Y = np.memmap(out / "Y_dir.dat", np.float32, "w+", shape=(n, LEADS, LAT, LON, 2))
    ncount = np.zeros((n, LEADS), np.float32)
    jobs = [(k, f, mask, GEFS_ZETA_BY_LEAD, params["deficit_min"]) for k, f in enumerate(files)]
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for k, feats, p300, nc in pool.map(process_init, jobs, chunksize=8):
            X[k] = feats; P[k] = p300; ncount[k] = nc
            init = pd.Timestamp(f"{inits[k][:4]}-{inits[k][4:6]}-{inits[k][6:]}")
            obs = np.stack([strike_mask(centres_by_date.get((init + pd.Timedelta(days=L + 1)).strftime("%Y%m%d"), []),
                                        lats, lons, 300.0) for L in range(LEADS)])
            Y[k] = strike_bust_labels(p300, obs)
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{n} | {time.time()-t0:.0f}s", flush=True)

    # standardize extra channels on train rows
    tr = np.where(is_train)[0][::5]
    samp = np.asarray(X[tr]).reshape(-1, len(CHANNELS))
    mu, sd = samp.mean(0), samp.std(0); sd[sd < 1e-6] = 1.0
    for s0 in range(0, n, 128):
        X[s0:s0 + 128] = (np.asarray(X[s0:s0 + 128]) - mu) / sd
    X.flush(); P.flush(); Y.flush()
    np.savez(out / "standardizer.npz", mean=mu, std=sd)

    era5_per_day = np.mean([len(centres_by_date.get(d, [])) for d in tdates])
    Yv = np.asarray(Y)
    stats = {"gefs_systems_per_member_by_lead": ncount.mean(0).round(3).tolist(),
             "era5_systems_per_day": round(float(era5_per_day), 3),
             "miss_rate_train": float(np.nanmean(Yv[is_train][..., 0])),
             "fa_rate_train": float(np.nanmean(Yv[is_train][..., 1])),
             "miss_rate_val": float(np.nanmean(Yv[~is_train][..., 0])),
             "fa_rate_val": float(np.nanmean(Yv[~is_train][..., 1]))}
    meta = {"n": n, "leads": LEADS, "lat": LAT, "lon": LON, "channels": CHANNELS, "inits": inits,
            "is_train": is_train.tolist(), "rich_cache": r"D:\sih-data\caches\rain_rich_10yr",
            "tracker": {**params, "gefs_zeta_by_lead": GEFS_ZETA_BY_LEAD}, "stats": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("stats " + json.dumps(stats), flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
