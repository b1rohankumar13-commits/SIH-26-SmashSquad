"""ERA5 western-disturbance truth: detect + track WDs on 500 hPa vorticity.

Loads the monthly ERA5 00 UTC files from scripts/download_era5_wd.py (Oct-Apr,
20-45N 40-100E) and runs src/detection/wd_tracker.py.

--tune : sweep zeta_min and report, per setting, WD tracks per Dec-Mar month that pass
         through the India box (published climatologies put ~4-7 per winter month) and the
         share of heavy (>= 35.6 mm) Dec-Mar IMD precipitation cells over NW India lying
         within GATE_KM of a WD centre the day before, of or after.
default: write data/processed/wd_truth.npz (India-affecting WD centres per day + track ids).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.detection import wd_tracker as wd
from src.detection.lps_tracker import link_tracks, strike_mask
from src.features.rainfall_labels import regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERA5 = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "wd"
IMD = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
OUT = PROJECT_ROOT / "data" / "processed" / "wd_truth.npz"
HEAVY_MM = 35.6


def load_era5() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(sorted(glob.glob(str(ERA5 / "era5_wd_*.nc"))), combine="by_coords")
    ds = ds.sel(pressure_level=500.0).sortby("latitude")
    times = pd.DatetimeIndex(ds["valid_time"].values).normalize()
    return times, ds["u"].values, ds["v"].values, ds["latitude"].values, ds["longitude"].values


def run_tracker(times, u, v, lats, lons, zeta_min):
    day = (times - pd.Timestamp("2000-01-01")).days.values
    dets = [d for k in range(len(times)) for d in wd.detect(u[k], v[k], lats, lons, int(day[k]), zeta_min=zeta_min)]
    return [tr for tr in link_tracks(dets, max_step_deg=wd.MAX_STEP_DEG) if wd.affects_india(tr)]


def gate_share(tracks, times) -> float:
    """Share of heavy Dec-Mar IMD cells over NW India within GATE_KM of a WD centre (IMD day -1..+1; the IMD day spans two 00 UTC analyses)."""
    glats, glons = canonical_centres(0.5)
    la, lo = np.meshgrid(glats, glons, indexing="ij")
    region = (la >= 24) & (lo <= 85)
    centres: dict[int, list] = {}
    for tr in tracks:
        for d in tr:
            centres.setdefault(d.t, []).append((d.lat, d.lon))
    hit = tot = 0
    for y in sorted(set(times.year)):
        f = IMD / str(y) / f"RF25_ind{y}_rfp25.nc"
        if not f.exists():
            continue
        o = regrid_obs_to_grid(xr.open_dataset(f))
        for t, field in zip(pd.DatetimeIndex(o["time"].values), o.values):
            if t.month not in (12, 1, 2, 3) or t.normalize() not in times:
                continue
            heavy = (field >= HEAVY_MM) & region
            if not heavy.any():
                continue
            k = (t.normalize() - pd.Timestamp("2000-01-01")).days
            near = strike_mask(centres.get(k - 1, []) + centres.get(k, []) + centres.get(k + 1, []), glats, glons, wd.GATE_KM)
            hit += int((heavy & near).sum()); tot += int(heavy.sum())
    return hit / max(tot, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--zeta-min", type=float, default=wd.ZETA_MIN)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    times, u, v, lats, lons = load_era5()
    winter = times.month.isin([12, 1, 2, 3])
    n_months = len(set(zip(times[winter].year, times[winter].month)))
    print(f"ERA5 {len(times)} days {times[0].date()}..{times[-1].date()} | {n_months} Dec-Mar months", flush=True)

    if args.tune:
        for z in (2.0e-5, 2.5e-5, 3.0e-5, 3.5e-5, 4.0e-5):
            tracks = run_tracker(times, u, v, lats, lons, z)
            start = pd.Timestamp("2000-01-01") + pd.to_timedelta([tr[0].t for tr in tracks], "D")
            per = np.isin(start.month, [12, 1, 2, 3]).sum() / max(n_months, 1)
            days = np.mean([len(tr) for tr in tracks]) if tracks else 0
            print(f"zeta_min {z:.1e}: {per:.1f} WDs per Dec-Mar month | mean life {days:.1f} d | "
                  f"heavy-precip cells gated {100 * gate_share(tracks, times):.1f}%", flush=True)
        return

    tracks = run_tracker(times, u, v, lats, lons, args.zeta_min)
    rows = [(d.t, d.lat, d.lon, d.zeta, k) for k, tr in enumerate(tracks) for d in tr]
    t, la, lo, z, tid = (np.array(c) for c in zip(*rows))
    np.savez(args.out, day=t.astype(np.int32), lat=la.astype(np.float32), lon=lo.astype(np.float32),
             zeta=z.astype(np.float32), track=tid.astype(np.int32), epoch="2000-01-01",
             zeta_min=args.zeta_min, gate_km=wd.GATE_KM)
    print(f"{len(tracks)} India-affecting WD tracks, {len(rows)} centre-days | heavy-precip cells gated "
          f"{100 * gate_share(tracks, times):.1f}% -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
