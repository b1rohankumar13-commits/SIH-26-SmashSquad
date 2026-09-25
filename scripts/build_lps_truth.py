"""ERA5 low-pressure-system truth: detect + track on the model grid, validate vs IBTrACS.

Loads the monthly ERA5 00 UTC files (u/v 850 hPa, MSLP), regrids them to the 0.5 deg
model grid (identical to GEFS), and runs src/detection/lps_tracker.py.

--tune : sweep (zeta_min, deficit_min) and report, per setting,
           * IBTrACS hit rate: share of in-mask IMD depression-or-stronger 00 UTC fixes
             (2000-2009) with an ERA5 detection within 300 km the same day
           * LPS per JJAS season (tracks lasting >= 2 days, genesis in June-September)
         Published catalogues put ~10-14 monsoon LPS per season; IBTrACS hit >= 90% wanted.
default: write data/processed/lps_truth.npz for the chosen thresholds (detections per day).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.detection.lps_tracker import _gc_deg, centre_mask, detect, link_tracks
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERA5 = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "lps"
IBTRACS = Path(r"D:\sih-data\raw\observations\ibtracs\ibtracs.NI.list.v04r01.csv")
OUT = PROJECT_ROOT / "data" / "processed" / "lps_truth.npz"


def load_era5_on_grid() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lats, lons = canonical_centres(0.5)
    pl = xr.open_mfdataset(sorted(glob.glob(str(ERA5 / "era5_pl_*.nc"))), combine="by_coords")
    sl = xr.open_mfdataset(sorted(glob.glob(str(ERA5 / "era5_sl_*.nc"))), combine="by_coords")
    pl = pl.squeeze("pressure_level", drop=True).sortby("latitude").interp(latitude=lats, longitude=lons)
    sl = sl.sortby("latitude").interp(latitude=lats, longitude=lons)
    times = pd.DatetimeIndex(pl["valid_time"].values)
    assert (times == pd.DatetimeIndex(sl["valid_time"].values)).all()
    return times, pl["u"].values, pl["v"].values, sl["msl"].values / 100.0, (lats, lons)


def orography_on_grid(lats, lons) -> np.ndarray:
    o = xr.open_dataset(ERA5 / "era5_orography.nc").squeeze(drop=True).sortby("latitude")
    return (o["z"] / 9.80665).interp(latitude=lats, longitude=lons).values


def ibtracs_fixes(times: pd.DatetimeIndex, mask: np.ndarray, lats, lons) -> pd.DataFrame:
    ib = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False,
                     usecols=["SID", "ISO_TIME", "LAT", "LON", "NEWDELHI_GRADE"])
    ib["t"] = pd.to_datetime(ib["ISO_TIME"])
    ib = ib[(ib["t"].dt.hour == 0) & ib["t"].dt.normalize().isin(times.normalize())]
    ib = ib[ib["NEWDELHI_GRADE"].isin(["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SUCS"])]
    ib["LAT"] = pd.to_numeric(ib["LAT"]); ib["LON"] = pd.to_numeric(ib["LON"])
    i = np.clip(np.round((ib["LAT"] - lats[0]) / 0.5).astype(int), 0, len(lats) - 1)
    j = np.clip(np.round((ib["LON"] - lons[0]) / 0.5).astype(int), 0, len(lons) - 1)
    inside = (ib["LAT"].between(lats[0], lats[-1]) & ib["LON"].between(lons[0], lons[-1])).to_numpy()
    return ib[inside & mask[i, j]]


def run(times, u, v, msl, lats, lons, mask, zeta_min, deficit_min):
    dets = []
    for t in range(len(times)):
        dets += detect(u[t], v[t], msl[t], lats, lons, t, mask=mask, zeta_min=zeta_min, deficit_min=deficit_min)
    return dets


def score(dets, tracks, times, ib) -> dict:
    by_t: dict[int, list] = {}
    for d in dets:
        by_t.setdefault(d.t, []).append(d)
    tpos = {t.normalize(): k for k, t in enumerate(times)}
    hits = 0
    for _, r in ib.iterrows():
        k = tpos[r["t"].normalize()]
        hits += any(_gc_deg(r["LAT"], r["LON"], d.lat, d.lon) * 111.2 <= 300 for d in by_t.get(k, []))
    long_tracks = [tr for tr in tracks if len(tr) >= 2]
    jjas = [tr for tr in long_tracks if times[tr[0].t].month in (6, 7, 8, 9)]
    years = sorted(set(times.year))
    return {"ibtracs_hit": hits / max(len(ib), 1), "n_fixes": len(ib),
            "lps_per_jjas": len(jjas) / len(years),
            "tracks_per_year": len(long_tracks) / len(years),
            "det_days_share": len(by_t) / len(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--zeta-min", type=float, default=None)
    ap.add_argument("--deficit-min", type=float, default=None)
    args = ap.parse_args()
    times, u, v, msl, (lats, lons) = load_era5_on_grid()
    mask = centre_mask(lats, lons, orography_on_grid(lats, lons))
    ib = ibtracs_fixes(times, mask, lats, lons)
    print(f"ERA5 {times[0].date()}..{times[-1].date()} ({len(times)} days) | IBTrACS D+ fixes in mask: {len(ib)} "
          f"({ib['SID'].nunique()} systems)", flush=True)
    if args.tune:
        rows = []
        for z in (1.5e-5, 2e-5, 3e-5, 4e-5, 5e-5):
            for dp in (1.0, 2.0, 3.0):
                dets = run(times, u, v, msl, lats, lons, mask, z, dp)
                s = score(dets, link_tracks(dets), times, ib)
                rows.append({"zeta_min": z, "deficit_min": dp, **s})
                print(f"zeta>={z:.1e} dp>={dp:.0f} | IBTrACS hit {100*s['ibtracs_hit']:.0f}% | "
                      f"LPS/JJAS {s['lps_per_jjas']:.1f} | tracks/yr {s['tracks_per_year']:.1f} | "
                      f"days with a detection {100*s['det_days_share']:.0f}%", flush=True)
        pd.DataFrame(rows).to_csv(PROJECT_ROOT / "logs" / "lps_tracker_tuning.csv", index=False)
        return
    z, dp = args.zeta_min, args.deficit_min
    dets = run(times, u, v, msl, lats, lons, mask, z, dp)
    tracks = link_tracks(dets)
    s = score(dets, tracks, times, ib)
    track_id = {id(d): k for k, tr in enumerate(tracks) for d in tr}
    track_len = {k: len(tr) for k, tr in enumerate(tracks)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, dates=np.asarray(times.strftime("%Y%m%d"), dtype=str),
             t=np.array([d.t for d in dets]), lat=np.array([d.lat for d in dets]),
             lon=np.array([d.lon for d in dets]), zeta=np.array([d.zeta for d in dets]),
             deficit=np.array([d.deficit for d in dets]),
             track=np.array([track_id[id(d)] for d in dets]),
             track_len=np.array([track_len[track_id[id(d)]] for d in dets]),
             mask=mask, params=json.dumps({"zeta_min": z, "deficit_min": dp, **s}))
    print(f"wrote {OUT} | {len(dets)} detections, {len(tracks)} tracks | " + json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
