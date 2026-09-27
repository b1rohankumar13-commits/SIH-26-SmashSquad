"""How much of GEFS's storm-position error is systematic (learnable) vs random scatter?

For every ERA5-tracked system on a valid date, find each GEFS member's nearest detection within
1000 km at the same time and record the displacement (GEFS - ERA5). Per lead and basin, report:
  match     share of observed systems that at least one member had within 1000 km
  bias      mean displacement vector (km north / east) - what a correction could remove
  rmse      root-mean-square displacement
  sys%      |bias|^2 / mean|d|^2, the share of squared error a constant shift would fix
A conditional model could do better than a constant shift, but if sys% is small in every
bucket, most of the error is scatter and a track-correction model is unlikely to pay off.

    python -m scripts.gefs_position_error --years 2000-2015
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KM = 111.2


def basin(lat: float, lon: float) -> str:
    if lat >= 23:
        return "north/land"
    return "arabian" if lon < 77.5 else ("bay" if lon >= 80 else "peninsula")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dets", default="data/processed/gefs_lps_dets_20yr.npz")
    ap.add_argument("--truth", default="data/processed/lps_truth_all.npz")
    ap.add_argument("--years", default="2000-2015")
    ap.add_argument("--max-km", type=float, default=1000.0)
    args = ap.parse_args()
    z = np.load(PROJECT_ROOT / args.truth)
    obs: dict[str, list] = {}
    for t, la, lo in zip(z["t"], z["lat"], z["lon"]):
        obs.setdefault(str(z["dates"][t]), []).append((float(la), float(lo)))
    d = np.load(PROJECT_ROOT / args.dets)
    y0, y1 = (int(y) for y in args.years.split("-"))
    df = pd.DataFrame({k: d[k] for k in ("init", "member", "lead", "lat", "lon")})
    df = df[df["init"].str[:4].astype(int).between(y0, y1)]
    df["valid"] = (pd.to_datetime(df["init"], format="%Y%m%d")
                   + pd.to_timedelta(df["lead"].astype(int), unit="D")).dt.strftime("%Y%m%d")
    inits = sorted(set(df["init"]))
    rows = []
    groups = {k: g for k, g in df.groupby(["init", "lead"])}
    for init in inits:
        for L in range(1, 10):
            valid = (pd.Timestamp(init) + pd.Timedelta(days=L)).strftime("%Y%m%d")
            g = groups.get((init, L))
            for la, lo in obs.get(valid, []):
                hit = False
                if g is not None:
                    for _, mg in g.groupby("member"):
                        dn = (mg["lat"].values - la) * KM
                        de = (mg["lon"].values - lo) * KM * np.cos(np.deg2rad(la))
                        dist = np.hypot(dn, de); k = int(np.argmin(dist))
                        if dist[k] <= args.max_km:
                            hit = True
                            rows.append((L, basin(la, lo), valid[4:6], dn[k], de[k], dist[k], init))
                rows.append((L, basin(la, lo), valid[4:6], np.nan, np.nan, np.nan, init) if not hit else None)
    rows = [r for r in rows if r is not None]
    r = pd.DataFrame(rows, columns=["lead", "basin", "month", "dn", "de", "dist", "init"])
    print(f"{args.years}: {r['dist'].notna().sum()} member matches, {r['dist'].isna().sum()} observed systems with no member within {args.max_km:.0f} km")
    for key in ("lead", "basin"):
        print(f"\nby {key}:")
        for v, g in r.groupby(key):
            m = g.dropna()
            if len(m) < 20:
                continue
            bias = np.array([m["dn"].mean(), m["de"].mean()])
            ms = float((m["dn"] ** 2 + m["de"] ** 2).mean())
            print(f"  {str(v):11s} n {len(m):6d} | bias {bias[0]:+6.0f} km N {bias[1]:+6.0f} km E | "
                  f"rmse {np.sqrt(ms):5.0f} km | sys {100 * (bias ** 2).sum() / ms:4.1f}%")
    print("\nby basin x lead band (bias N/E km, sys%):")
    r["band"] = pd.cut(r["lead"], [0, 3, 6, 9], labels=["D1-3", "D4-6", "D7-9"])
    for (b, band), g in r.dropna().groupby(["basin", "band"], observed=True):
        if len(g) >= 50:
            bias = np.array([g["dn"].mean(), g["de"].mean()]); ms = float((g["dn"] ** 2 + g["de"] ** 2).mean())
            print(f"  {b:11s} {band}: n {len(g):5d} bias {bias[0]:+5.0f} N {bias[1]:+5.0f} E | rmse {np.sqrt(ms):4.0f} | sys {100 * (bias ** 2).sum() / ms:4.1f}%")


if __name__ == "__main__":
    main()
