"""Storm-environment channels for the LPS model: MJO state at issue time + seasonal storm climatology.

Long-range storm busts are blurry because the model knows *whether* the situation favours a
storm but not *where*. Two cheap, legitimately-known-at-issue-time clues:
  * MJO (BoM RMM index, data/raw/indices/rmm.74toRealtime.txt) on the day before issue - the
    real-time index is published with about a day's delay. RMM1, RMM2 and amplitude; the MJO
    is the main known source of North Indian Ocean genesis predictability at 1-2 weeks.
  * Seasonal climatology: how often an ERA5-tracked system passes within 300 km of each cell
    around the valid day of year (+-15 days), from training years only, plus sin/cos of the
    valid day of year.

Writes X_env.dat [n, lead, lat, lon, 6] next to X_lps.dat, standardized with training-row
stats, and adds "env_channels" to meta.json. For a test cache pass --from-cache so the
climatology and standardizer come from the training cache:

    python -m scripts.build_lps_env --cache-dir D:/sih-data/caches/lps_20yr
    python -m scripts.build_lps_env --cache-dir D:/sih-data/caches/lps_test_2022 --from-cache D:/sih-data/caches/lps_20yr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from src.detection.lps_tracker import strike_mask
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RMM = PROJECT_ROOT / "data" / "raw" / "indices" / "rmm.74toRealtime.txt"
ENV = ["rmm1", "rmm2", "rmm_amp", "sin_doy", "cos_doy", "clim_strike300"]


def load_rmm() -> pd.DataFrame:
    rows = []
    for line in RMM.read_text().splitlines()[2:]:
        f = line.split()
        if len(f) >= 7:
            rows.append((pd.Timestamp(int(f[0]), int(f[1]), int(f[2])), float(f[3]), float(f[4]), float(f[6])))
    df = pd.DataFrame(rows, columns=["date", "rmm1", "rmm2", "amp"]).set_index("date")
    return df.mask(df.abs() > 100)  # missing = 1e36 / 999


def climatology(truth: Path, last_year: int) -> np.ndarray:
    """[366, lat, lon] share of training days with a system within 300 km, +-15-day circular smoothing."""
    z = np.load(truth)
    lats, lons = canonical_centres(0.5)
    dates = pd.to_datetime(z["dates"], format="%Y%m%d")
    keep = dates.year <= last_year
    hits = np.zeros((366, len(lats), len(lons)), np.float64); days = np.zeros(366)
    by_t: dict[int, list] = {}
    for t, la, lo in zip(z["t"], z["lat"], z["lon"]):
        by_t.setdefault(int(t), []).append((float(la), float(lo)))
    for t in np.where(keep)[0]:
        d = dates[t].dayofyear - 1
        days[d] += 1
        if t in by_t:
            hits[d] += strike_mask(by_t[t], lats, lons, 300.0)
    k = np.ones(31) / 31
    hits = ndimage.convolve1d(hits, k, axis=0, mode="wrap"); days = ndimage.convolve1d(days, k, mode="wrap")
    return (hits / np.maximum(days, 1e-9)[:, None, None]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--from-cache", default=None, help="Training cache to take climatology/standardizer from.")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W = meta["n"], meta["leads"], meta["lat"], meta["lon"]
    src = Path(args.from_cache) if args.from_cache else cache
    if args.from_cache:
        clim = np.load(src / "clim_strike300.npy"); std = np.load(src / "env_standardizer.npz")
    else:
        last = max(int(s[:4]) for s, tr in zip(meta["inits"], meta["is_train"]) if tr)
        truth = Path(meta.get("truth", "data/processed/lps_truth_20yr.npz"))
        clim = climatology(truth if truth.is_absolute() else PROJECT_ROOT / truth, last)
        np.save(cache / "clim_strike300.npy", clim)
        std = None

    rmm = load_rmm()
    inits = pd.to_datetime(meta["inits"], format="%Y%m%d")
    prev = rmm.reindex(inits - pd.Timedelta(days=1))
    missing = int(prev["rmm1"].isna().sum())
    prev = prev.fillna(0.0)
    X = np.memmap(cache / "X_env.dat", np.float32, "w+", shape=(n, L, H, W, len(ENV)))
    for i in range(n):
        valid = inits[i] + pd.to_timedelta(np.arange(1, L + 1), unit="D")
        doy = valid.dayofyear.values
        ang = 2 * np.pi * doy / 365.25
        x = np.empty((L, H, W, len(ENV)), np.float32)
        x[..., 0], x[..., 1], x[..., 2] = prev["rmm1"].iloc[i], prev["rmm2"].iloc[i], prev["amp"].iloc[i]
        x[..., 3] = np.sin(ang)[:, None, None]; x[..., 4] = np.cos(ang)[:, None, None]
        x[..., 5] = clim[doy - 1]
        X[i] = x
    if std is None:
        tr = np.where(np.asarray(meta["is_train"]))[0][::5]
        samp = np.asarray(X[tr]).reshape(-1, len(ENV))
        mu, sd = samp.mean(0), samp.std(0); sd[sd < 1e-6] = 1.0
        np.savez(cache / "env_standardizer.npz", mean=mu, std=sd)
    else:
        mu, sd = std["mean"], std["std"]
    for s in range(0, n, 256):
        X[s:s + 256] = (np.asarray(X[s:s + 256]) - mu) / sd
    X.flush()
    meta["env_channels"] = ENV
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"X_env.dat [{n}, {L}, {H}, {W}, {len(ENV)}] | inits without an RMM value: {missing} | "
          f"climatology max {float(clim.max()):.3f}", flush=True)


if __name__ == "__main__":
    main()
