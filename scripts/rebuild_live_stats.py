"""Rebuild the small statistics the live pipeline needs, without the 20-year training caches.

The live rain and storm models read three things from D:\\sih-data\\caches that were fitted on the
GEFSv12 reforecast: the per-channel input standardizers (rain 24 ch, storm 6 ch, storm "env"
6 ch), the seasonal storm climatology, and the storm cache's meta.json. If those caches are lost,
this script recreates them:

- storm climatology and env standardizer: exact (from data/processed/lps_truth_20yr.npz, the BoM
  RMM index and the daily 2000-2015 training inits, every 5th, as in build_lps_env.py);
- rain and storm standardizers: approximate, re-fitted on a sample of recent operational GEFS runs
  (two per month over a year, rain store only) instead of the original 300 reforecast inits.

Each rebuilt folder gets a REBUILT.txt saying so.

    python -m scripts.rebuild_live_stats                 # download sample (~140 MB/run) + fit
    python -m scripts.rebuild_live_stats --no-download   # fit on whatever sample is on disk
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.build_lps_cache import CHANNELS as LPS_CHANNELS, GEFS_ZETA_BY_LEAD, process_init
from scripts.build_lps_env import ENV, climatology, load_rmm
from src.acquisition.download_gefs_operational import build_session, execute_init, load_config
from src.features.build_gridded_sequences import ChannelStandardizer
from src.features.build_gridded_sequences_nbhd import NBHD_CHANNEL_NAMES, build_nbhd_sequences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHES = Path(r"D:\sih-data\caches")
RAIN_CACHE = CACHES / "rain_20yr_24ch_nbhd"
LPS_CACHE = CACHES / "lps_20yr"
STORE = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "operational" / "rain"
LPS_TRUTH = PROJECT_ROOT / "data" / "processed" / "lps_truth_20yr.npz"
TRAIN_LAST = 2015
LEADS = 9


def sample_dates(end: str = "2026-10-01") -> list[str]:
    """1st and 16th of each of the 12 months before `end`."""
    months = pd.date_range(end=pd.Timestamp(end), periods=12, freq="MS")
    return [d.strftime("%Y%m%d") for m in months for d in (m, m + pd.Timedelta(days=15))]


def download(dates: list[str]) -> None:
    config = copy.deepcopy(load_config())
    for name, store in config["stores"].items():  # rain store only, also in Mar-Jun
        if name != "rain":
            store["months"] = []
    session = build_session()
    for k, d in enumerate(dates, 1):
        if (STORE / f"{d}00.nc").exists():
            continue
        t0 = time.time()
        execute_init(session, config, d + "00", config["members"])
        print(f"  [{k}/{len(dates)}] {d} {time.time() - t0:.0f}s", flush=True)


def note(folder: Path, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "REBUILT.txt").write_text(f"{pd.Timestamp.now():%Y-%m-%d %H:%M} scripts/rebuild_live_stats.py\n{text}\n")


def rain_standardizer(files: list[Path]) -> None:
    seqs = []
    for f in files:
        with xr.open_dataset(f) as ds:
            seqs.append(build_nbhd_sequences(ds, sequence_length=LEADS).load())
    sample = xr.concat(seqs, dim="run")
    s = ChannelStandardizer.fit(sample, list(sample["run"].values))
    np.savez(RAIN_CACHE / "standardizer.npz", mean=s.mean, std=s.std)
    note(RAIN_CACHE, f"standardizer.npz re-fitted on {len(files)} operational GEFS runs (approximate).")
    print("rain standardizer: " + " ".join(f"{n}={m:.3g}/{v:.3g}" for n, m, v in zip(NBHD_CHANNEL_NAMES, s.mean, s.std)))


def lps_stats(files: list[Path]) -> None:
    truth = np.load(LPS_TRUTH)
    params = json.loads(str(truth["params"]))
    feats = [process_init((0, str(f), truth["mask"], GEFS_ZETA_BY_LEAD, params["deficit_min"], "ops_rain"))[1]
             for f in files]
    x = np.stack(feats).reshape(-1, len(LPS_CHANNELS))
    mu, sd = x.mean(0), x.std(0); sd[sd < 1e-6] = 1.0
    np.savez(LPS_CACHE / "standardizer.npz", mean=mu, std=sd)
    print("lps standardizer: " + " ".join(f"{n}={m:.3g}/{v:.3g}" for n, m, v in zip(LPS_CHANNELS, mu, sd)))

    clim = climatology(LPS_TRUTH, TRAIN_LAST)
    np.save(LPS_CACHE / "clim_strike300.npy", clim)
    # env standardizer exactly as build_lps_env: every 5th daily training init, all cells and leads
    inits = pd.date_range("2000-01-01", f"{TRAIN_LAST}-12-31", freq="D")[::5]
    prev = load_rmm().reindex(inits - pd.Timedelta(days=1)).fillna(0.0)
    s1 = np.zeros(len(ENV)); s2 = np.zeros(len(ENV)); n = 0
    cells = clim.shape[1] * clim.shape[2]
    for i, t in enumerate(inits):
        doy = (t + pd.to_timedelta(np.arange(1, LEADS + 1), unit="D")).dayofyear.values
        ang = 2 * np.pi * doy / 365.25
        r = prev.iloc[i]
        cols = [np.full(LEADS, r["rmm1"]), np.full(LEADS, r["rmm2"]), np.full(LEADS, r["amp"]), np.sin(ang), np.cos(ang)]
        for j, c in enumerate(cols):
            s1[j] += c.sum() * cells; s2[j] += (c ** 2).sum() * cells
        cl = clim[doy - 1]
        s1[5] += cl.sum(); s2[5] += (cl.astype(np.float64) ** 2).sum()
        n += LEADS * cells
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)); sd[sd < 1e-6] = 1.0
    np.savez(LPS_CACHE / "env_standardizer.npz", mean=mu.astype(np.float32), std=sd.astype(np.float32))
    meta = {"channels": LPS_CHANNELS, "env_channels": ENV, "rich_cache": str(RAIN_CACHE), "feature_ch": 24,
            "store": "rain", "truth": str(LPS_TRUTH.relative_to(PROJECT_ROOT)),
            "tracker": {**params, "gefs_zeta_by_lead": GEFS_ZETA_BY_LEAD}, "rebuilt": True}
    (LPS_CACHE / "meta.json").write_text(json.dumps(meta, indent=1))
    note(LPS_CACHE, f"standardizer.npz re-fitted on {len(files)} operational GEFS runs (approximate); "
                    "clim_strike300.npy and env_standardizer.npz recomputed exactly; meta.json minimal.")
    print("env standardizer: " + " ".join(f"{n}={m:.3g}/{v:.3g}" for n, m, v in zip(ENV, mu, sd)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    dates = sample_dates()
    if not args.no_download:
        download(dates)
    files = [STORE / f"{d}00.nc" for d in dates if (STORE / f"{d}00.nc").exists()]
    print(f"fitting on {len(files)} runs", flush=True)
    RAIN_CACHE.mkdir(parents=True, exist_ok=True); LPS_CACHE.mkdir(parents=True, exist_ok=True)
    rain_standardizer(files)
    lps_stats(files)


if __name__ == "__main__":
    main()
