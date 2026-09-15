"""Engineered per-cell tabular features for the XGBoost branch.

Deliberately different from the GNN's mean/spread channels: ensemble exceedance
probabilities at the rainfall thresholds, member spread/extremes, a wind-speed term,
local spatial neighbourhood + gradient, a temporal trend across leads, and position /
season. These give the tree model complementary, non-spatial-structural signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

THRESHOLDS_MM = (64.5, 115.6, 204.5)

FEATURE_NAMES = [
    "rain_mean", "rain_std", "rain_max", "rain_min",
    "exc_heavy", "exc_very_heavy", "exc_extreme",
    "mslp_mean", "u850_mean", "v850_mean", "wind850", "q850_mean", "gh500_mean",
    "rain_nbhd3", "rain_grad", "rain_trend",
    "lat", "lon", "lead_day", "doy_sin", "doy_cos",
]


def tabular_features(dataset: xr.Dataset) -> np.ndarray:
    """Return engineered features [lead, lat, lon, len(FEATURE_NAMES)] for one init."""
    def field(name):
        da = dataset[name]
        return da.isel(run=0) if "run" in da.dims else da

    tp = field("total_precipitation").values                 # [member, lead, lat, lon]
    rain_mean, rain_std = tp.mean(0), tp.std(0)
    rain_max, rain_min = tp.max(0), tp.min(0)
    exceed = [(tp >= t).mean(0) for t in THRESHOLDS_MM]       # each [lead, lat, lon]

    def ctx(name, level=None):
        da = field(name)
        if level is not None:
            da = da.sel(level=level)
        return da.mean("member").values

    mslp = ctx("mean_sea_level_pressure")
    u850, v850 = ctx("u_wind", 850), ctx("v_wind", 850)
    wind850 = np.sqrt(u850 ** 2 + v850 ** 2)
    q850, gh500 = ctx("specific_humidity", 850), ctx("geopotential_height", 500)

    leads = rain_mean.shape[0]
    nbhd = np.stack([uniform_filter(rain_mean[l], size=3, mode="nearest") for l in range(leads)])
    gy, gx = np.gradient(rain_mean, axis=1), np.gradient(rain_mean, axis=2)
    grad = np.sqrt(gy ** 2 + gx ** 2)
    trend = np.diff(rain_mean, axis=0, prepend=rain_mean[:1])

    L, H, W = rain_mean.shape
    lat = np.broadcast_to(field("total_precipitation").latitude.values[None, :, None], (L, H, W))
    lon = np.broadcast_to(field("total_precipitation").longitude.values[None, None, :], (L, H, W))
    lead_day = np.broadcast_to(np.asarray(dataset["lead"].values)[:, None, None], (L, H, W))

    init = str(dataset["run"].values[0])
    doy = pd.Timestamp(f"{init[:4]}-{init[4:6]}-{init[6:8]}").dayofyear
    doy_sin = np.full((L, H, W), np.sin(2 * np.pi * doy / 365.0))
    doy_cos = np.full((L, H, W), np.cos(2 * np.pi * doy / 365.0))

    stacked = np.stack([
        rain_mean, rain_std, rain_max, rain_min,
        exceed[0], exceed[1], exceed[2],
        mslp, u850, v850, wind850, q850, gh500,
        nbhd, grad, trend,
        lat, lon, lead_day, doy_sin, doy_cos,
    ], axis=-1)
    return stacked.astype(np.float32)
