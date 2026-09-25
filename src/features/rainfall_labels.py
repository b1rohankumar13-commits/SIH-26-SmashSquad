"""Per-cell rainfall-bust labels: heavy-rain miss or false-alarm, forecast vs IMD obs.

A cell/lead is a bust if the ensemble missed observed heavy rain (prob < miss_below) or
false-alarmed (prob >= false_alarm_at with no heavy rain). Cells without a valid
observation are NaN and masked from the loss. Lead day d is matched to IMD rainfall on
``init + (d-1)`` days (a pilot approximation of the 03-UTC alignment in grid.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.detection.heavy_rainfall import ensemble_exceedance_probability, event_error_masks
from src.models.graphnet.grid_graph import canonical_centres

HEAVY_THRESHOLD_MM = 64.5
MISS_BELOW = 0.20
FALSE_ALARM_AT = 0.50


def regrid_obs_to_grid(imd: xr.Dataset, *, step: float = 0.5) -> xr.DataArray:
    rain = imd["RAINFALL"].rename({"TIME": "time", "LATITUDE": "latitude", "LONGITUDE": "longitude"})
    rain = rain.sortby("latitude").sortby("longitude")
    lats, lons = canonical_centres(step)
    return rain.interp(latitude=lats, longitude=lons, method="linear")


def _obs_for_init(obs_grid: xr.DataArray, init_str: str, leads: np.ndarray) -> xr.DataArray:
    init = pd.Timestamp(f"{init_str[:4]}-{init_str[4:6]}-{init_str[6:8]}")
    valids = [init + pd.Timedelta(days=int(d) - 1) for d in leads]
    sub = obs_grid.reindex(time=valids)
    return sub.rename({"time": "lead"}).assign_coords(lead=list(leads))


def build_bust_labels(
    forecast: xr.Dataset, obs_grid: xr.DataArray,
    *, threshold: float = HEAVY_THRESHOLD_MM,
    miss_below: float = MISS_BELOW, false_alarm_at: float = FALSE_ALARM_AT,
) -> np.ndarray:
    init_str = str(forecast["run"].values[0])
    precip = forecast["total_precipitation"].isel(run=0)
    leads = forecast["lead"].values
    observed = _obs_for_init(obs_grid, init_str, leads)

    labels = []
    for lead in leads:
        probability = ensemble_exceedance_probability(precip.sel(lead=lead).values, threshold)
        obs = observed.sel(lead=lead).values
        miss, false_alarm, _ = event_error_masks(
            probability, obs, threshold, miss_below, false_alarm_at)
        valid = np.isfinite(probability) & np.isfinite(obs)
        labels.append(np.where(valid, (miss | false_alarm).astype(np.float32), np.nan))
    return np.stack(labels).astype(np.float32)
