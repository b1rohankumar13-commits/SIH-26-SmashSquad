"""Assemble ensemble forecast fields into the spatial branch's gridded sequences.

Output is channels-last ``[run, lead, latitude, longitude, channel]``; channels are the
mean/spread pairs from ``configs/convlstm.yaml``. Standardisation is a separate step so
the caller fits it on training runs only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

CANONICAL_DIMS = ("run", "lead", "latitude", "longitude")

EXCEEDANCE_THRESHOLDS_MM: tuple[float, ...] = (64.5, 115.6, 204.5)

CHANNEL_NAMES: tuple[str, ...] = (
    "rainfall_mean_mm", "rainfall_spread_mm",
    "mslp_mean_pa", "mslp_spread_pa",
    "u850_mean_ms", "u850_spread_ms",
    "v850_mean_ms", "v850_spread_ms",
    "q850_mean_kgkg", "q850_spread_kgkg",
    "gh500_mean_gpm", "gh500_spread_gpm",
    # Ensemble exceedance probabilities at the bust thresholds (the tail signal the
    # mean/spread channels miss; XGBoost found exc_heavy highly informative).
    "rain_exc_heavy", "rain_exc_very_heavy", "rain_exc_extreme",
    # Physics derived from the ensemble-mean fields.
    "moisture_flux_850", "convergence_850", "vorticity_850", "gh500_grad",
)

N_CHANNELS: int = len(CHANNEL_NAMES)

_BASE_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("rainfall", "rainfall_mean_mm", "rainfall_spread_mm"),
    ("mslp", "mslp_mean_pa", "mslp_spread_pa"),
    ("u850", "u850_mean_ms", "u850_spread_ms"),
    ("v850", "v850_mean_ms", "v850_spread_ms"),
    ("q850", "q850_mean_kgkg", "q850_spread_kgkg"),
    ("gh500", "gh500_mean_gpm", "gh500_spread_gpm"),
)

DEFAULT_VARIABLE_MAP: dict[str, tuple[str, dict | None]] = {
    "rainfall": ("total_precipitation", None),
    "mslp": ("mean_sea_level_pressure", None),
    "u850": ("u_wind", {"level": 850}),
    "v850": ("v_wind", {"level": 850}),
    "q850": ("specific_humidity", {"level": 850}),
    "gh500": ("geopotential_height", {"level": 500}),
}


def _select_base(
    dataset: xr.Dataset, variable: str, selection: dict | None, member_dim: str
) -> xr.DataArray:
    if variable not in dataset.variables:
        raise ValueError(f"Dataset is missing forecast variable {variable!r}")
    field = dataset[variable]
    if selection:
        for coordinate, value in selection.items():
            if coordinate not in field.coords:
                raise ValueError(
                    f"Variable {variable!r} has no coordinate {coordinate!r} to select {value}"
                )
        field = field.sel(selection).drop_vars(list(selection), errors="ignore")
    if member_dim not in field.dims:
        raise ValueError(
            f"Variable {variable!r} must have an ensemble dimension {member_dim!r}"
        )
    missing = [dimension for dimension in CANONICAL_DIMS if dimension not in field.dims]
    if missing:
        raise ValueError(f"Variable {variable!r} is missing required dims {missing}")
    return field


def _mean_and_spread(field: xr.DataArray, member_dim: str) -> tuple[xr.DataArray, xr.DataArray]:
    mean = field.mean(member_dim, skipna=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cell -> NaN spread
        spread = field.std(member_dim, ddof=1, skipna=True)
    return mean.transpose(*CANONICAL_DIMS), spread.transpose(*CANONICAL_DIMS)


def _derived_channels(
    rainfall_field: xr.DataArray, means: dict[str, xr.DataArray], member_dim: str
) -> list[xr.DataArray]:
    """Exceedance-probability and physics channels from the ensemble fields already read."""
    out: list[xr.DataArray] = []

    exc_names = ("rain_exc_heavy", "rain_exc_very_heavy", "rain_exc_extreme")
    for threshold, name in zip(EXCEEDANCE_THRESHOLDS_MM, exc_names):
        prob = (rainfall_field >= threshold).mean(member_dim).transpose(*CANONICAL_DIMS)
        out.append(prob.rename(name))

    wind = np.sqrt(means["u850"] ** 2 + means["v850"] ** 2)
    out.append((means["q850"] * wind).rename("moisture_flux_850"))

    ref = means["u850"]
    u, v, gh = means["u850"].values, means["v850"].values, means["gh500"].values
    du_dlat, du_dlon = np.gradient(u, axis=2), np.gradient(u, axis=3)
    dv_dlat, dv_dlon = np.gradient(v, axis=2), np.gradient(v, axis=3)
    dgh_dlat, dgh_dlon = np.gradient(gh, axis=2), np.gradient(gh, axis=3)
    derived = {
        "convergence_850": -(du_dlon + dv_dlat),   # -(du/dx + dv/dy)
        "vorticity_850": dv_dlon - du_dlat,          # dv/dx - du/dy
        "gh500_grad": np.sqrt(dgh_dlat ** 2 + dgh_dlon ** 2),
    }
    for name, values in derived.items():
        out.append(xr.DataArray(values, dims=ref.dims, coords=ref.coords, name=name))
    return out


def build_gridded_sequences(
    dataset: xr.Dataset,
    *,
    variable_map: dict[str, tuple[str, dict | None]] | None = None,
    member_dim: str = "member",
    sequence_length: int | None = None,
) -> xr.DataArray:
    variable_map = variable_map or DEFAULT_VARIABLE_MAP
    missing_bases = [base for base, *_ in _BASE_CHANNELS if base not in variable_map]
    if missing_bases:
        raise ValueError(f"variable_map is missing base fields {missing_bases}")

    channels: list[xr.DataArray] = []
    means: dict[str, xr.DataArray] = {}
    rainfall_field: xr.DataArray | None = None
    for base, mean_name, spread_name in _BASE_CHANNELS:
        variable, selection = variable_map[base]
        field = _select_base(dataset, variable, selection, member_dim)
        mean, spread = _mean_and_spread(field, member_dim)
        channels.append(mean.rename(mean_name))
        channels.append(spread.rename(spread_name))
        means[base] = mean.transpose(*CANONICAL_DIMS)
        if base == "rainfall":
            rainfall_field = field

    channels.extend(_derived_channels(rainfall_field, means, member_dim))

    reference = channels[0]
    for channel in channels[1:]:
        if channel.sizes != reference.sizes:
            raise ValueError("Base fields do not share the same run/lead/grid shape")

    stacked = xr.concat(channels, dim="channel").assign_coords(channel=list(CHANNEL_NAMES))
    stacked = stacked.transpose(*CANONICAL_DIMS, "channel")
    if sequence_length is not None and stacked.sizes["lead"] != int(sequence_length):
        raise ValueError(
            f"sequence_length={sequence_length} does not match lead count {stacked.sizes['lead']}"
        )
    stacked.attrs["channels"] = list(CHANNEL_NAMES)
    return stacked


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, sequences: xr.DataArray, train_runs) -> "ChannelStandardizer":
        if "channel" not in sequences.dims:
            raise ValueError("sequences must have a channel dimension")
        train = sequences.sel(run=list(train_runs))
        if train.sizes["run"] == 0:
            raise ValueError("train_runs selected no runs")
        reduce_dims = [dimension for dimension in train.dims if dimension != "channel"]
        mean = train.mean(reduce_dims, skipna=True).values.astype(np.float64)
        std = train.std(reduce_dims, skipna=True).values.astype(np.float64)
        if not np.isfinite(mean).all():
            raise ValueError("A channel has no finite training values to standardise")
        std = np.where(std > 0, std, 1.0)
        return cls(mean=mean, std=std)

    def transform(self, sequences: xr.DataArray) -> np.ndarray:
        if list(sequences.coords["channel"].values) != list(CHANNEL_NAMES):
            raise ValueError("sequences channels do not match CHANNEL_NAMES")
        array = sequences.transpose(*CANONICAL_DIMS, "channel").values.astype(np.float64)
        if array.shape[-1] != self.mean.size:
            raise ValueError("Channel count does not match the fitted standardizer")
        standardized = np.nan_to_num((array - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(standardized).all():
            raise ValueError("Standardised sequences are not finite")
        return standardized.astype(np.float32)
