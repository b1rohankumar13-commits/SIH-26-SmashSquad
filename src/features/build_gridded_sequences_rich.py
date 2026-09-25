"""Richer 29-variable channel set for the multivar rainfall-bust experiment.

The 6-base builder in build_gridded_sequences.py is left untouched (it feeds the
20-year rain model). This module reads the 29-variable multivar store, emitting
mean+spread for every variable plus the same exceedance/physics derived channels,
so we can test whether the extra variables beat the 6-variable baseline at equal
years. Reuses the base module's helpers to keep the mean/spread/derived logic in
one place.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.features.build_gridded_sequences import (
    CANONICAL_DIMS, _derived_channels, _mean_and_spread, _select_base,
)

# Variables as named in the multivar per-init NetCDFs (each pressure level is its
# own variable, so no level selection is needed).
MV_VARS: tuple[str, ...] = (
    "total_precipitation", "pres_msl", "pres_sfc", "tmp_2m", "spfh_2m", "cape",
    "ugrd_10m", "vgrd_10m", "olr",
    "tmp_850", "tmp_700", "tmp_500", "tmp_300",
    "ugrd_850", "ugrd_700", "ugrd_500", "ugrd_300",
    "vgrd_850", "vgrd_700", "vgrd_500", "vgrd_300",
    "spfh_850", "spfh_700", "spfh_500", "spfh_300",
    "hgt_850", "hgt_700", "hgt_500", "hgt_300",
)

# Map the multivar names onto the keys _derived_channels expects for the physics.
_PHYS_KEYS: dict[str, str] = {
    "ugrd_850": "u850", "vgrd_850": "v850", "spfh_850": "q850", "hgt_500": "gh500",
}

_DERIVED_NAMES: tuple[str, ...] = (
    "rain_exc_heavy", "rain_exc_very_heavy", "rain_exc_extreme",
    "moisture_flux_850", "convergence_850", "vorticity_850", "gh500_grad",
)

RICH_CHANNEL_NAMES: tuple[str, ...] = tuple(
    [name for var in MV_VARS for name in (f"{var}_mean", f"{var}_spread")]
) + _DERIVED_NAMES

RICH_N_CHANNELS: int = len(RICH_CHANNEL_NAMES)


def build_rich_sequences(
    dataset: xr.Dataset, *, member_dim: str = "member", sequence_length: int | None = None,
) -> xr.DataArray:
    channels: list[xr.DataArray] = []
    means: dict[str, xr.DataArray] = {}
    rainfall_field: xr.DataArray | None = None
    for var in MV_VARS:
        field = _select_base(dataset, var, None, member_dim)
        mean, spread = _mean_and_spread(field, member_dim)
        channels.append(mean.rename(f"{var}_mean"))
        channels.append(spread.rename(f"{var}_spread"))
        if var == "total_precipitation":
            rainfall_field = field
        if var in _PHYS_KEYS:
            means[_PHYS_KEYS[var]] = mean.transpose(*CANONICAL_DIMS)

    if rainfall_field is None:
        raise ValueError("multivar dataset is missing total_precipitation")
    channels.extend(_derived_channels(rainfall_field, means, member_dim))

    reference = channels[0]
    for channel in channels[1:]:
        if channel.sizes != reference.sizes:
            raise ValueError("Base fields do not share the same run/lead/grid shape")

    stacked = xr.concat(channels, dim="channel").assign_coords(channel=list(RICH_CHANNEL_NAMES))
    stacked = stacked.transpose(*CANONICAL_DIMS, "channel")
    if sequence_length is not None and stacked.sizes["lead"] != int(sequence_length):
        raise ValueError(
            f"sequence_length={sequence_length} does not match lead count {stacked.sizes['lead']}")
    stacked.attrs["channels"] = list(RICH_CHANNEL_NAMES)
    return stacked


def standardize_rich(sequences: xr.DataArray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize with a fitted mean/std (channel-count-agnostic, unlike the base transform)."""
    array = sequences.transpose(*CANONICAL_DIMS, "channel").values.astype(np.float64)
    if array.shape[-1] != mean.size:
        raise ValueError("Channel count does not match the fitted standardizer")
    out = np.nan_to_num((array - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)
