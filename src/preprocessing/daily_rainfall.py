"""Convert cumulative forecast rainfall into daily increments."""

from __future__ import annotations

import numpy as np
import xarray as xr


def cumulative_to_daily(
    cumulative: xr.DataArray,
    *,
    lead_dimension: str = "step",
) -> xr.DataArray:
    """Difference cumulative rainfall after validating its lead coordinate.

    The input must contain a zero-hour accumulation followed by strictly
    increasing lead times.  Consequently, differencing ``[A0, A1, ..., A10]``
    returns the documented Day 1--10 values ``[A1-A0, ..., A10-A9]``.  Requiring
    the zero-hour field prevents a file that starts at Day 1 from silently
    losing the first forecast day.
    """
    if not isinstance(cumulative, xr.DataArray):
        raise TypeError("cumulative must be an xarray.DataArray")
    if lead_dimension not in cumulative.dims:
        raise ValueError(
            f"Missing lead dimension {lead_dimension!r}; found {cumulative.dims}"
        )
    if cumulative.sizes[lead_dimension] < 2:
        raise ValueError("At least zero-hour and one forecast lead are required")

    coordinate = cumulative[lead_dimension]
    values = np.asarray(coordinate.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        lead_hours = values.astype("timedelta64[h]").astype(np.int64)
    elif np.issubdtype(values.dtype, np.number):
        lead_hours = values.astype(np.float64)
    else:
        raise TypeError(
            f"Lead coordinate must be numeric or timedelta64; found {values.dtype}"
        )

    if not np.isclose(lead_hours[0], 0):
        raise ValueError(
            "Cumulative rainfall must include a zero-hour field before Day 1"
        )
    if np.any(np.diff(lead_hours) <= 0):
        raise ValueError("Forecast lead times must be strictly increasing")

    daily = cumulative.diff(lead_dimension)
    daily.attrs = {
        **cumulative.attrs,
        "long_name": "forecast rainfall accumulated over each lead interval",
        "derivation": "difference of consecutive cumulative rainfall fields",
    }
    return daily
