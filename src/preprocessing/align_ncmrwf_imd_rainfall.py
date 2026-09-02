"""Align the first NCMRWF ensemble forecast with matching IMD rainfall."""

from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "decoded"
    / "tigge"
    / "ncmrwf"
    / "2024"
    / "07"
    / "01"
    / "tigge_ncmrwf_20240701_00_daily_rainfall_india.nc"
)
OBSERVATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "decoded"
    / "imd"
    / "rainfall"
    / "2024"
    / "imd_rainfall_20240702_20240711.nc"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aligned"
    / "rainfall"
    / "2024"
    / "07"
    / "01"
    / "ncmrwf_imd_20240701_00_imd_window_day01_day09.nc"
)

IMD_WINDOW_END_HOUR_UTC = 3
HOURS_PER_DAY = 24


def _date_values(values):
    return np.asarray(values).astype("datetime64[D]")


def _cell_bounds(centres):
    """Infer cell edges from strictly increasing one-dimensional centres."""
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("Grid centres must be a strictly increasing 1-D array.")
    bounds = np.empty(centres.size + 1, dtype=np.float64)
    bounds[1:-1] = (centres[:-1] + centres[1:]) / 2.0
    bounds[0] = centres[0] - (centres[1] - centres[0]) / 2.0
    bounds[-1] = centres[-1] + (centres[-1] - centres[-2]) / 2.0
    return bounds


def _overlap_weights(source_centres, target_centres, latitude=False):
    """Return spherical cell-overlap factors for a rectilinear grid axis."""
    source_bounds = _cell_bounds(source_centres)
    target_bounds = _cell_bounds(target_centres)
    weights = np.zeros(
        (len(target_centres), len(source_centres)), dtype=np.float64
    )

    for target_index in range(len(target_centres)):
        lower = np.maximum(source_bounds[:-1], target_bounds[target_index])
        upper = np.minimum(source_bounds[1:], target_bounds[target_index + 1])
        valid = upper > lower
        if latitude:
            weights[target_index, valid] = np.abs(
                np.sin(np.deg2rad(upper[valid]))
                - np.sin(np.deg2rad(lower[valid]))
            )
        else:
            weights[target_index, valid] = np.deg2rad(
                upper[valid] - lower[valid]
            )

    if np.any(weights.sum(axis=1) == 0):
        raise ValueError("At least one target-grid cell does not overlap the source grid.")
    return weights


def _conservative_regrid(forecast, target_latitude, target_longitude):
    """First-order area-conservative remapping of rainfall depth."""
    forecast = forecast.transpose("number", "time", "latitude", "longitude")
    source_values = np.asarray(forecast.values, dtype=np.float64)
    valid = np.isfinite(source_values)
    filled = np.where(valid, source_values, 0.0)

    latitude_weights = _overlap_weights(
        forecast.latitude.values, target_latitude, latitude=True
    )
    longitude_weights = _overlap_weights(
        forecast.longitude.values, target_longitude, latitude=False
    )

    numerator = np.einsum(
        "ai,ntij,bj->ntab",
        latitude_weights,
        filled,
        longitude_weights,
        optimize=True,
    )
    denominator = np.einsum(
        "ai,ntij,bj->ntab",
        latitude_weights,
        valid.astype(np.float64),
        longitude_weights,
        optimize=True,
    )
    remapped = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    ).astype("float32")

    return xr.DataArray(
        remapped,
        dims=("number", "time", "latitude", "longitude"),
        coords={
            "number": forecast.number,
            "time": forecast.time,
            "latitude": target_latitude,
            "longitude": target_longitude,
        },
        name="forecast_member_rainfall_mm",
        attrs={
            "long_name": "IMD-window-adjusted forecast rainfall",
            "units": "mm",
            "regridding": "first-order spherical area-conservative remapping",
        },
    )


def align_rainfall() -> Path:
    """Match IMD 03 UTC windows and conservatively remap forecast rainfall."""
    if not FORECAST_FILE.is_file():
        raise FileNotFoundError(f"Forecast file not found: {FORECAST_FILE}")
    if not OBSERVATION_FILE.is_file():
        raise FileNotFoundError(f"Observation file not found: {OBSERVATION_FILE}")

    forecast_dataset = xr.open_dataset(FORECAST_FILE, engine="netcdf4")
    observation_dataset = xr.open_dataset(OBSERVATION_FILE, engine="netcdf4")

    forecast = forecast_dataset["forecast_rainfall_mm"]
    observation = observation_dataset["observed_rainfall_mm"]

    forecast_dates = _date_values(forecast_dataset["valid_time"].values)
    observation_dates = _date_values(observation_dataset["time"].values)
    if not np.array_equal(forecast_dates, observation_dates):
        raise ValueError(
            "Forecast and observation dates do not match: "
            f"forecast={forecast_dates.tolist()}, observation={observation_dates.tolist()}"
        )

    # The decoded GRIB contains a scalar ``time`` coordinate for the forecast
    # initialization as well as a ``step`` dimension for valid leads. Preserve
    # the initialization timestamp under an unambiguous name before promoting
    # valid lead dates to the time dimension.
    if "time" in forecast.coords and "time" not in forecast.dims:
        forecast = forecast.rename({"time": "forecast_reference_time"})
    forecast = forecast.rename({"step": "time"}).assign_coords(time=forecast_dates)
    observation = observation.assign_coords(time=observation_dates)

    # IMD's value labelled date D covers the preceding 24 hours ending at
    # 03 UTC on D. The available pilot forecast contains only 00-to-00 UTC
    # daily totals. Approximate 03-to-03 UTC by taking 21 hours (7/8) from
    # forecast day D and 3 hours (1/8) from forecast day D+1, assuming rain
    # is uniform within each daily forecast total. This yields nine matched
    # days; an exact tenth day would require an additional forecast lead.
    earlier = forecast.isel(time=slice(0, -1)).copy()
    later_values = forecast.isel(time=slice(1, None)).values
    forecast = (
        (1.0 - IMD_WINDOW_END_HOUR_UTC / HOURS_PER_DAY) * earlier
        + (IMD_WINDOW_END_HOUR_UTC / HOURS_PER_DAY) * later_values
    ).astype("float32")
    forecast = forecast.assign_coords(time=forecast_dates[:-1])
    observation = observation.sel(time=forecast.time)

    if forecast.sizes["time"] != 9 or observation.sizes["time"] != 9:
        raise ValueError(
            "Expected nine IMD-window-aligned pilot days; "
            f"forecast={forecast.sizes['time']}, observation={observation.sizes['time']}"
        )

    # Sort coordinates so slicing and interpolation are deterministic.
    forecast = forecast.sortby("latitude").sortby("longitude")
    observation = observation.sortby("latitude").sortby("longitude")

    latitude_min = max(float(forecast.latitude.min()), float(observation.latitude.min()))
    latitude_max = min(float(forecast.latitude.max()), float(observation.latitude.max()))
    longitude_min = max(float(forecast.longitude.min()), float(observation.longitude.min()))
    longitude_max = min(float(forecast.longitude.max()), float(observation.longitude.max()))

    if latitude_min >= latitude_max or longitude_min >= longitude_max:
        raise ValueError("Forecast and observation grids do not overlap.")

    observation = observation.sel(
        latitude=slice(latitude_min, latitude_max),
        longitude=slice(longitude_min, longitude_max),
    )

    forecast_on_imd_grid = _conservative_regrid(
        forecast,
        observation.latitude.values,
        observation.longitude.values,
    )

    ensemble_mean = forecast_on_imd_grid.mean("number", skipna=True).rename(
        "forecast_ensemble_mean_mm"
    )
    ensemble_spread = forecast_on_imd_grid.std("number", skipna=True).rename(
        "forecast_ensemble_spread_mm"
    )
    error = (ensemble_mean - observation).where(np.isfinite(observation)).rename(
        "forecast_error_mm"
    )
    absolute_error = np.abs(error).rename("absolute_error_mm")

    bias = float(error.mean(skipna=True).compute())
    mae = float(absolute_error.mean(skipna=True).compute())
    rmse = float(np.sqrt((error ** 2).mean(skipna=True).compute()))

    aligned = xr.Dataset(
        {
            "forecast_member_rainfall_mm": forecast_on_imd_grid,
            "forecast_ensemble_mean_mm": ensemble_mean,
            "forecast_ensemble_spread_mm": ensemble_spread,
            "observed_rainfall_mm": observation.astype("float32"),
            "forecast_error_mm": error.astype("float32"),
            "absolute_error_mm": absolute_error.astype("float32"),
        }
    )
    aligned.attrs = {
        "forecast_source": "TIGGE NCMRWF perturbed forecast",
        "observation_source": "IMD 0.25-degree daily gridded rainfall",
        "forecast_initialization": "2024-07-01T00:00:00Z",
        "imd_accumulation_window": "24 hours ending 03:00 UTC on the labelled date",
        "temporal_adjustment": "7/8 current forecast day plus 1/8 next forecast day",
        "temporal_adjustment_status": "pilot approximation from 24-hour forecast totals",
        "production_warning": "Use sub-daily forecasts for exact 03-to-03 UTC accumulation",
        "regridding": "first-order spherical area-conservative forecast-to-IMD remapping",
        "baseline_bias_mm": bias,
        "baseline_mae_mm": mae,
        "baseline_rmse_mm": rmse,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    compression = {
        name: {"dtype": "float32", "zlib": True, "complevel": 4}
        for name in aligned.data_vars
    }
    aligned.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4",
        encoding=compression,
    )

    print(f"Aligned IMD dates: {forecast.time.values[0]} to {forecast.time.values[-1]}")
    print(f"Aligned dimensions: {dict(aligned.sizes)}")
    print(f"Common latitude range: {latitude_min:.2f} to {latitude_max:.2f}")
    print(f"Common longitude range: {longitude_min:.2f} to {longitude_max:.2f}")
    print(f"Baseline bias: {bias:.4f} mm")
    print(f"Baseline MAE: {mae:.4f} mm")
    print(f"Baseline RMSE: {rmse:.4f} mm")
    print("Temporal window: IMD 03-to-03 UTC; daily-forecast 7/8 + 1/8 approximation")
    print("Regridding: first-order spherical area-conservative")
    print(f"Saved: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    align_rainfall()
