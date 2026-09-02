"""Align the approved 30-day NCMRWF rainfall pilot with IMD observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from src.acquisition.download_tigge_30_day_pilot import pilot_dates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_ROOT = PROJECT_ROOT / "data" / "interim" / "decoded" / "tigge" / "ncmrwf"
OBSERVATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "decoded"
    / "imd"
    / "rainfall"
    / "2019"
    / "imd_rainfall_20190716_20190823.nc"
)
OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "aligned" / "rainfall"
GRID_CONFIG = PROJECT_ROOT / "configs" / "grid.yaml"


def _cell_bounds(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("Grid centres must be a strictly increasing 1-D array")
    bounds = np.empty(centres.size + 1, dtype=np.float64)
    bounds[1:-1] = (centres[:-1] + centres[1:]) / 2.0
    bounds[0] = centres[0] - np.diff(centres[:2])[0] / 2.0
    bounds[-1] = centres[-1] + np.diff(centres[-2:])[0] / 2.0
    return bounds


def _overlap_weights(
    source_centres: np.ndarray,
    target_centres: np.ndarray,
    *,
    latitude: bool,
) -> np.ndarray:
    source_bounds = _cell_bounds(source_centres)
    target_bounds = _cell_bounds(target_centres)
    weights = np.zeros((target_centres.size, source_centres.size), dtype=np.float64)
    for target_index in range(target_centres.size):
        lower = np.maximum(source_bounds[:-1], target_bounds[target_index])
        upper = np.minimum(source_bounds[1:], target_bounds[target_index + 1])
        valid = upper > lower
        if latitude:
            weights[target_index, valid] = np.abs(
                np.sin(np.deg2rad(upper[valid]))
                - np.sin(np.deg2rad(lower[valid]))
            )
        else:
            weights[target_index, valid] = np.deg2rad(upper[valid] - lower[valid])
    return weights


def conservative_regrid(
    field: xr.DataArray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> xr.DataArray:
    """Area-average a rainfall-depth field onto the configured rectilinear grid."""
    leading_dims = tuple(dim for dim in field.dims if dim not in {"latitude", "longitude"})
    field = field.transpose(*leading_dims, "latitude", "longitude")
    source = np.asarray(field.values, dtype=np.float64)
    valid = np.isfinite(source)
    latitude_weights = _overlap_weights(
        field.latitude.values, target_latitude, latitude=True
    )
    longitude_weights = _overlap_weights(
        field.longitude.values, target_longitude, latitude=False
    )
    numerator = np.einsum(
        "ai,...ij,bj->...ab",
        latitude_weights,
        np.where(valid, source, 0.0),
        longitude_weights,
        optimize=True,
    )
    denominator = np.einsum(
        "ai,...ij,bj->...ab",
        latitude_weights,
        valid.astype(np.float64),
        longitude_weights,
        optimize=True,
    )
    values = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    ).astype("float32")
    coordinates = {dim: field.coords[dim] for dim in leading_dims}
    coordinates.update(latitude=target_latitude, longitude=target_longitude)
    return xr.DataArray(
        values,
        dims=(*leading_dims, "latitude", "longitude"),
        coords=coordinates,
        name=field.name,
        attrs=field.attrs,
    )


def _target_grid() -> tuple[np.ndarray, np.ndarray]:
    with GRID_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["domain"].get("bounds_role") != "cell_edges":
        raise ValueError("The approved grid must explicitly use domain bounds as cell edges")
    spacing = float(config["resolution_degrees"])
    centres = config["cell_centres"]
    latitude = np.arange(
        float(centres["latitude_start"]),
        float(centres["latitude_end"]) + spacing / 2,
        spacing,
    )
    longitude = np.arange(
        float(centres["longitude_start"]),
        float(centres["longitude_end"]) + spacing / 2,
        spacing,
    )
    return latitude, longitude


def _forecast_path(initialization) -> Path:
    token = initialization.strftime("%Y%m%d")
    return (
        FORECAST_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / f"tigge_ncmrwf_{token}_00_surface_ensemble_day01-day10.nc"
    )


def _output_path(initialization) -> Path:
    token = initialization.strftime("%Y%m%d")
    return (
        OUTPUT_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / f"ncmrwf_imd_{token}_00_common0p5_day01-day09.nc"
    )


def align_initialization(
    initialization,
    observation: xr.DataArray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    *,
    overwrite: bool,
) -> Path:
    forecast_path = _forecast_path(initialization)
    output_path = _output_path(initialization)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    dataset = xr.open_dataset(forecast_path, engine="netcdf4")
    forecast = dataset["forecast_rainfall_mm"].squeeze("init_time", drop=True)
    forecast = forecast.sortby("latitude").sortby("longitude")
    if forecast.sizes["lead_hours"] != 10:
        raise ValueError(f"Expected ten daily forecast fields in {forecast_path}")

    # Explicitly approved pilot approximation: IMD date D is the 24-hour
    # window ending 03 UTC, approximated from daily 00-to-00 UTC forecast totals.
    adjusted = (
        0.875 * forecast.isel(lead_hours=slice(0, -1))
        + 0.125 * forecast.isel(lead_hours=slice(1, None)).values
    ).swap_dims({"lead_hours": "lead_day"}).drop_vars("lead_hours")
    adjusted = adjusted.assign_coords(lead_day=np.arange(1, 10, dtype="int8"))
    valid_time = dataset.valid_time.isel(lead_hours=slice(0, -1)).values
    adjusted = adjusted.assign_coords(valid_time=("lead_day", valid_time))

    observed = observation.sel(valid_time=valid_time)
    if observed.sizes["valid_time"] != 9:
        raise ValueError(f"Expected nine matching IMD dates for {initialization}")
    observed = observed.rename({"valid_time": "lead_day"}).assign_coords(
        lead_day=np.arange(1, 10, dtype="int8"),
        valid_time=("lead_day", valid_time),
    )
    observed = observed.sortby("latitude").sortby("longitude")

    forecast_common = conservative_regrid(
        adjusted, target_latitude, target_longitude
    ).rename("forecast_member_rainfall_mm")
    observed_common = conservative_regrid(
        observed, target_latitude, target_longitude
    ).rename("observed_rainfall_mm")
    ensemble_mean = forecast_common.mean("member", skipna=True).rename(
        "forecast_ensemble_mean_mm"
    )
    ensemble_spread = forecast_common.std("member", skipna=True).rename(
        "forecast_ensemble_spread_mm"
    )
    output = xr.Dataset(
        {
            "forecast_member_rainfall_mm": forecast_common,
            "forecast_ensemble_mean_mm": ensemble_mean,
            "forecast_ensemble_spread_mm": ensemble_spread,
            "observed_rainfall_mm": observed_common,
        }
    ).assign_coords(
        valid_time=("lead_day", valid_time)
    ).expand_dims(init_time=[np.datetime64(initialization.isoformat(), "ns")])
    output.attrs = {
        "forecast_source": "NCMRWF via ECMWF TIGGE",
        "observation_source": "IMD 0.25-degree daily gridded rainfall",
        "temporal_alignment": "7/8 current forecast day plus 1/8 next day",
        "temporal_alignment_status": "explicitly approved pilot approximation",
        "retained_lead_days": 9,
        "target_grid": "0.5-degree cell centres; configured domain bounds are edges",
        "regridding": "first-order spherical area-conservative rainfall remapping",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding={
            name: {"dtype": "float32", "zlib": True, "complevel": 4}
            for name in output.data_vars
        },
    )
    return output_path


def align_all(*, overwrite: bool = False) -> list[Path]:
    forecast_paths = [_forecast_path(initialization) for initialization in pilot_dates()]
    missing = [str(path) for path in forecast_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The decoded 30-day surface pilot is incomplete:\n" + "\n".join(missing)
        )
    if not OBSERVATION_FILE.is_file():
        raise FileNotFoundError(OBSERVATION_FILE)
    observation = xr.open_dataset(OBSERVATION_FILE, engine="netcdf4")[
        "observed_rainfall_mm"
    ]
    target_latitude, target_longitude = _target_grid()
    outputs = []
    for initialization in pilot_dates():
        result = align_initialization(
            initialization,
            observation,
            target_latitude,
            target_longitude,
            overwrite=overwrite,
        )
        outputs.append(result)
        print(f"Aligned {initialization}: {result}")
    print(f"Aligned initializations: {len(outputs)}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-derived", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    align_all(overwrite=arguments.overwrite_derived)
