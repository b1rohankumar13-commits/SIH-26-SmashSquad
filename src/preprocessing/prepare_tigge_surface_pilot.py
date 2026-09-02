"""Decode one NCMRWF/TIGGE control+perturbed surface forecast pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from src.preprocessing.daily_rainfall import cumulative_to_daily


EXPECTED_LEAD_HOURS = np.arange(0, 241, 24)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _open_field(path: Path, short_name: str) -> xr.DataArray:
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": short_name},
        },
    )
    if short_name not in dataset:
        raise ValueError(f"{path} does not contain {short_name!r}")
    if dataset.attrs.get("GRIB_centreDescription") != "New Delhi":
        raise ValueError(f"Expected NCMRWF/New Delhi data in {path}")
    lead_hours = dataset.step.values.astype("timedelta64[h]").astype(np.int64)
    if not np.array_equal(lead_hours, EXPECTED_LEAD_HOURS):
        raise ValueError(
            f"Expected leads {EXPECTED_LEAD_HOURS.tolist()} in {path}; "
            f"found {lead_hours.tolist()}"
        )
    return dataset[short_name]


def _canonical_members(field: xr.DataArray, forecast_type: str) -> xr.DataArray:
    if forecast_type == "cf":
        if "number" in field.dims:
            raise ValueError("Control forecast unexpectedly contains a number dimension")
        return field.expand_dims(member=["cf"])
    if forecast_type != "pf" or "number" not in field.dims:
        raise ValueError("Perturbed forecast must contain a number dimension")
    member_names = [f"pf{int(number):02d}" for number in field.number.values]
    return field.rename({"number": "member"}).assign_coords(member=member_names)


def _drop_scalar_grib_coordinates(field: xr.DataArray) -> xr.DataArray:
    for coordinate in tuple(field.coords):
        if coordinate not in field.dims and coordinate not in {"valid_time"}:
            field = field.reset_coords(coordinate, drop=True)
    return field


def prepare_surface_pair(
    control_file: Path,
    perturbed_file: Path,
    output_file: Path,
) -> Path:
    """Create a canonical Day 1--10 rainfall/MSLP ensemble NetCDF."""
    for path in (control_file, perturbed_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    rainfall_parts = []
    pressure_parts = []
    initialization = None
    for forecast_type, path in (("cf", control_file), ("pf", perturbed_file)):
        accumulated = _open_field(path, "tp")
        pressure = _open_field(path, "msl")
        current_initialization = np.asarray(accumulated.time.values).astype(
            "datetime64[ns]"
        )[()]
        if initialization is None:
            initialization = current_initialization
        elif current_initialization != initialization:
            raise ValueError("Control and perturbed initialization times do not match")

        rainfall = cumulative_to_daily(accumulated).rename("forecast_rainfall_mm")
        if float(rainfall.min().compute()) < -0.01:
            raise ValueError(f"Negative daily rainfall increments detected in {path}")
        rainfall = rainfall.clip(min=0.0)
        pressure = pressure.isel(step=slice(1, None)).rename(
            "mean_sea_level_pressure_pa"
        )
        rainfall_parts.append(
            _canonical_members(_drop_scalar_grib_coordinates(rainfall), forecast_type)
        )
        pressure_parts.append(
            _canonical_members(_drop_scalar_grib_coordinates(pressure), forecast_type)
        )

    rainfall = xr.concat(rainfall_parts, dim="member", join="exact")
    pressure = xr.concat(pressure_parts, dim="member", join="exact")
    rainfall = rainfall.rename({"step": "lead_hours"}).assign_coords(
        lead_hours=EXPECTED_LEAD_HOURS[1:].astype("int16")
    )
    pressure = pressure.rename({"step": "lead_hours"}).assign_coords(
        lead_hours=EXPECTED_LEAD_HOURS[1:].astype("int16")
    )
    valid_time = initialization + EXPECTED_LEAD_HOURS[1:].astype("timedelta64[h]")

    output = xr.Dataset(
        {
            "forecast_rainfall_mm": rainfall.astype("float32"),
            "mean_sea_level_pressure_pa": pressure.astype("float32"),
        }
    ).assign_coords(
        lead_day=("lead_hours", np.arange(1, 11, dtype="int8")),
        valid_time=("lead_hours", valid_time),
    )
    output = output.expand_dims(init_time=[initialization])
    output["forecast_rainfall_mm"].attrs.update(
        units="mm",
        derivation=(
            "difference of consecutive cumulative TIGGE precipitation fields; "
            "tolerated GRIB packing remnants below zero clipped to zero"
        ),
    )
    output["mean_sea_level_pressure_pa"].attrs.update(units="Pa")
    output.attrs = {
        "provider": "NCMRWF via ECMWF TIGGE",
        "product": "control and perturbed surface ensemble",
        "control_source_file": control_file.name,
        "perturbed_source_file": perturbed_file.name,
        "processing_note": "Decoded and combined only; native grid retained",
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding={
            name: {"dtype": "float32", "zlib": True, "complevel": 4}
            for name in output.data_vars
        },
    )
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--perturbed-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = prepare_surface_pair(
        arguments.control_file,
        arguments.perturbed_file,
        arguments.output_file,
    )
    print(f"Saved: {result}")


if __name__ == "__main__":
    main()
