"""Decode one NCMRWF/TIGGE control+perturbed pressure forecast pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


EXPECTED_LEAD_HOURS = np.arange(24, 241, 24)
EXPECTED_LEVELS_HPA = np.array([500, 850])
VARIABLE_NAMES = {
    "u": "u_wind_ms",
    "v": "v_wind_ms",
    "q": "specific_humidity_kgkg",
    "gh": "geopotential_height_gpm",
}


def _open_pressure(path: Path, forecast_type: str) -> xr.Dataset:
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    if dataset.attrs.get("GRIB_centreDescription") != "New Delhi":
        raise ValueError(f"Expected NCMRWF/New Delhi data in {path}")
    if set(dataset.data_vars) != set(VARIABLE_NAMES):
        raise ValueError(f"Unexpected pressure variables in {path}: {list(dataset.data_vars)}")
    lead_hours = dataset.step.values.astype("timedelta64[h]").astype(np.int64)
    if not np.array_equal(lead_hours, EXPECTED_LEAD_HOURS):
        raise ValueError(f"Unexpected pressure leads in {path}: {lead_hours.tolist()}")
    levels = np.sort(np.atleast_1d(dataset.isobaricInhPa.values).astype(int))
    if not np.array_equal(levels, EXPECTED_LEVELS_HPA):
        raise ValueError(f"Unexpected pressure levels in {path}: {levels.tolist()}")

    for variable in dataset.data_vars.values():
        if variable.attrs.get("GRIB_dataType") != forecast_type:
            raise ValueError(f"Wrong forecast type in {path}")
    return dataset


def _canonical_members(dataset: xr.Dataset, forecast_type: str) -> xr.Dataset:
    if forecast_type == "cf":
        if "number" in dataset.dims:
            raise ValueError("Control forecast unexpectedly has a number dimension")
        if "number" in dataset.coords:
            dataset = dataset.reset_coords("number", drop=True)
        return dataset.expand_dims(member=["cf"])
    if "number" not in dataset.dims or dataset.sizes["number"] != 11:
        raise ValueError("Perturbed forecast must contain eleven members")
    names = [f"pf{int(number):02d}" for number in dataset.number.values]
    return dataset.rename({"number": "member"}).assign_coords(member=names)


def prepare_pressure_pair(
    control_file: Path,
    perturbed_file: Path,
    output_file: Path,
) -> Path:
    """Create a canonical Day 1--10 pressure-level ensemble NetCDF."""
    for path in (control_file, perturbed_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    control = _open_pressure(control_file, "cf")
    perturbed = _open_pressure(perturbed_file, "pf")
    control_time = np.asarray(control.time.values).astype("datetime64[ns]")[()]
    perturbed_time = np.asarray(perturbed.time.values).astype("datetime64[ns]")[()]
    if control_time != perturbed_time:
        raise ValueError("Control and perturbed initialization times do not match")

    parts = []
    for forecast_type, dataset in (("cf", control), ("pf", perturbed)):
        dataset = _canonical_members(dataset, forecast_type)
        for coordinate in ("time",):
            if coordinate in dataset.coords and coordinate not in dataset.dims:
                dataset = dataset.reset_coords(coordinate, drop=True)
        parts.append(dataset)

    output = xr.concat(parts, dim="member", join="exact")
    output = output.rename(
        {
            **VARIABLE_NAMES,
            "step": "lead_hours",
            "isobaricInhPa": "pressure_hpa",
        }
    ).assign_coords(
        lead_hours=EXPECTED_LEAD_HOURS.astype("int16"),
        lead_day=("lead_hours", np.arange(1, 11, dtype="int8")),
        valid_time=(
            "lead_hours",
            control_time + EXPECTED_LEAD_HOURS.astype("timedelta64[h]"),
        ),
    )
    output = output.expand_dims(init_time=[control_time])
    output.attrs = {
        "provider": "NCMRWF via ECMWF TIGGE",
        "product": "control and perturbed pressure-level ensemble",
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


if __name__ == "__main__":
    arguments = parse_args()
    result = prepare_pressure_pair(
        arguments.control_file,
        arguments.perturbed_file,
        arguments.output_file,
    )
    print(f"Saved: {result}")
