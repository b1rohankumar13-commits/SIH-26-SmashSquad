"""Prepare the first TIGGE/NCMRWF rainfall case for forecast verification.

The raw GRIB is preserved unchanged. This script extracts the India-domain
buffer and differences consecutive accumulated-precipitation fields to obtain
Day 1 through Day 10 rainfall totals.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from src.preprocessing.daily_rainfall import cumulative_to_daily


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "forecasts"
    / "tigge"
    / "ncmrwf"
    / "2024"
    / "07"
    / "01"
    / "tigge_ncmrwf_20240701_00_tp_ensemble_global.grib2"
)
OUTPUT_FILE = (
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

EXPECTED_LEAD_HOURS = np.arange(0, 241, 24)


def _configured_domain() -> dict[str, float]:
    """Read the binding model domain from ``configs/grid.yaml``."""
    config_path = PROJECT_ROOT / "configs" / "grid.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    return {
        name: float(configuration["domain"][name])
        for name in ("north", "south", "west", "east")
    }


def prepare_rainfall() -> Path:
    """Create a compact India-domain Day 1–10 rainfall dataset."""
    if not RAW_FILE.is_file():
        raise FileNotFoundError(f"Raw TIGGE file not found: {RAW_FILE}")

    dataset = xr.open_dataset(
        RAW_FILE,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    if "tp" not in dataset:
        raise ValueError(f"Expected total precipitation 'tp'; found {list(dataset.data_vars)}")
    if "number" not in dataset.dims or "step" not in dataset.dims:
        raise ValueError(f"Expected ensemble and lead dimensions; found {dict(dataset.sizes)}")

    centre = dataset.attrs.get("GRIB_centreDescription")
    if centre != "New Delhi":
        raise ValueError(f"Expected NCMRWF/New Delhi data; found centre={centre!r}")

    actual_lead_hours = (
        dataset.step.values.astype("timedelta64[h]").astype(np.int64)
    )
    if not np.array_equal(actual_lead_hours, EXPECTED_LEAD_HOURS):
        raise ValueError(
            f"Expected leads {EXPECTED_LEAD_HOURS.tolist()}; "
            f"found {actual_lead_hours.tolist()}"
        )

    units = dataset["tp"].attrs.get("units")
    if units not in {"kg m**-2", "kg m-2", "mm"}:
        raise ValueError(
            "Expected precipitation depth in kg m**-2 (numerically equal to mm) "
            f"or mm; found units={units!r}"
        )

    domain = _configured_domain()
    # Latitude is north-to-south in this GRIB; longitude is west-to-east.
    india = dataset.sel(
        latitude=slice(domain["north"], domain["south"]),
        longitude=slice(domain["west"], domain["east"]),
    )
    if india.sizes.get("latitude", 0) == 0 or india.sizes.get("longitude", 0) == 0:
        raise ValueError("India subsetting produced an empty grid.")

    accumulated = india["tp"].astype("float32")
    daily = cumulative_to_daily(accumulated).rename("forecast_rainfall_mm")
    daily.attrs = {
        "long_name": "24-hour accumulated forecast rainfall",
        "units": "mm",
        "derivation": "Difference of consecutive TIGGE total-precipitation accumulations",
    }
    for scalar_coordinate in ("time", "surface"):
        if scalar_coordinate in daily.coords and scalar_coordinate not in daily.dims:
            daily = daily.reset_coords(scalar_coordinate, drop=True)

    minimum_daily_value = float(daily.min().compute())
    if minimum_daily_value < -0.01:
        raise ValueError(
            "Negative daily rainfall differences detected "
            f"(minimum={minimum_daily_value:.4f} mm); check accumulation semantics."
        )

    daily = daily.rename({"number": "member", "step": "lead_hours"})
    output = daily.to_dataset().assign_coords(
        lead_hours=actual_lead_hours[1:].astype("int16"),
        lead_day=("lead_hours", np.arange(1, 11, dtype="int8")),
        member=india.number.values,
    )
    if "valid_time" in india.coords:
        output = output.assign_coords(
            valid_time=(
                "lead_hours",
                india.valid_time.isel(step=slice(1, None)).values,
            )
        )
    output = output.expand_dims(
        init_time=[np.asarray(india.time.values).astype("datetime64[ns]")[()]]
    )
    init_time = np.datetime_as_string(
        np.asarray(india.time.values).astype("datetime64[s]"), unit="s"
    )
    output.attrs = {
        "provider": "NCMRWF via ECMWF TIGGE",
        "product": "TIGGE perturbed ensemble forecast",
        "forecast_type": "perturbed",
        "source_file": RAW_FILE.name,
        "forecast_initialization": f"{init_time}Z",
        "domain": ", ".join(f"{name}={value}" for name, value in domain.items()),
        "raw_precipitation_units": units,
        "processing_note": (
            "Raw GRIB preserved; cumulative rainfall converted to Day 1-10 "
            "increments; configured India domain extracted; no regridding or "
            "land mask applied"
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4",
        encoding={
            "forecast_rainfall_mm": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
            }
        },
    )

    print(f"Raw dimensions: {dict(dataset.sizes)}")
    print(f"India dimensions: {dict(output.sizes)}")
    print(f"Ensemble members: {output.sizes['member']}")
    print(f"Daily forecast leads: {output.sizes['lead_hours']}")
    print(f"Minimum daily rainfall: {minimum_daily_value:.4f} mm")
    print(f"Saved: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    prepare_rainfall()
