"""Validate IMD 0.25-degree rainfall and extract requested verification dates."""

import argparse
import calendar
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "observations"
    / "imd"
    / "rainfall"
    / "2024"
    / "RF25_ind2024_rfp25.nc"
)
def _find_name(names, candidates):
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _configured_domain() -> dict[str, float]:
    config_path = PROJECT_ROOT / "configs" / "grid.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    return {
        name: float(configuration["domain"][name])
        for name in ("north", "south", "west", "east")
    }


def _default_output_file(start_date: str, end_date: str) -> Path:
    year = start_date[:4]
    start_token = start_date.replace("-", "")
    end_token = end_date.replace("-", "")
    return (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "decoded"
        / "imd"
        / "rainfall"
        / year
        / f"imd_rainfall_{start_token}_{end_token}.nc"
    )


def prepare_imd_rainfall(
    start_date: str,
    end_date: str,
    *,
    raw_file: Path = RAW_FILE,
    output_file: Path | None = None,
) -> Path:
    """Validate the yearly IMD file and save the explicitly requested dates."""
    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw IMD file not found: {raw_file}")
    if start_date[:4] != end_date[:4]:
        raise ValueError("One conversion output must remain within a single year")
    if output_file is None:
        output_file = _default_output_file(start_date, end_date)

    dataset = xr.open_dataset(raw_file, engine="netcdf4")
    coordinate_names = list(dataset.coords) + list(dataset.dims)

    time_name = _find_name(coordinate_names, ["time"])
    latitude_name = _find_name(coordinate_names, ["latitude", "lat"])
    longitude_name = _find_name(coordinate_names, ["longitude", "lon"])
    rainfall_name = _find_name(
        dataset.data_vars,
        ["rainfall", "rf", "precipitation", "precip"],
    )

    missing = [
        label
        for label, value in {
            "time": time_name,
            "latitude": latitude_name,
            "longitude": longitude_name,
            "rainfall variable": rainfall_name,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Could not identify {missing}; dimensions={dict(dataset.sizes)}, "
            f"variables={list(dataset.variables)}"
        )

    rename = {
        time_name: "time",
        latitude_name: "latitude",
        longitude_name: "longitude",
        rainfall_name: "observed_rainfall_mm",
    }
    dataset = dataset.rename({old: new for old, new in rename.items() if old != new})

    source_year = int(start_date[:4])
    expected_year_days = 366 if calendar.isleap(source_year) else 365
    if dataset.sizes["time"] != expected_year_days:
        raise ValueError(
            f"Expected {expected_year_days} days for {source_year}; "
            f"found {dataset.sizes['time']}"
        )

    rainfall = dataset["observed_rainfall_mm"].astype("float32")
    rainfall.attrs = {
        **rainfall.attrs,
        "long_name": "IMD daily gridded rainfall",
        "units": "mm",
    }
    domain = _configured_domain()
    selected = rainfall.sel(
        time=slice(start_date, end_date),
        latitude=slice(domain["south"], domain["north"]),
        longitude=slice(domain["west"], domain["east"]),
    )
    expected_dates = (
        np.datetime64(end_date, "D") - np.datetime64(start_date, "D")
    ).astype(int) + 1
    if selected.sizes["time"] != expected_dates:
        dates = [str(value) for value in selected.time.values]
        raise ValueError(
            f"Expected {expected_dates} matching dates; found {len(dates)}: {dates}"
        )

    finite = np.isfinite(selected)
    finite_count = int(finite.sum().compute())
    total_count = int(selected.size)
    if finite_count == 0:
        raise ValueError("All selected IMD rainfall values are missing.")

    minimum = float(selected.min(skipna=True).compute())
    maximum = float(selected.max(skipna=True).compute())
    if minimum < 0:
        raise ValueError(f"Negative IMD rainfall detected: minimum={minimum:.4f} mm")

    output = selected.rename({"time": "valid_time"}).to_dataset()
    output.attrs = {
        "provider": "India Meteorological Department",
        "product": "0.25-degree daily gridded rainfall",
        "source_file": raw_file.name,
        "selected_period": f"{start_date}/{end_date}",
        "domain": ", ".join(f"{name}={value}" for name, value in domain.items()),
        "processing_note": (
            "Date and configured-domain extraction only; original 0.25-degree "
            "grid retained; no regridding performed"
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(
        output_file,
        engine="netcdf4",
        encoding={
            "observed_rainfall_mm": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
            }
        },
    )

    missing_fraction = 1.0 - finite_count / total_count
    print(f"Yearly dimensions: {dict(dataset.sizes)}")
    print(f"Selected dimensions: {dict(output.sizes)}")
    print(
        f"Selected dates: {output.valid_time.values[0]} "
        f"to {output.valid_time.values[-1]}"
    )
    print(f"Rainfall range: {minimum:.4f} to {maximum:.4f} mm")
    print(f"Missing fraction: {missing_fraction:.4%}")
    print(f"Saved: {output_file}")
    return output_file


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="First YYYY-MM-DD date")
    parser.add_argument("--end-date", required=True, help="Last YYYY-MM-DD date")
    parser.add_argument("--raw-file", type=Path, default=RAW_FILE)
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    prepare_imd_rainfall(
        arguments.start_date,
        arguments.end_date,
        raw_file=arguments.raw_file,
        output_file=arguments.output_file,
    )
