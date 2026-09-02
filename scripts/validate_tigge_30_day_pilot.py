"""Validate completed NCMRWF/TIGGE pilot GRIB files against the API contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cfgrib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.download_tigge_30_day_pilot import (
    ACCUMULATION_LEADS,
    DAILY_LEADS,
    request_plan,
)


EXPECTED_VARIABLES = {
    "surface": {"tp", "msl"},
    "pressure": {"gh", "q", "u", "v"},
}


def validate_grib(path: Path, initialization, forecast_type: str, product: str) -> None:
    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    if not datasets:
        raise ValueError(f"No GRIB datasets decoded from {path}")

    variables = set()
    levels = set()
    expected_leads = np.asarray(
        ACCUMULATION_LEADS if product == "surface" else DAILY_LEADS,
        dtype=np.int64,
    )
    expected_time = np.datetime64(initialization.isoformat(), "ns")

    for dataset in datasets:
        if dataset.attrs.get("GRIB_centreDescription") != "New Delhi":
            raise ValueError(f"Wrong forecast centre in {path}")
        if np.asarray(dataset.time.values).astype("datetime64[ns]")[()] != expected_time:
            raise ValueError(f"Wrong initialization in {path}")
        actual_leads = dataset.step.values.astype("timedelta64[h]").astype(np.int64)
        if not np.array_equal(actual_leads, expected_leads):
            raise ValueError(f"Wrong lead sequence in {path}: {actual_leads.tolist()}")
        if not (4.9 <= float(dataset.latitude.min()) <= 5.2):
            raise ValueError(f"Wrong southern boundary in {path}")
        if not (37.8 <= float(dataset.latitude.max()) <= 38.1):
            raise ValueError(f"Wrong northern boundary in {path}")
        if not (65.0 <= float(dataset.longitude.min()) <= 65.3):
            raise ValueError(f"Wrong western boundary in {path}")
        if not (99.7 <= float(dataset.longitude.max()) <= 100.1):
            raise ValueError(f"Wrong eastern boundary in {path}")

        for name, variable in dataset.data_vars.items():
            variables.add(name)
            if variable.attrs.get("GRIB_dataType") != forecast_type:
                raise ValueError(f"Wrong forecast type for {name} in {path}")
        if "isobaricInhPa" in dataset.coords:
            levels.update(np.atleast_1d(dataset.isobaricInhPa.values).astype(int).tolist())

        has_members = "number" in dataset.dims
        if forecast_type == "pf" and (
            not has_members or dataset.sizes["number"] != 11
        ):
            raise ValueError(f"Expected 11 perturbed members in {path}")
        if forecast_type == "cf" and has_members:
            raise ValueError(f"Control forecast unexpectedly has members in {path}")

    if variables != EXPECTED_VARIABLES[product]:
        raise ValueError(
            f"Wrong variables in {path}: expected {EXPECTED_VARIABLES[product]}, "
            f"found {variables}"
        )
    if product == "pressure" and levels != {500, 850}:
        raise ValueError(f"Wrong pressure levels in {path}: {levels}")


def validate_pilot(available_only: bool) -> tuple[int, int]:
    plan = tuple(request_plan("full"))
    validated = 0
    missing = 0
    for item in plan:
        path = item["target"]
        if not path.is_file():
            missing += 1
            if available_only:
                continue
            raise FileNotFoundError(path)
        validate_grib(
            path,
            item["date"],
            item["forecast_type"],
            item["product"],
        )
        validated += 1
        print(f"Verified: {path}")
    return validated, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Validate completed files and report missing files without failing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    validated_count, missing_count = validate_pilot(arguments.available_only)
    print(f"Validated files: {validated_count}")
    print(f"Missing files: {missing_count}")
