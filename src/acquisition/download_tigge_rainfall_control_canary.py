"""Download the verified NCMRWF rainfall pressure-control canary via ECDS.

The acquisition contract intentionally matches the existing perturbed-member
files for 15 July 2019. Run without ``--execute`` to inspect the two requests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cdsapi


DATASET = "tigge-forecasts"
INITIALIZATION_DATE = "2019-07-15"
CYCLES = ("00:00", "12:00")
LEVELS = ("500_hpa", "850_hpa")
VARIABLES = (
    "geopotential_height",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
)
LEADTIME_HOURS = tuple(str(hour) for hour in range(12, 241, 12))
# ECDS area order: north, west, south, east.
AREA = (38, 65, 5, 100)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "forecasts"
    / "tigge"
    / "ncmrwf"
    / "2019"
    / "07"
    / "15"
)


def build_request(cycle: str) -> dict[str, object]:
    """Build one ECDS request from the approved acquisition contract."""
    if cycle not in CYCLES:
        raise ValueError(f"Unsupported cycle {cycle!r}; expected one of {CYCLES}.")
    return {
        "origin": "ncmrwf",
        "year": "2019",
        "month": "07",
        "day": "15",
        "time": cycle,
        "level_type": "pressure",
        # Current ECDS TIGGE form schema uses the singular key "level_value".
        "level_value": list(LEVELS),
        "variable": list(VARIABLES),
        "forecast_type": "control_forecast",
        "leadtime_hour": list(LEADTIME_HOURS),
        "data_format": "grib",
        "area": list(AREA),
    }


def output_path(cycle: str) -> Path:
    cycle_hour = cycle[:2]
    return (
        RAW_ROOT
        / cycle_hour
        / (
            f"tigge_ncmrwf_20190715_{cycle_hour}_pressure_cf_"
            "f012-f240_12h_uvqgh_p500-p850.grib2"
        )
    )


def selected_cycles(value: str) -> tuple[str, ...]:
    return CYCLES if value == "both" else (f"{value}:00",)


def print_plan(cycles: tuple[str, ...]) -> None:
    for cycle in cycles:
        target = output_path(cycle)
        print(f"\nCycle: {cycle}")
        print(f"Target: {target}")
        print(json.dumps(build_request(cycle), indent=2))


def download(cycles: tuple[str, ...], overwrite: bool) -> None:
    client = cdsapi.Client()
    for cycle in cycles:
        target = output_path(cycle)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Target already exists: {target}. Inspect it first or rerun with --overwrite."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Submitting {INITIALIZATION_DATE} {cycle} control request")
        client.retrieve(DATASET, build_request(cycle), str(target))
        print(f"Downloaded: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycle",
        choices=("00", "12", "both"),
        default="both",
        help="Cycle to plan/download; the approved canary requires both.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit requests. Without this flag the script only prints the plan.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing target file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = selected_cycles(args.cycle)
    print_plan(cycles)
    if args.execute:
        download(cycles, overwrite=args.overwrite)
    else:
        print("\nDry run only. Add --execute after reviewing this plan.")


if __name__ == "__main__":
    main()
