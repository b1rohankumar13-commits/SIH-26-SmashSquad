"""Acquire the approved 15 July--13 August 2019 NCMRWF/TIGGE pilot.

The default mode prints every API request without submitting it.  ``test``
downloads only the first date's control/surface file.  ``full`` downloads all
30 daily 00 UTC initializations, split by forecast type and level type so every
raw file has a simple, auditable acquisition contract.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta, timezone, datetime
import hashlib
import json
from pathlib import Path

import cdsapi


DATASET = "tigge-forecasts"
ORIGIN = "ncmrwf"
START_DATE = date(2019, 7, 15)
END_DATE = date(2019, 8, 13)
CYCLE = "00:00"
AREA = (38, 65, 5, 100)  # ECDS order: north, west, south, east.
DAILY_LEADS = tuple(str(hour) for hour in range(24, 241, 24))
ACCUMULATION_LEADS = ("0",) + DAILY_LEADS
FORECAST_TYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}
PRODUCTS = {
    "surface": {
        "level_type": "single_level",
        "variable": ["total_precipitation", "mean_sea_level_pressure"],
        "leadtime_hour": list(ACCUMULATION_LEADS),
        "filename_fields": "tp-mslp",
    },
    "pressure": {
        "level_type": "pressure",
        "level_value": ["500_hpa", "850_hpa"],
        "variable": [
            "geopotential_height",
            "specific_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "leadtime_hour": list(DAILY_LEADS),
        "filename_fields": "uvqgh-p500-p850",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "forecasts" / "tigge" / "ncmrwf"
ACQUISITION_LOG = PROJECT_ROOT / "data" / "metadata" / "acquisition_log.jsonl"


def pilot_dates():
    current = START_DATE
    while current <= END_DATE:
        yield current
        current += timedelta(days=1)


def build_request(
    initialization: date,
    forecast_type: str,
    product: str,
) -> dict[str, object]:
    if forecast_type not in FORECAST_TYPES:
        raise ValueError(f"Unknown forecast type: {forecast_type}")
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product: {product}")
    product_contract = PRODUCTS[product]
    request = {
        "origin": ORIGIN,
        "year": f"{initialization.year:04d}",
        "month": f"{initialization.month:02d}",
        "day": f"{initialization.day:02d}",
        "time": CYCLE,
        "level_type": product_contract["level_type"],
        "variable": list(product_contract["variable"]),
        "forecast_type": FORECAST_TYPES[forecast_type],
        "leadtime_hour": list(product_contract["leadtime_hour"]),
        "data_format": "grib",
        "area": list(AREA),
    }
    if "level_value" in product_contract:
        request["level_value"] = list(product_contract["level_value"])
    return request


def output_path(initialization: date, forecast_type: str, product: str) -> Path:
    token = initialization.strftime("%Y%m%d")
    fields = PRODUCTS[product]["filename_fields"]
    return (
        RAW_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / "00"
        / f"tigge_ncmrwf_{token}_00_{product}_{forecast_type}_{fields}.grib2"
    )


def request_plan(mode: str):
    dates = (START_DATE,) if mode == "test" else tuple(pilot_dates())
    forecast_types = ("cf",) if mode == "test" else tuple(FORECAST_TYPES)
    products = ("surface",) if mode == "test" else tuple(PRODUCTS)
    for initialization in dates:
        for forecast_type in forecast_types:
            for product in products:
                yield {
                    "date": initialization,
                    "forecast_type": forecast_type,
                    "product": product,
                    "request": build_request(initialization, forecast_type, product),
                    "target": output_path(initialization, forecast_type, product),
                }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_log(record: dict[str, object]) -> None:
    ACQUISITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACQUISITION_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def download(plan) -> None:
    client = cdsapi.Client()
    for item in plan:
        target = item["target"]
        if target.exists():
            print(f"Preserved existing raw file: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(
                f"Incomplete target already exists and needs inspection: {temporary}"
            )
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            client.retrieve(DATASET, item["request"], str(temporary))
            byte_size = temporary.stat().st_size
            if byte_size == 0:
                raise ValueError("TIGGE returned an empty file")
            checksum = sha256(temporary)
            temporary.replace(target)
            append_log(
                {
                    "source_id": target.stem,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "stored",
                    "bytes": byte_size,
                    "sha256": checksum,
                    "local_path": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "validation_status": "checksum_only_pending_grib_validation",
                    "error": None,
                }
            )
            print(f"Stored: {target}")
        except Exception as error:
            append_log(
                {
                    "source_id": target.stem,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "bytes": temporary.stat().st_size if temporary.exists() else 0,
                    "sha256": None,
                    "local_path": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "validation_status": "failed",
                    "error": str(error),
                }
            )
            raise


def print_plan(plan) -> None:
    plan = list(plan)
    print(f"Planned requests: {len(plan)}")
    for item in plan:
        print(f"\nTarget: {item['target']}")
        print(json.dumps(item["request"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "test", "full"),
        default="plan",
        help="plan is dry-run; test downloads one control/surface request; full downloads all requests",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    effective_mode = "full" if arguments.mode == "plan" else arguments.mode
    plan = tuple(request_plan(effective_mode))
    if arguments.mode == "plan":
        print_plan(plan)
        print("\nDry run only; no API request was submitted.")
    else:
        download(plan)


if __name__ == "__main__":
    main()
