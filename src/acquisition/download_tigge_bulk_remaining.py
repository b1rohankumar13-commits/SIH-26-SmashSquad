"""Bulk-acquire the missing portion of the approved 30-day TIGGE pilot.

The already-finalized per-initialization GRIB files remain immutable.  This
downloader groups the missing dates into July and August requests and stores
the returned multi-date GRIB files in a separate raw bulk area.  The files can
then be validated and split/decoded without replacing existing raw inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cdsapi

from src.acquisition.download_tigge_30_day_pilot import (
    ACCUMULATION_LEADS,
    ACQUISITION_LOG,
    AREA,
    CYCLE,
    DAILY_LEADS,
    DATASET,
    FORECAST_TYPES,
    PRODUCTS,
    PROJECT_ROOT,
)


BULK_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "forecasts" / "tigge" / "ncmrwf" / "2019" / "_bulk"
)

# July 15--18 are complete.  On July 19 only pressure/pf is missing.
DATE_CHUNKS = {
    "201907": {
        "year": "2019",
        "month": "07",
        "days_by_item": {
            ("surface", "cf"): tuple(f"{day:02d}" for day in range(20, 32)),
            ("surface", "pf"): tuple(f"{day:02d}" for day in range(20, 32)),
            ("pressure", "cf"): tuple(f"{day:02d}" for day in range(20, 32)),
            ("pressure", "pf"): tuple(f"{day:02d}" for day in range(19, 32)),
        },
    },
    "201908": {
        "year": "2019",
        "month": "08",
        "days_by_item": {
            (product, forecast_type): tuple(f"{day:02d}" for day in range(1, 14))
            for product in PRODUCTS
            for forecast_type in FORECAST_TYPES
        },
    },
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


def build_request(
    *, year: str, month: str, days: tuple[str, ...], product: str, forecast_type: str
) -> dict[str, object]:
    contract = PRODUCTS[product]
    request: dict[str, object] = {
        "origin": "ncmrwf",
        "year": [year],
        "month": [month],
        "day": list(days),
        "time": CYCLE,
        "level_type": contract["level_type"],
        "variable": list(contract["variable"]),
        "forecast_type": FORECAST_TYPES[forecast_type],
        "leadtime_hour": list(
            ACCUMULATION_LEADS if product == "surface" else DAILY_LEADS
        ),
        "data_format": "grib",
        "area": list(AREA),
    }
    if "level_value" in contract:
        request["level_value"] = list(contract["level_value"])
    return request


def request_plan() -> tuple[dict[str, object], ...]:
    plan: list[dict[str, object]] = []
    for chunk_name, chunk in DATE_CHUNKS.items():
        for (product, forecast_type), days in chunk["days_by_item"].items():
            fields = PRODUCTS[product]["filename_fields"]
            target = BULK_ROOT / (
                f"tigge_ncmrwf_{chunk_name}_missing_00_"
                f"{product}_{forecast_type}_{fields}.grib2"
            )
            plan.append(
                {
                    "chunk": chunk_name,
                    "product": product,
                    "forecast_type": forecast_type,
                    "days": days,
                    "request": build_request(
                        year=chunk["year"],
                        month=chunk["month"],
                        days=days,
                        product=product,
                        forecast_type=forecast_type,
                    ),
                    "target": target,
                }
            )
    return tuple(plan)


def download_one(item: dict[str, object]) -> str:
    target = item["target"]
    assert isinstance(target, Path)
    if target.is_file():
        return f"Preserved existing bulk raw file: {target}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete bulk target needs inspection: {temporary}")

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        cdsapi.Client().retrieve(DATASET, item["request"], str(temporary))
        byte_size = temporary.stat().st_size
        if byte_size == 0:
            raise ValueError("TIGGE returned an empty bulk file")
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
                "request_dates": list(item["days"]),
                "error": None,
            }
        )
        return f"Stored bulk file: {target}"
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
                "request_dates": list(item["days"]),
                "error": str(error),
            }
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "full"), default="plan")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="Concurrent ECDS requests; four minimizes wall-clock queue time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = request_plan()
    if args.mode == "plan":
        print(f"Bulk requests: {len(plan)}")
        for item in plan:
            print(f"\nTarget: {item['target']}")
            print(json.dumps(item["request"], indent=2))
        print("\nDry run only; no API request was submitted.")
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item): item for item in plan}
        for future in as_completed(futures):
            item = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                message = f"FAILED {item['target']}: {error}"
                failures.append(message)
                print(message, flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    main()
