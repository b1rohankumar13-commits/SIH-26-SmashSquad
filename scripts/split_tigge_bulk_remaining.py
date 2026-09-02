"""Split validated multi-date TIGGE downloads into canonical daily raw GRIB files."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import sys

from eccodes import (
    codes_get,
    codes_grib_new_from_file,
    codes_release,
    codes_write,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.download_tigge_30_day_pilot import output_path
from src.acquisition.download_tigge_bulk_remaining import request_plan


def split_bulk_file(item: dict[str, object]) -> tuple[int, int]:
    source = item["target"]
    assert isinstance(source, Path)
    if not source.is_file():
        raise FileNotFoundError(source)

    product = str(item["product"])
    forecast_type = str(item["forecast_type"])
    chunk = str(item["chunk"])
    expected_dates = {
        datetime.strptime(f"{chunk}{day}", "%Y%m%d").date()
        for day in item["days"]
    }
    targets = {
        initialization: output_path(initialization, forecast_type, product)
        for initialization in expected_dates
    }
    pending = {
        initialization: target
        for initialization, target in targets.items()
        if not target.is_file()
    }
    if not pending:
        print(f"All canonical targets already exist for {source}")
        return 0, len(targets)

    partials = {
        initialization: target.with_suffix(target.suffix + ".part")
        for initialization, target in pending.items()
    }
    for target in pending.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    for partial in partials.values():
        if partial.exists():
            raise FileExistsError(f"Incomplete split target needs inspection: {partial}")

    message_counts = {initialization: 0 for initialization in expected_dates}
    with source.open("rb") as input_stream, ExitStack() as stack:
        output_streams = {
            initialization: stack.enter_context(partial.open("wb"))
            for initialization, partial in partials.items()
        }
        while True:
            gid = codes_grib_new_from_file(input_stream)
            if gid is None:
                break
            try:
                initialization = datetime.strptime(
                    str(codes_get(gid, "dataDate")), "%Y%m%d"
                ).date()
                if initialization not in expected_dates:
                    raise ValueError(
                        f"Unexpected initialization {initialization} in {source}"
                    )
                if int(codes_get(gid, "dataTime")) != 0:
                    raise ValueError(f"Non-00-UTC message in {source}")
                message_counts[initialization] += 1
                if initialization in output_streams:
                    codes_write(gid, output_streams[initialization])
            finally:
                codes_release(gid)

    empty_dates = [date for date, count in message_counts.items() if count == 0]
    if empty_dates:
        raise ValueError(f"Dates without GRIB messages in {source}: {empty_dates}")
    for initialization, partial in partials.items():
        if partial.stat().st_size == 0:
            raise ValueError(f"Empty split target: {partial}")
        partial.replace(pending[initialization])
        print(
            f"Split {initialization}: {message_counts[initialization]} messages -> "
            f"{pending[initialization]}"
        )
    return len(pending), len(targets) - len(pending)


def main() -> None:
    created = 0
    preserved = 0
    for item in request_plan():
        item_created, item_preserved = split_bulk_file(item)
        created += item_created
        preserved += item_preserved
    print(f"Created canonical files: {created}")
    print(f"Preserved canonical files: {preserved}")


if __name__ == "__main__":
    main()
