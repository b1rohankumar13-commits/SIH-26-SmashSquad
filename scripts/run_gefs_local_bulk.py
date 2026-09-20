"""Run the approved 2021--2023 GEFS acquisition in resumable daily batches."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.download_gefs_reforecast import download_gefs_reforecast


def _dates(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += dt.timedelta(days=1)


def _completed_dates(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    completed = set()
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("status") == "complete":
                completed.add(entry["date"])
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acquisition_gefs_2021_2023.yaml")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument(
        "--date-workers",
        type=int,
        default=1,
        help="Number of initialization dates to acquire concurrently.",
    )
    arguments = parser.parse_args(argv)
    config_path = PROJECT_ROOT / arguments.config
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = PROJECT_ROOT / config["output"]["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_dates(checkpoint)
    pending = [
        date
        for date in _dates(dt.date.fromisoformat(arguments.start), dt.date.fromisoformat(arguments.end))
        if date not in completed
    ]
    checkpoint_lock = threading.Lock()

    def acquire_date(date: str) -> tuple[str, dict]:
        for attempt in range(1, 4):
            try:
                summary = download_gefs_reforecast(str(config_path), [date], plan_only=False)
                return date, summary
            except Exception as error:
                print(json.dumps({"event": "date_failed", "date": date, "attempt": attempt, "error": repr(error)}), flush=True)
                if attempt == 3:
                    raise
                time.sleep(30 * attempt)
        raise RuntimeError(f"Unreachable retry state for {date}")

    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.date_workers) as pool:
        futures = {pool.submit(acquire_date, date): date for date in pending}
        for future in concurrent.futures.as_completed(futures):
            date = futures[future]
            try:
                _, summary = future.result()
            except Exception as error:
                failed = True
                print(json.dumps({"event": "date_abandoned", "date": date, "error": repr(error)}), flush=True)
                continue
            with checkpoint_lock:
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"date": date, "status": "complete", "summary": summary}) + "\n")
            print(json.dumps({"event": "date_complete", "date": date, **summary}), flush=True)
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
