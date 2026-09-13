"""Poll the download checkpoint; launch 2000-2004 training once a 2005 date appears."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "data" / "metadata" / "gefs_reforecast_completed_dates.jsonl"
WATCH_LOG = PROJECT_ROOT / "logs" / "auto_train_watcher.log"
TRAIN_LOG = PROJECT_ROOT / "logs" / "graphnet_pilot_2000_2004.out.log"

TRIGGER_YEAR = 2005
POLL_SECONDS = 300
TRAIN_YEARS = "2000-2004"
TRAIN_EPOCHS = "40"


def _log(message: str) -> None:
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()}  {message}\n")


def _completed_years() -> set[int]:
    if not CHECKPOINT.exists():
        return set()
    years: set[int] = set()
    with open(CHECKPOINT, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                years.add(int(json.loads(raw)["init"][:4]))
    return years


def main() -> None:
    _log(f"watcher started; waiting for a {TRIGGER_YEAR} date")
    while TRIGGER_YEAR not in _completed_years():
        _log(f"not ready; years present: {sorted(_completed_years())}")
        time.sleep(POLL_SECONDS)

    _log(f"{TRIGGER_YEAR} date present; launching training on {TRAIN_YEARS}")
    with open(TRAIN_LOG, "w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_graphnet_pilot",
             "--years", TRAIN_YEARS, "--epochs", TRAIN_EPOCHS],
            cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
        )
    _log(f"training finished with exit code {result.returncode}; log -> {TRAIN_LOG}")


if __name__ == "__main__":
    main()
