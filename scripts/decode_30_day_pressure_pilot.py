"""Decode all 30 downloaded NCMRWF/TIGGE pressure ensemble pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.download_tigge_30_day_pilot import pilot_dates
from src.preprocessing.prepare_tigge_pressure_pilot import prepare_pressure_pair


RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "forecasts" / "tigge" / "ncmrwf"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "decoded" / "tigge" / "ncmrwf"
PROCESSING_LOG = PROJECT_ROOT / "data" / "metadata" / "processing_history.jsonl"
CONFIG_FILES = (
    PROJECT_ROOT / "configs" / "grid.yaml",
    PROJECT_ROOT / "configs" / "variables.yaml",
)


def config_hash() -> str:
    digest = hashlib.sha256()
    for path in CONFIG_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def paths(initialization):
    directory = (
        RAW_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / "00"
    )
    token = initialization.strftime("%Y%m%d")
    control = directory / f"tigge_ncmrwf_{token}_00_pressure_cf_uvqgh-p500-p850.grib2"
    perturbed = directory / f"tigge_ncmrwf_{token}_00_pressure_pf_uvqgh-p500-p850.grib2"
    output = (
        OUTPUT_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / f"tigge_ncmrwf_{token}_00_pressure_ensemble_day01-day10.nc"
    )
    return control, perturbed, output


def append_processing_record(control: Path, perturbed: Path, output: Path) -> None:
    record = {
        "record_type": "processing",
        "input_ids": [control.stem, perturbed.stem],
        "operation": "decode_and_combine_pressure_ensemble",
        "config_hash": config_hash(),
        "output_path": str(output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with PROCESSING_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def decode_all(overwrite: bool = False) -> list[Path]:
    plan = [(initialization, *paths(initialization)) for initialization in pilot_dates()]
    missing = [
        str(path)
        for _, control, perturbed, _ in plan
        for path in (control, perturbed)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The 30-day raw pressure acquisition is incomplete:\n" + "\n".join(missing)
        )

    outputs = []
    for initialization, control, perturbed, output in plan:
        if output.exists() and not overwrite:
            raise FileExistsError(f"Derived output already exists: {output}")
        result = prepare_pressure_pair(control, perturbed, output)
        append_processing_record(control, perturbed, result)
        outputs.append(result)
        print(f"Decoded {initialization}: {result}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Replace decoded outputs only; raw files are never changed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    decoded = decode_all(overwrite=arguments.overwrite_derived)
    print(f"Decoded initializations: {len(decoded)}")
