"""Download one already-prepared TIGGE job using resumable byte ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import cdsapi
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_LOG = PROJECT_ROOT / "logs" / "acquisition.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_log(record: dict[str, object]) -> None:
    ACQUISITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACQUISITION_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _ranges(size: int, workers: int) -> list[tuple[int, int, int]]:
    chunk_size = (size + workers - 1) // workers
    return [
        (index, start, min(size - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, size, chunk_size))
    ]


def download_prepared_job(job_id: str, target: Path, workers: int) -> Path:
    """Download, assemble and checksum one completed ECDS retrieval job."""
    target = target.resolve()
    if not target.is_relative_to(PROJECT_ROOT):
        raise ValueError("Target must stay inside the SIH project")
    if target.is_file():
        print(f"Preserved existing final file: {target}")
        return target

    started_at = datetime.now(timezone.utc).isoformat()
    remote = cdsapi.Client(quiet=True).client.get_remote(job_id)
    results = remote.get_results()
    size = int(results.content_length)
    location = results.location
    base_headers = dict(results.headers)
    ranges = _ranges(size, workers)
    staging_root = (
        Path(tempfile.gettempdir()) / "sih_tigge_ranges" / job_id
    )
    staging_root.mkdir(parents=True, exist_ok=True)

    def fetch(item: tuple[int, int, int]) -> tuple[int, int]:
        index, start, end = item
        expected = end - start + 1
        part = staging_root / f"range_{index:03d}_{start}_{end}.part"
        if part.is_file() and part.stat().st_size == expected:
            return index, expected
        if part.exists() and part.stat().st_size > expected:
            part.unlink()
        for attempt in range(1, 11):
            downloaded = part.stat().st_size if part.exists() else 0
            if downloaded == expected:
                return index, expected
            headers = {
                **base_headers,
                "Range": f"bytes={start + downloaded}-{end}",
            }
            try:
                with requests.get(
                    location,
                    headers=headers,
                    stream=True,
                    timeout=(60, 180),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Range {index} returned HTTP {response.status_code}, "
                            "expected 206"
                        )
                    with part.open("ab") as stream:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                stream.write(block)
                actual = part.stat().st_size
                if actual == expected:
                    return index, actual
                if actual > expected:
                    raise ValueError(
                        f"Range {index} exceeded {expected} bytes ({actual})"
                    )
            except (requests.RequestException, RuntimeError) as error:
                if attempt == 10:
                    raise RuntimeError(
                        f"Range {index} failed after {attempt} attempts"
                    ) from error
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Range {index} did not complete")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item for item in ranges}
        for future in as_completed(futures):
            index, byte_count = future.result()
            completed += byte_count
            print(
                f"Completed range {index + 1}/{len(ranges)}; "
                f"{completed}/{size} bytes staged",
                flush=True,
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    assembled = target.with_suffix(target.suffix + ".range.part")
    with assembled.open("wb") as output:
        for index, start, end in ranges:
            part = staging_root / f"range_{index:03d}_{start}_{end}.part"
            with part.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output, length=1024 * 1024)
    if assembled.stat().st_size != size:
        raise ValueError(f"Assembled size does not match prepared result: {assembled}")
    checksum = _sha256(assembled)
    assembled.replace(target)
    _append_log(
        {
            "source_id": target.stem,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "stored",
            "bytes": size,
            "sha256": checksum,
            "local_path": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "validation_status": "checksum_only_pending_grib_validation",
            "prepared_job_id": job_id,
            "transfer_method": "parallel_http_ranges",
            "error": None,
        }
    )
    shutil.rmtree(staging_root)
    print(f"Stored prepared TIGGE file: {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("target", type=Path)
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 33))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download_prepared_job(arguments.job_id, arguments.target, arguments.workers)
