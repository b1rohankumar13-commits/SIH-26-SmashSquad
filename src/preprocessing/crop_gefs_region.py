"""Crop native 0.5-degree GEFS GRIB messages to the approved regional 1-degree grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from eccodes import (
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
    codes_set,
    codes_set_array,
    codes_write,
)


_INDEX_CACHE: dict[tuple, np.ndarray] = {}


def _target_coordinates(north: float, south: float, west: float, east: float, resolution: float):
    """Return cell centres for edge bounds, north-to-south and west-to-east."""
    latitudes = np.arange(north - resolution / 2, south, -resolution, dtype=np.float64)
    longitudes = np.arange(west + resolution / 2, east, resolution, dtype=np.float64)
    return latitudes, longitudes


def _extract_values(handle, target_latitudes: np.ndarray, target_longitudes: np.ndarray) -> np.ndarray:
    cache_key = (
        int(codes_get(handle, "Ni")),
        int(codes_get(handle, "Nj")),
        float(codes_get(handle, "latitudeOfFirstGridPointInDegrees")),
        float(codes_get(handle, "longitudeOfFirstGridPointInDegrees")),
        float(codes_get(handle, "iDirectionIncrementInDegrees")),
        float(codes_get(handle, "jDirectionIncrementInDegrees")),
        int(codes_get(handle, "iScansNegatively")),
        int(codes_get(handle, "jScansPositively")),
    )
    source_values = np.asarray(codes_get_array(handle, "values"), dtype=np.float64)
    indices = _INDEX_CACHE.get(cache_key)
    if indices is None:
        source_latitudes = np.asarray(codes_get_array(handle, "latitudes"), dtype=np.float64)
        source_longitudes = np.mod(np.asarray(codes_get_array(handle, "longitudes"), dtype=np.float64), 360.0)
        coordinate_index = {
            (round(float(latitude), 6), round(float(longitude), 6)): position
            for position, (latitude, longitude) in enumerate(zip(source_latitudes, source_longitudes))
        }
        requested = [
            (round(float(latitude), 6), round(float(longitude), 6))
            for latitude in target_latitudes
            for longitude in target_longitudes
        ]
        missing = [coordinate for coordinate in requested if coordinate not in coordinate_index]
        if missing:
            raise ValueError(f"Native grid is missing {len(missing)} approved regional coordinates; first={missing[0]}")
        indices = np.asarray([coordinate_index[coordinate] for coordinate in requested], dtype=np.int64)
        _INDEX_CACHE[cache_key] = indices
    return source_values[indices]


def _set_regional_grid(handle, latitudes: np.ndarray, longitudes: np.ndarray, values: np.ndarray) -> None:
    codes_set(handle, "gridType", "regular_ll")
    codes_set(handle, "Ni", int(longitudes.size))
    codes_set(handle, "Nj", int(latitudes.size))
    codes_set(handle, "latitudeOfFirstGridPointInDegrees", float(latitudes[0]))
    codes_set(handle, "latitudeOfLastGridPointInDegrees", float(latitudes[-1]))
    codes_set(handle, "longitudeOfFirstGridPointInDegrees", float(longitudes[0]))
    codes_set(handle, "longitudeOfLastGridPointInDegrees", float(longitudes[-1]))
    codes_set(handle, "iDirectionIncrementInDegrees", float(longitudes[1] - longitudes[0]))
    codes_set(handle, "jDirectionIncrementInDegrees", float(latitudes[0] - latitudes[1]))
    codes_set(handle, "iScansNegatively", 0)
    codes_set(handle, "jScansPositively", 0)
    codes_set(handle, "jPointsAreConsecutive", 0)
    codes_set_array(handle, "values", values)


def crop_grib_file(
    source: Path,
    destination: Path,
    *,
    north: float = 38.0,
    south: float = 5.0,
    west: float = 65.0,
    east: float = 100.0,
    resolution: float = 1.0,
) -> dict:
    target_latitudes, target_longitudes = _target_coordinates(north, south, west, east, resolution)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    message_count = 0
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        while True:
            handle = codes_grib_new_from_file(input_handle)
            if handle is None:
                break
            try:
                values = _extract_values(handle, target_latitudes, target_longitudes)
                _set_regional_grid(handle, target_latitudes, target_longitudes, values)
                codes_write(handle, output_handle)
                message_count += 1
            finally:
                codes_release(handle)
    if message_count == 0:
        raise ValueError(f"No GRIB messages found in {source}")
    verified = verify_regional_grib(
        temporary,
        expected_messages=message_count,
        expected_ni=int(target_longitudes.size),
        expected_nj=int(target_latitudes.size),
        expected_resolution=resolution,
    )
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "source": str(source),
        "destination": str(destination),
        "messages": message_count,
        "ni": int(target_longitudes.size),
        "nj": int(target_latitudes.size),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "verified": verified,
    }


def verify_regional_grib(
    path: Path,
    *,
    expected_messages: int,
    expected_ni: int,
    expected_nj: int,
    expected_resolution: float,
) -> bool:
    messages = 0
    with path.open("rb") as input_handle:
        while True:
            handle = codes_grib_new_from_file(input_handle)
            if handle is None:
                break
            try:
                assert int(codes_get(handle, "Ni")) == expected_ni
                assert int(codes_get(handle, "Nj")) == expected_nj
                assert np.isclose(float(codes_get(handle, "iDirectionIncrementInDegrees")), expected_resolution)
                assert np.isclose(float(codes_get(handle, "jDirectionIncrementInDegrees")), expected_resolution)
                assert len(codes_get_array(handle, "values")) == expected_ni * expected_nj
                messages += 1
            finally:
                codes_release(handle)
    if messages != expected_messages:
        raise ValueError(f"Expected {expected_messages} messages in {path}; decoded {messages}")
    return True


def crop_grib_tree(source_root: Path, destination_root: Path) -> dict:
    files = sorted(source_root.rglob("*.grib2"))
    completed = 0
    messages = 0
    source_bytes = 0
    destination_bytes = 0
    for source in files:
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        result = crop_grib_file(source, destination)
        completed += 1
        messages += int(result["messages"])
        source_bytes += source.stat().st_size
        destination_bytes += int(result["bytes"])
    return {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "files": completed,
        "messages": messages,
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "compression_ratio": source_bytes / destination_bytes if destination_bytes else None,
        "verified": completed == len(files),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = _parse_args(sys.argv[1:])
    result = (
        crop_grib_tree(arguments.source, arguments.destination)
        if arguments.source.is_dir()
        else crop_grib_file(arguments.source, arguments.destination)
    )
    print(json.dumps(result, indent=2))
