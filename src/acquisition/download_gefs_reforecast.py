"""Download GEFSv12 reforecast fields, crop to India, regrid, and store per-init NetCDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "acquisition_gefs_reforecast.yaml"
USER_AGENT = "SIH-Forecast-Bust-Research/1.0 (low-rate NOAA open-data acquisition)"

_INSTANT_STEP = re.compile(r"^(\d+) hour fcst$")
_ACCUM_STEP = re.compile(r"^(\d+)-(\d+) hour acc fcst$")
_AVG_STEP = re.compile(r"^(\d+)-(\d+) hour ave fcst$")
_MAX_STEP = re.compile(r"^(\d+)-(\d+) hour max fcst$")
_MIN_STEP = re.compile(r"^(\d+)-(\d+) hour min fcst$")


@dataclass(frozen=True)
class IndexRecord:
    message: int
    offset: int
    variable: str
    level: str
    step: str
    length: int | None = None

    @property
    def instant_hour(self) -> int | None:
        match = _INSTANT_STEP.match(self.step)
        return int(match.group(1)) if match else None

    @property
    def accumulation(self) -> tuple[int, int] | None:
        match = _ACCUM_STEP.match(self.step)
        return (int(match.group(1)), int(match.group(2))) if match else None

    @property
    def average(self) -> tuple[int, int] | None:
        match = _AVG_STEP.match(self.step)
        return (int(match.group(1)), int(match.group(2))) if match else None

    @property
    def maximum(self) -> tuple[int, int] | None:
        match = _MAX_STEP.match(self.step)
        return (int(match.group(1)), int(match.group(2))) if match else None

    @property
    def minimum(self) -> tuple[int, int] | None:
        match = _MIN_STEP.match(self.step)
        return (int(match.group(1)), int(match.group(2))) if match else None


def parse_index(text: str) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(":")
        if len(fields) < 6:
            raise ValueError(f"Malformed idx line: {line!r}")
        records.append(IndexRecord(
            message=int(fields[0]), offset=int(fields[1]),
            variable=fields[3], level=fields[4], step=fields[5],
        ))
    if not records:
        raise ValueError("Empty or unreadable idx content")
    return records


def add_lengths(records: list[IndexRecord], file_size: int) -> list[IndexRecord]:
    ordered = sorted(records, key=lambda record: record.offset)
    sized: list[IndexRecord] = []
    for index, record in enumerate(ordered):
        end = ordered[index + 1].offset if index + 1 < len(ordered) else file_size
        if end <= record.offset:
            raise ValueError(f"Non-increasing offsets near message {record.message}")
        sized.append(IndexRecord(**{**record.__dict__, "length": end - record.offset}))
    return sized


def index_lengths(records: list[IndexRecord]) -> list[IndexRecord]:
    """Message lengths from consecutive offsets; the final message is open-ended (no HEAD)."""
    ordered = sorted(records, key=lambda record: record.offset)
    sized: list[IndexRecord] = []
    for index, record in enumerate(ordered):
        if index + 1 < len(ordered):
            end = ordered[index + 1].offset
            if end <= record.offset:
                raise ValueError(f"Non-increasing offsets near message {record.message}")
            length = end - record.offset
        else:
            length = None
        sized.append(IndexRecord(**{**record.__dict__, "length": length}))
    return sized


def select_instant(
    records: list[IndexRecord], variable: str, level: str, lead_hours: list[int]
) -> list[IndexRecord]:
    wanted = set(lead_hours)
    chosen = [
        record for record in records
        if record.variable == variable and record.level == level
        and record.instant_hour in wanted
    ]
    missing = sorted(wanted - {record.instant_hour for record in chosen})
    if missing:
        raise ValueError(f"{variable} {level}: missing lead hours {missing}")
    return sorted(chosen, key=lambda record: record.instant_hour)


def select_precip_6h_buckets(
    records: list[IndexRecord], variable: str, lead_days: int
) -> list[IndexRecord]:
    """Six-hour accumulation buckets up to lead_days (four per day summed to daily rainfall)."""
    horizon = lead_days * 24
    chosen = [
        record for record in records
        if record.variable == variable and record.level == "surface"
        and record.accumulation is not None
        and record.accumulation[1] - record.accumulation[0] == 6
        and record.accumulation[1] <= horizon
    ]
    ends = {record.accumulation[1] for record in chosen}
    missing = sorted(set(range(6, horizon + 1, 6)) - ends)
    if missing:
        raise ValueError(f"{variable}: missing 6h accumulation ends {missing}")
    return sorted(chosen, key=lambda record: record.accumulation[1])


def select_avg_6h_buckets(
    records: list[IndexRecord], variable: str, level: str, lead_days: int
) -> list[IndexRecord]:
    """Six-hour time-average buckets up to lead_days (four per day meaned to a daily mean)."""
    horizon = lead_days * 24
    chosen = [
        record for record in records
        if record.variable == variable and record.level == level
        and record.average is not None
        and record.average[1] - record.average[0] == 6
        and record.average[1] <= horizon
    ]
    ends = {record.average[1] for record in chosen}
    missing = sorted(set(range(6, horizon + 1, 6)) - ends)
    if missing:
        raise ValueError(f"{variable}: missing 6h average ends {missing}")
    return sorted(chosen, key=lambda record: record.average[1])


def select_max_6h_buckets(
    records: list[IndexRecord], variable: str, level: str, lead_days: int
) -> list[IndexRecord]:
    """Six-hour maximum buckets up to lead_days (four per day reduced to a daily max)."""
    horizon = lead_days * 24
    chosen = [
        record for record in records
        if record.variable == variable and record.level == level
        and record.maximum is not None
        and record.maximum[1] - record.maximum[0] == 6
        and record.maximum[1] <= horizon
    ]
    ends = {record.maximum[1] for record in chosen}
    missing = sorted(set(range(6, horizon + 1, 6)) - ends)
    if missing:
        raise ValueError(f"{variable}: missing 6h maximum ends {missing}")
    return sorted(chosen, key=lambda record: record.maximum[1])


def select_min_6h_buckets(
    records: list[IndexRecord], variable: str, level: str, lead_days: int
) -> list[IndexRecord]:
    """Six-hour minimum buckets up to lead_days (four per day reduced to a daily min)."""
    horizon = lead_days * 24
    chosen = [
        record for record in records
        if record.variable == variable and record.level == level
        and record.minimum is not None
        and record.minimum[1] - record.minimum[0] == 6
        and record.minimum[1] <= horizon
    ]
    ends = {record.minimum[1] for record in chosen}
    missing = sorted(set(range(6, horizon + 1, 6)) - ends)
    if missing:
        raise ValueError(f"{variable}: missing 6h minimum ends {missing}")
    return sorted(chosen, key=lambda record: record.minimum[1])


def merge_ranges(records: list[IndexRecord]) -> list[tuple[int, int | None]]:
    spans = sorted(
        (r.offset, None if r.length is None else r.offset + r.length) for r in records
    )
    merged: list[list] = []
    for start, end in spans:
        if merged and (merged[-1][1] is None or start <= merged[-1][1]):
            previous = merged[-1]
            previous[1] = None if (previous[1] is None or end is None) else max(previous[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def selected_bytes(records: list[IndexRecord]) -> int:
    return sum(record.length for record in records if record.length is not None)


def build_session(pool: int = 32) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _variable_url(config: dict, year: int, init: str, member: str, var_file: str) -> str:
    source = config["source"]
    return (f"{source['base_url']}/{source['prefix']}/{year}/{init}/{member}/"
            f"{source['day_range_folder']}/{var_file}_{init}_{member}.grib2")


def fetch_index(session: requests.Session, grib_url: str, timeout: int) -> list[IndexRecord]:
    idx = session.get(grib_url + ".idx", timeout=timeout)
    idx.raise_for_status()
    return index_lengths(parse_index(idx.text))


def fetch_messages(
    session: requests.Session, grib_url: str, records: list[IndexRecord], timeout: int
) -> bytes:
    chunks: list[bytes] = []
    for start, end in merge_ranges(records):
        header = f"bytes={start}-" if end is None else f"bytes={start}-{end - 1}"
        response = session.get(grib_url, headers={"Range": header}, timeout=timeout)
        response.raise_for_status()
        chunks.append(response.content)
    return b"".join(chunks)


_GRIB_VAR_LEVEL: dict[str, tuple[str, str]] = {
    "mslp": ("PRES", "mean sea level"),
    "u850": ("UGRD", "850 mb"),
    "v850": ("VGRD", "850 mb"),
    "q850": ("SPFH", "850 mb"),
    "gh500": ("HGT", "500 mb"),
}

_TARGET_VARIABLE: dict[str, tuple[str, int | None]] = {
    "rainfall": ("total_precipitation", None),
    "mslp": ("mean_sea_level_pressure", None),
    "u850": ("u_wind", 850),
    "v850": ("v_wind", 850),
    "q850": ("specific_humidity", 850),
    "gh500": ("geopotential_height", 500),
}


def _open_concat(path: Path) -> xr.DataArray:
    import cfgrib

    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    arrays = [ds[list(ds.data_vars)[0]] for ds in datasets]
    combined = xr.concat(arrays, dim="step") if len(arrays) > 1 else arrays[0]
    return combined.sortby("step")


def _step_hours(field: xr.DataArray) -> np.ndarray:
    return field["step"].values.astype("timedelta64[h]").astype(int)


def _strip_to_grid(field: xr.DataArray) -> xr.DataArray:
    keep = {"lead", "latitude", "longitude"}
    return field.drop_vars([coord for coord in field.coords if coord not in keep])


def _decode_instant(path: Path, lead_days: int) -> xr.DataArray:
    field = _open_concat(path)
    hours = _step_hours(field)
    field = field.assign_coords(lead=("step", (hours // 24).astype(int)))
    return _strip_to_grid(field.swap_dims({"step": "lead"}).sortby("lead"))


def _decode_precip_daily(path: Path, lead_days: int) -> xr.DataArray:
    field = _open_concat(path)
    day = ((_step_hours(field) - 1) // 24 + 1).astype(int)
    field = field.assign_coords(lead=("step", day))
    return _strip_to_grid(field.groupby("lead").sum().sortby("lead"))


def _decode_avg_daily(path: Path, lead_days: int) -> xr.DataArray:
    field = _open_concat(path)
    day = ((_step_hours(field) - 1) // 24 + 1).astype(int)
    field = field.assign_coords(lead=("step", day))
    return _strip_to_grid(field.groupby("lead").mean().sortby("lead"))


def _decode_max_daily(path: Path, lead_days: int) -> xr.DataArray:
    field = _open_concat(path)
    day = ((_step_hours(field) - 1) // 24 + 1).astype(int)
    field = field.assign_coords(lead=("step", day))
    return _strip_to_grid(field.groupby("lead").max().sortby("lead"))


def _decode_min_daily(path: Path, lead_days: int) -> xr.DataArray:
    field = _open_concat(path)
    day = ((_step_hours(field) - 1) // 24 + 1).astype(int)
    field = field.assign_coords(lead=("step", day))
    return _strip_to_grid(field.groupby("lead").min().sortby("lead"))


def _target_centres(domain: dict, step: float) -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(domain["south"] + step / 2, domain["north"], step)
    lons = np.arange(domain["west"] + step / 2, domain["east"], step)
    return lats, lons


def crop_and_regrid(field: xr.DataArray, domain: dict, step: float) -> xr.DataArray:
    field = field.sortby("latitude").sortby("longitude")
    margin = 1.0
    cropped = field.sel(
        latitude=slice(domain["south"] - margin, domain["north"] + margin),
        longitude=slice(domain["west"] - margin, domain["east"] + margin),
    )
    lats, lons = _target_centres(domain, step)
    return cropped.interp(latitude=lats, longitude=lons, method="linear")


def _select_for(base: str, records: list[IndexRecord], lead_days: int,
                entry: dict | None = None) -> list[IndexRecord]:
    if base == "rainfall":
        return select_precip_6h_buckets(records, "APCP", lead_days)
    if base == "olr":
        return select_avg_6h_buckets(records, "ULWRF", "top of atmosphere", lead_days)
    # Instant fields read their GRIB variable/level from the config entry when
    # present, falling back to the built-in map for the original channel set.
    if entry and entry.get("grib_var"):
        variable, level = entry["grib_var"], entry["grib_level"]
    else:
        variable, level = _GRIB_VAR_LEVEL[base]
    # Per-variable sampling. Non-diurnal fields (pressure, upper-air) keep the
    # 00Z snapshot; surface fields with a diurnal cycle must declare `timing`
    # (daily_max/daily_min/daily_mean, or an afternoon snapshot) or they get
    # silently sampled at dawn — the bug that produced dawn CAPE and 00Z tmp_2m.
    timing = (entry or {}).get("timing", "instant")
    if timing == "daily_max":
        return select_max_6h_buckets(records, variable, level, lead_days)
    if timing == "daily_min":
        return select_min_6h_buckets(records, variable, level, lead_days)
    if timing == "daily_mean":
        return select_avg_6h_buckets(records, variable, level, lead_days)
    if timing == "afternoon":
        offset = int((entry or {}).get("hour_offset", 9))  # 09Z ~ 14:30 IST
        return select_instant(records, variable, level,
                              [24 * d + offset for d in range(1, lead_days + 1)])
    if timing != "instant":
        raise ValueError(f"{base}: unknown timing {timing!r}")
    return select_instant(records, variable, level, [24 * d for d in range(1, lead_days + 1)])


def fetch_field_raw(session, config, year, init, member, base, lead_days, timeout, staging,
                    index_cache=None) -> Path:
    var_file = config["channel_sources"][base]["file"]
    url = _variable_url(config, year, init, member, var_file)
    if index_cache is not None and (member, var_file) in index_cache:
        records = index_cache[(member, var_file)]
    else:
        records = fetch_index(session, url, timeout)
    raw = fetch_messages(session, url, _select_for(base, records, lead_days,
                                                   config["channel_sources"][base]), timeout)
    path = staging / f"{member}_{base}.grib2"
    path.write_bytes(raw)
    return path


def decode_field(base: str, path: Path, lead_days: int, timing: str = "instant") -> xr.DataArray:
    if base == "rainfall":
        return _decode_precip_daily(path, lead_days)
    if base == "olr":
        return _decode_avg_daily(path, lead_days)
    if timing == "daily_max":
        return _decode_max_daily(path, lead_days)
    if timing == "daily_min":
        return _decode_min_daily(path, lead_days)
    if timing == "daily_mean":
        return _decode_avg_daily(path, lead_days)
    # "instant" and "afternoon" are both single-record-per-day snapshots.
    return _decode_instant(path, lead_days)


def _decode_and_crop(base: str, path_str: str, lead_days: int, domain: dict, step: float,
                     timing: str = "instant") -> xr.DataArray:
    return crop_and_regrid(decode_field(base, Path(path_str), lead_days, timing), domain, step)


def execute_init(session, config, year: int, init: str, members: list[str],
                 *, decode_pool: ProcessPoolExecutor | None = None) -> tuple[Path, xr.Dataset]:
    domain = config["domain"]
    step = float(domain["target_grid_degrees"])
    timeout = int(config["limits"]["request_timeout_seconds"])
    lead_days = int(config["schedule"]["retained_lead_days"])
    limits = config["limits"]
    workers = int(limits.get("concurrent_file_workers", limits.get("concurrent_data_requests", 8)))
    staging = PROJECT_ROOT / config["output"]["staging_root"]
    staging.mkdir(parents=True, exist_ok=True)

    tasks = [(member, base) for base in config["channel_sources"] for member in members]

    # Phase 1: fetch each (member, variable-file) index exactly once. Pressure
    # levels of the same variable share one file, so this avoids re-downloading
    # the same .idx for every level.
    unique_files = {(member, config["channel_sources"][base]["file"]) for member, base in tasks}
    index_cache: dict[tuple[str, str], list] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        index_futures = {
            pool.submit(fetch_index, session,
                        _variable_url(config, year, init, member, var_file), timeout): (member, var_file)
            for member, var_file in unique_files
        }
        for future in as_completed(index_futures):
            index_cache[index_futures[future]] = future.result()

    # Phase 2: fetch each base's messages using the cached index.
    paths: dict[tuple[str, str], Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_field_raw, session, config, year, init, member, base,
                        lead_days, timeout, staging, index_cache): (member, base)
            for member, base in tasks
        }
        for future in as_completed(futures):
            paths[futures[future]] = future.result()

    fields: dict[tuple[str, str], xr.DataArray] = {}
    if decode_pool is not None:
        jobs = {
            decode_pool.submit(_decode_and_crop, base, str(paths[(member, base)]),
                               lead_days, domain, step,
                               config["channel_sources"][base].get("timing", "instant")): (member, base)
            for member, base in tasks
        }
        for job in as_completed(jobs):
            fields[jobs[job]] = job.result()
    else:
        for member, base in tasks:
            fields[(member, base)] = _decode_and_crop(
                base, str(paths[(member, base)]), lead_days, domain, step,
                config["channel_sources"][base].get("timing", "instant"))

    data_vars: dict[str, xr.DataArray] = {}
    for base, entry in config["channel_sources"].items():
        # Original channels keep their built-in target name/level; new channels
        # take a flat name from the config (defaulting to the base name).
        target_name, level = _TARGET_VARIABLE.get(
            base, (entry.get("target", base), entry.get("out_level")))
        stacked = xr.concat([fields[(member, base)] for member in members],
                            dim=pd.Index(members, name="member")).expand_dims(run=[init])
        if level is not None:
            stacked = stacked.expand_dims(level=[level])
        data_vars[target_name] = stacked

    dataset = xr.Dataset(data_vars)
    out_root = PROJECT_ROOT / config["output"]["regional_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{init}.nc"
    dataset.to_netcdf(out_path)
    shutil.rmtree(staging, ignore_errors=True)
    return out_path, dataset


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["init"])
    return done


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_acquisition(
    session, config: dict, years: list[int], members: list[str],
    *, limit: int | None = None, resume: bool = True,
) -> None:
    output = config["output"]
    regional_root = PROJECT_ROOT / output["regional_root"]
    manifest = PROJECT_ROOT / output["manifest"]
    acquisition_log = PROJECT_ROOT / output["acquisition_log"]
    checkpoint = PROJECT_ROOT / output["checkpoint"]
    staging = PROJECT_ROOT / output["staging_root"]
    ceiling = float(config["limits"]["storage_ceiling_gb"]) * 1e9
    stop_bytes = ceiling * float(config["limits"]["stop_at_fraction"])
    decode_workers = int(config["limits"].get("decode_workers", 6))

    completed = _load_completed(checkpoint) if resume else set()
    all_dates = [(year, init) for year in years for init in _dates_for_year(year)]
    pending = [(year, init) for year, init in all_dates if init not in completed]
    if limit is not None:
        pending = pending[:limit]

    print(f"Years {years[0]}-{years[-1]}: {len(all_dates)} dates, "
          f"{len(completed)} already done, {len(pending)} to process"
          + (f" (limited to {limit})" if limit is not None else "")
          + f"  [{decode_workers} decode procs]")
    _append_jsonl(acquisition_log, {"event": "run_start", "years": years,
                                    "members": members, "pending": len(pending), "time": _now()})

    downloaded = failed = 0
    with ProcessPoolExecutor(max_workers=decode_workers) as decode_pool:
        for index, (year, init) in enumerate(pending, start=1):
            retained = _dir_size_bytes(regional_root)
            if retained >= stop_bytes:
                print(f"Storage ceiling: {retained / 1e9:.1f} GB >= {stop_bytes / 1e9:.1f} GB - stopping.")
                _append_jsonl(acquisition_log, {"event": "stopped_storage",
                                                "retained_gb": retained / 1e9, "time": _now()})
                break
            started = time.time()
            try:
                out_path, dataset = execute_init(session, config, year, init, members,
                                                 decode_pool=decode_pool)
                size = out_path.stat().st_size
                seconds = round(time.time() - started, 1)
                _append_jsonl(manifest, {
                    "init": init, "year": year, "members": members, "bytes": size,
                    "sha256": _sha256(out_path), "dims": dict(dataset.sizes),
                    "variables": list(dataset.data_vars), "seconds": seconds, "time": _now(),
                })
                _append_jsonl(checkpoint, {"init": init})
                _append_jsonl(acquisition_log, {"event": "downloaded", "init": init,
                                                "bytes": size, "seconds": seconds, "time": _now()})
                downloaded += 1
                print(f"[{index}/{len(pending)}] {init}  {size / 1e6:5.1f} MB  {seconds:5.1f}s  "
                      f"(retained {retained / 1e9:.2f} GB)")
            except Exception as exc:
                shutil.rmtree(staging, ignore_errors=True)
                _append_jsonl(acquisition_log, {"event": "failed", "init": init,
                                                "error": repr(exc), "time": _now()})
                failed += 1
                print(f"[{index}/{len(pending)}] {init}  FAILED: {exc}")

    _append_jsonl(acquisition_log, {"event": "run_end", "downloaded": downloaded,
                                    "failed": failed, "time": _now()})
    print(f"\nRun complete: {downloaded} downloaded, {failed} failed. Checkpoint: {checkpoint}")


def plan_member_day(session, config: dict, year: int, init: str, member: str) -> dict:
    lead_days = int(config["schedule"]["retained_lead_days"])
    lead_hours = [24 * day for day in range(1, lead_days + 1)]
    plan = {"init": init, "member": member, "fields": {}, "total_bytes": 0}
    for base, spec in config["channel_sources"].items():
        url = _variable_url(config, year, init, member, spec["file"])
        records = fetch_index(session, url, config["limits"]["request_timeout_seconds"])
        if base == "rainfall":
            selected = select_precip_6h_buckets(records, "APCP", lead_days)
        else:
            variable, level = _GRIB_VAR_LEVEL[base]
            selected = select_instant(records, variable, level, lead_hours)
        size = selected_bytes(selected)
        plan["fields"][base] = {"url": url, "messages": len(selected), "bytes": size,
                                "ranges": len(merge_ranges(selected))}
        plan["total_bytes"] += size
    return plan


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dates_for_year(year: int):
    day = date(year, 1, 1)
    while day.year == year:
        yield day.strftime("%Y%m%d") + "00"
        day += timedelta(days=1)


def _print_plan(session, config: dict, year: int, sample_init: str) -> None:
    member = config["members"][0]
    print(f"Planning one init/member: {sample_init} / {member} (year {year})\n")
    plan = plan_member_day(session, config, year, sample_init, member)
    for base, info in plan["fields"].items():
        print(f"  {base:9s} {info['messages']:3d} msgs  "
              f"{info['ranges']:2d} ranges  {info['bytes'] / 1e6:7.2f} MB")
    per_member_mb = plan["total_bytes"] / 1e6
    members, years = len(config["members"]), len(config["period"]["years"])
    print(f"\n  per init/member : {per_member_mb:7.2f} MB")
    print(f"  per init        : {per_member_mb * members:7.2f} MB ({members} members)")
    print(f"  per year (~365d): {per_member_mb * members * 365 / 1e3:7.2f} GB")
    print(f"  full {years} years : {per_member_mb * members * 365 * years / 1e3:7.2f} GB (raw; deleted after crop)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--year", type=int, help="Run one whole year (default: validation_slice year).")
    parser.add_argument("--all", action="store_true", help="Run every year in the config period.")
    parser.add_argument("--date", type=str, help="Run a single init YYYYMMDD (00Z) only.")
    parser.add_argument("--execute", action="store_true", help="Download (else dry-run plan).")
    parser.add_argument("--max-members", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    session = build_session()
    members = config["members"][:args.max_members] if args.max_members else config["members"]
    print(f"Config: {args.config}")
    print(f"Archive: {config['source']['base_url']}/{config['source']['prefix']}")

    if not args.execute:
        year = args.year or int(config["validation_slice"]["year"])
        sample_init = (args.date + "00") if args.date else next(_dates_for_year(year))
        _print_plan(session, config, year, sample_init)
        print("\nDry run only. Add --execute to download, or --execute --all for the full set.")
        return

    if args.date:
        init, year = args.date + "00", int(args.date[:4])
        print(f"\nExecuting single date {init} for members {members} ...")
        out_path, dataset = execute_init(session, config, year, init, members)
        print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e6:.3f} MB)  dims: {dict(dataset.sizes)}")
        return

    years = config["period"]["years"] if args.all else [args.year or int(config["validation_slice"]["year"])]
    run_acquisition(session, config, [int(y) for y in years], members,
                    limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
