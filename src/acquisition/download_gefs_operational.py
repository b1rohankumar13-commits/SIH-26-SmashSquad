"""Download operational GEFSv12 (noaa-gefs-pds) into the reforecast per-init NetCDF layout.

The operational archive stores one GRIB2 file per member and forecast hour, where the
reforecast stores one file per variable. This fetches the .idx of every needed hour,
selects messages with the reforecast selectors, byte-range downloads them into one GRIB
per (member, field), and reuses the reforecast decode/crop/regrid - so the output is a
drop-in for the cache builders (same variables, levels, members, leads and grid).

    python -m src.acquisition.download_gefs_operational --date 20220601      # one init
    python -m src.acquisition.download_gefs_operational --execute            # whole period
    python -m src.acquisition.download_gefs_operational --execute --start 2022-05-01 --end 2022-08-31
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from src.acquisition.download_gefs_reforecast import (
    PROJECT_ROOT, _TARGET_VARIABLE, _append_jsonl, _decode_and_crop, _now, _select_for,
    build_session, fetch_index, fetch_messages,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "acquisition_gefs_operational.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_url(config: dict, init: str, member: str, file_set: dict, hour: int) -> str:
    day = init[:8]
    return (f"{config['source']['base_url']}/gefs.{day}/00/atmos/{file_set['folder']}/"
            f"ge{member}.t00z.{file_set['stem']}.f{hour:03d}")


def hours_for(file_set: dict, lead_days: int) -> list[int]:
    step = 6 if file_set["hours"] == "every_6h" else 24
    return list(range(step, lead_days * 24 + 1, step))


def stores_for(config: dict, init: str) -> list[str]:
    month = int(init[4:6])
    return [name for name, s in config["stores"].items() if month in s["months"]]


def fetch_member_indexes(session, config: dict, init: str, members: list[str], lead_days: int,
                         timeout: int, workers: int) -> dict[str, list[tuple[object, str]]]:
    """member -> [(record, url)], ordered by file-set priority so the first match wins."""
    jobs = [(m, k, fs, h) for m in members for k, fs in enumerate(config["source"]["file_sets"])
            for h in hours_for(fs, lead_days)]
    found: dict[tuple, list] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_index, session, file_url(config, init, m, fs, h), timeout): (m, k, h, fs)
                   for m, k, fs, h in jobs}
        for f in as_completed(futures):
            m, k, h, fs = futures[f]
            found[(m, k, h)] = [(r, file_url(config, init, m, fs, h)) for r in f.result()]
    return {m: [pair for key in sorted(k for k in found if k[0] == m) for pair in found[key]]
            for m in members}


def select_records(base: str, entry: dict, pairs: list[tuple[object, str]], lead_days: int):
    """Run the reforecast selector over all hours' records; keep the first copy of each message."""
    url_of = {id(r): u for r, u in pairs}
    chosen, seen = [], set()
    for r in _select_for(base, [r for r, _ in pairs], lead_days, entry or None):
        key = (r.variable, r.level, r.step)
        if key not in seen:
            seen.add(key); chosen.append((r, url_of[id(r)]))
    return chosen


def fetch_field(session, chosen, path: Path, timeout: int) -> Path:
    by_url: dict[str, list] = {}
    for r, u in chosen:
        by_url.setdefault(u, []).append(r)
    path.write_bytes(b"".join(fetch_messages(session, u, rs, timeout) for u, rs in by_url.items()))
    return path


def execute_init(session, config: dict, init: str, members: list[str],
                 decode_pool: ProcessPoolExecutor | None = None) -> dict[str, Path]:
    domain, lim = config["domain"], config["limits"]
    step, lead_days = float(domain["target_grid_degrees"]), int(config["schedule"]["retained_lead_days"])
    timeout, workers = int(lim["request_timeout_seconds"]), int(lim["concurrent_file_workers"])
    staging = PROJECT_ROOT / config["output"]["staging_root"] / init
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    stores = stores_for(config, init)
    pairs = fetch_member_indexes(session, config, init, members, lead_days, timeout, workers)

    tasks = [(s, base, m) for s in stores for base in config["stores"][s]["channel_sources"] for m in members]
    paths: dict[tuple, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for s, base, m in tasks:
            entry = config["stores"][s]["channel_sources"][base]
            chosen = select_records(base, entry, pairs[m], lead_days)
            futures[pool.submit(fetch_field, session, chosen, staging / f"{s}_{base}_{m}.grib2", timeout)] = (s, base, m)
        for f in as_completed(futures):
            paths[futures[f]] = f.result()

    def timing(s, base):
        return config["stores"][s]["channel_sources"][base].get("timing", "instant")
    fields: dict[tuple, xr.DataArray] = {}
    if decode_pool is not None:
        jobs = {decode_pool.submit(_decode_and_crop, base, str(paths[(s, base, m)]), lead_days, domain, step,
                                   timing(s, base)): (s, base, m) for s, base, m in tasks}
        for j in as_completed(jobs):
            fields[jobs[j]] = j.result()
    else:
        for s, base, m in tasks:
            fields[(s, base, m)] = _decode_and_crop(base, str(paths[(s, base, m)]), lead_days, domain, step,
                                                    timing(s, base))

    written = {}
    for s in stores:
        data_vars = {}
        for base, entry in config["stores"][s]["channel_sources"].items():
            name, level = _TARGET_VARIABLE.get(base, (entry.get("target", base), entry.get("out_level")))
            stacked = xr.concat([fields[(s, base, m)] for m in members],
                                dim=pd.Index(members, name="member")).expand_dims(run=[init])
            if level is not None:
                stacked = stacked.expand_dims(level=[level])
            data_vars[name] = stacked
        out = PROJECT_ROOT / config["stores"][s]["regional_root"] / f"{init}.nc"
        out.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(data_vars).to_netcdf(out)
        written[s] = out
    shutil.rmtree(staging, ignore_errors=True)
    return written


def period_inits(config: dict) -> list[str]:
    start, end = (date.fromisoformat(config["period"][k]) for k in ("start", "end"))
    out, d = [], start
    while d <= end:
        out.append(d.strftime("%Y%m%d") + "00"); d += timedelta(days=1)
    return [i for i in out if stores_for(config, i)]


def done_inits(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["init"] for line in path.read_text().splitlines() if line.strip()}


def run(config: dict, members: list[str], limit: int | None = None) -> None:
    session = build_session()
    checkpoint = PROJECT_ROOT / config["output"]["checkpoint"]
    log = PROJECT_ROOT / config["output"]["acquisition_log"]
    done = done_inits(checkpoint)
    pending = [i for i in period_inits(config) if i not in done][:limit]
    print(f"{len(pending)} inits to fetch ({len(done)} already done)", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=int(config["limits"]["decode_workers"])) as pool:
        for k, init in enumerate(pending, 1):
            t0 = time.time()
            try:
                written = execute_init(session, config, init, members, pool)
                _append_jsonl(checkpoint, {"init": init, "stores": sorted(written)})
                _append_jsonl(log, {"event": "downloaded", "init": init, "stores": sorted(written),
                                    "seconds": round(time.time() - t0, 1), "time": _now()})
                print(f"[{k}/{len(pending)}] {init} {'+'.join(sorted(written))} {time.time()-t0:.0f}s", flush=True)
            except Exception as exc:
                failed += 1
                shutil.rmtree(PROJECT_ROOT / config["output"]["staging_root"] / init, ignore_errors=True)
                _append_jsonl(log, {"event": "failed", "init": init, "error": repr(exc), "time": _now()})
                print(f"[{k}/{len(pending)}] {init} FAILED: {exc!r}", flush=True)
    print(f"done, {failed} failed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--date", help="Fetch a single init YYYYMMDD and exit.")
    ap.add_argument("--execute", action="store_true", help="Fetch every init in the config period.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start", help="Override period start (YYYY-MM-DD), e.g. to run shards in parallel.")
    ap.add_argument("--end", help="Override period end (YYYY-MM-DD).")
    args = ap.parse_args()
    config = load_config(args.config)
    for k in ("start", "end"):
        if getattr(args, k):
            config["period"][k] = getattr(args, k)
    members = config["members"]
    if args.date:
        t0 = time.time()
        written = execute_init(build_session(), config, args.date + "00", members)
        for s, p in written.items():
            print(f"{s}: {p} ({p.stat().st_size / 1e6:.1f} MB)")
        print(f"{time.time() - t0:.0f}s")
    elif args.execute:
        run(config, members, args.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
