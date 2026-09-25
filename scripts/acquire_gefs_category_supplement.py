"""Acquire only missing category fields; reuse approved GEFS scope and crop code."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(r'C:\Users\b1sun\OneDrive\Desktop\sih_project')
sys.path.insert(0, str(ROOT))
import yaml
import eccodes as ec
import numpy as np
from src.acquisition import download_gefs_reforecast as gefs

SELECTORS = [
    {'variable': 'RH', 'level': '2 m above ground'},
    {'variable': 'SOILW', 'level': '0-0.1 m below ground'},
    {'variable': 'TCDC', 'level': 'entire atmosphere'},
]

_CHECKPOINT_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def emit(payload):
    with _PRINT_LOCK:
        print(json.dumps(payload), flush=True)

def retained_bytes(directory):
    total=0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False): total+=retained_bytes(entry.path)
            elif entry.is_file(follow_symlinks=False): total+=entry.stat().st_size
    return total

def inspect(path, date, lead):
    result = []
    with path.open('rb') as handle:
        while (gid := ec.codes_grib_new_from_file(handle)) is not None:
            try:
                keys = ['shortName','typeOfLevel','level','units','stepType','startStep','endStep','validityDate','validityTime','Ni','Nj']
                meta = {k: ec.codes_get(gid,k) for k in keys}
                values = ec.codes_get_values(gid)
                missing = ec.codes_get(gid,'missingValue')
                usable = values[np.isfinite(values) & (values != missing)]
                assert usable.size, (path,meta)
                assert meta['Ni'] == 35 and meta['Nj'] == 33, meta
                assert meta['endStep'] == lead, meta
                valid = dt.datetime.strptime(date,'%Y%m%d') + dt.timedelta(hours=lead)
                assert meta['validityDate'] == int(valid.strftime('%Y%m%d')) and meta['validityTime'] == 0, meta
                meta.update(minimum=float(usable.min()),maximum=float(usable.max()),valid_cells=int(usable.size),missing_cells=int(values.size-usable.size))
                result.append(meta)
            finally:
                ec.codes_release(gid)
    assert len(result) == 3, result
    assert {r['shortName'] for r in result} == {'2r','soilw','tcc'}, result
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',required=True)
    p.add_argument('--end',required=True)
    p.add_argument('--control-only',action='store_true')
    p.add_argument('--date-workers',type=int,default=1)
    a=p.parse_args()
    if not 1 <= a.date_workers <= 10:
        p.error('--date-workers must be between 1 and 10')
    start,end=dt.date.fromisoformat(a.start),dt.date.fromisoformat(a.end)
    assert dt.date(2021,1,1)<=start<=end<=dt.date(2023,12,31)
    config=yaml.safe_load((ROOT/'configs/acquisition_gefs_2021_2023.yaml').read_text())
    base=ROOT/'data/interim/forecasts/gefs/category_supplement_2021_2023'
    metadata=ROOT/'data/metadata'
    metadata.mkdir(parents=True,exist_ok=True)
    members=config['members'][:1] if a.control_only else config['members']
    checkpoint=metadata/('gefs_category_supplement_control_dates.jsonl' if a.control_only else 'gefs_category_supplement_completed_dates.jsonl')
    completed={json.loads(x)['date'] for x in checkpoint.read_text().splitlines()} if checkpoint.exists() else set()
    spec={
        'source_configuration':'configs/acquisition_gefs_2021_2023.yaml',
        'selectors':SELECTORS,'members':config['members'],
        'lead_hours':config['schedule']['daily_lead_hours'],
        'categories':{'heat_waves':['2m_relative_humidity','soil_water_0_10cm','total_cloud_cover'],
                      'active_break_monsoon':['total_cloud_cover']},
        'time_semantics':'RH and soil moisture are instantaneous at each daily valid time; cloud retains GRIB startStep/endStep (typically final six hours), never called a full daily average.',
        'regional_root':str(base.relative_to(ROOT)),
        'climatology_years':None,'training_years':None,
    }
    (metadata/'gefs_category_supplement_spec.json').write_text(json.dumps(spec,indent=2))
    # Count the complete retained project data against the approved ceiling.
    emit({'event':'storage_audit_started','date_workers':a.date_workers})
    retained=retained_bytes(ROOT/'data')
    ceiling=config['limits']['storage_ceiling_gb']*1024**3*config['limits']['stop_at_fraction']
    projected=1095*len(config['members'])*10*len(SELECTORS)*65536
    if retained+projected>ceiling or shutil.disk_usage(ROOT).free<projected:
        raise RuntimeError('Supplement would exceed approved retained-data ceiling or free disk capacity.')
    current=start
    dates=[]
    while current<=end:
        date=current.strftime('%Y%m%d');current+=dt.timedelta(days=1)
        if date not in completed:
            dates.append(date)

    def process_date_once(date):
        started=time.monotonic()
        jobs=[(date,m,int(lead),'pgrb2a',SELECTORS) for m in members for lead in config['schedule']['daily_lead_hours']]
        if a.date_workers == 1:
            with cf.ThreadPoolExecutor(max_workers=config['limits']['concurrent_index_requests']) as pool:
                plan=list(pool.map(lambda job:gefs._plan_one(config,*job),jobs))
        else:
            # Ten date workers already provide bounded cross-date concurrency.
            # Keep each worker sequential so aggregate index/data requests never
            # multiply by the per-date thread counts in the base configuration.
            plan=[gefs._plan_one(config,*job) for job in jobs]
        # Global bytes bound staging; conservative 64 KiB per message bounds regional output.
        transfer=sum(item.selected_bytes for item in plan)
        if shutil.disk_usage(ROOT).free<transfer:
            raise RuntimeError('Supplement would exceed approved retained-data ceiling or free disk capacity.')
        emit({'event':'plan','date':date,'objects':len(plan),'transfer_bytes':transfer,'retained_project_bytes':retained,'supplement_upper_bound_bytes':projected})
        def download(item):
            return gefs._download_one(config,item,
                ROOT/'data/raw/forecasts/gefs/category_supplement_canary',
                ROOT/'data/raw/forecasts/gefs/.staging_category_supplement',
                base,metadata/'gefs_category_supplement_source_manifest.jsonl',
                metadata/'gefs_category_supplement_regional_manifest.jsonl')
        if a.date_workers == 1:
            with cf.ThreadPoolExecutor(max_workers=config['limits']['concurrent_data_requests']) as pool:
                results=list(pool.map(download,plan))
        else:
            results=[download(item) for item in plan]
        # Verify every retained message, including checkpoint-resumed files.
        inventory={}
        for item in plan:
            path=gefs._destination(base,item)
            with gefs._CROP_LOCK:
                inventory[str(path.relative_to(base))]=inspect(path,date,item.lead_hour)
        qc=metadata/'gefs_category_supplement_qc'/f'{date}_{"control" if a.control_only else "ensemble"}.json'
        qc.parent.mkdir(exist_ok=True)
        qc.write_text(json.dumps(inventory,indent=2))
        payload={'date':date,'status':'complete','objects':len(plan),'regional_bytes':sum(r['bytes'] for r in results),'transfer_bytes':transfer,'elapsed_seconds':round(time.monotonic()-started,2),'qc_path':str(qc)}
        with _CHECKPOINT_LOCK:
            with checkpoint.open('a') as handle:
                handle.write(json.dumps(payload)+'\n')
        emit(payload)

    def process_date(date):
        for attempt in range(1,4):
            try:
                process_date_once(date)
                return
            except Exception as error:
                emit({'event':'date_failed','date':date,'attempt':attempt,'error':repr(error)})
                if attempt == 3:
                    raise
                time.sleep(30)

    emit({'event':'run_started','date_workers':a.date_workers,'remaining_dates':len(dates)})
    failures=[]
    with cf.ThreadPoolExecutor(max_workers=a.date_workers,thread_name_prefix='gefs-date') as pool:
        future_dates={pool.submit(process_date,date):date for date in dates}
        for future in cf.as_completed(future_dates):
            try:
                future.result()
            except Exception as error:
                failures.append((future_dates[future],repr(error)))
    if failures:
        raise RuntimeError(f'{len(failures)} dates failed after retries: {failures[:5]}')
    emit({'event':'run_complete','date_workers':a.date_workers,'completed_dates':len(dates)})

if __name__=='__main__':
    for attempt in range(1,4):
        try:
            main()
            break
        except Exception as error:
            print(json.dumps({'event':'run_failed','attempt':attempt,'error':repr(error)}),flush=True)
            if attempt==3: raise
            time.sleep(30)
