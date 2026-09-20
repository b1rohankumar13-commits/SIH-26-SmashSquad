"""Acquire the user's confirmed IMD scope; preserve raw annual provider files."""
import argparse
import calendar
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import xarray as xr
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', required=True)
parser.add_argument('--skip-grids', action='store_true')
args = parser.parse_args()
root = Path(args.project_root).resolve()
raw = root / 'data/raw/observations/imd'
log = root / 'data/metadata/imd_acquisition_2021_2023.jsonl'
log.parent.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.headers['User-Agent'] = 'SIH-Forecast-Bust-Research/1.0 (official-data acquisition)'

def record(item):
    item['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    with log.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item) + '\n')
    print(json.dumps(item), flush=True)

def request(method, url, **kwargs):
    host = urlparse(url).hostname or ''
    if not (host == 'imdpune.gov.in' or host.endswith('.imdpune.gov.in') or host.endswith('.imd.gov.in')):
        raise ValueError('Non-IMD URL: ' + url)
    for attempt in range(3):
        try:
            response = session.request(method, url, timeout=(45, 180), **kwargs)
            response.raise_for_status()
            final_host = urlparse(response.url).hostname or ''
            if not (final_host == 'imdpune.gov.in' or final_host.endswith('.imdpune.gov.in') or final_host.endswith('.imd.gov.in')):
                raise ValueError('Non-IMD redirect')
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)

def validate(path, kind, year):
    size = path.stat().st_size
    details = {'bytes': size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if kind == 'rainfall':
        with xr.open_dataset(path) as ds:
            details['dimensions'] = dict(ds.sizes)
            details['variables'] = list(ds.data_vars)
            t = next(v for k, v in ds.coords.items() if k.lower() == 'time')
            dates = np.asarray(t.values).astype('datetime64[D]')
            expected = np.arange(np.datetime64(f'{year}-01-01'), np.datetime64(f'{year+1}-01-01'))
            if not np.array_equal(dates, expected):
                raise ValueError('Rainfall dates do not cover the complete year exactly')
            details['start_date'], details['end_date'] = str(dates[0]), str(dates[-1])
            details['coordinates'] = {k: {'first': float(v.values[0]), 'last': float(v.values[-1]), 'count': v.size} for k,v in ds.coords.items() if k.lower() in ('lat','latitude','lon','longitude')}
            details['fields'] = {}
            for k, v in ds.data_vars.items():
                if v.ndim == 3:
                    values = v.values
                    details['fields'][k] = {'units': str(v.attrs.get('units','')), 'missing_fraction': float(np.isnan(values).mean()), 'min': float(np.nanmin(values)), 'max': float(np.nanmax(values))}
                    if not np.isfinite(values).any() or np.nanmin(values) < 0:
                        raise ValueError('Invalid rainfall range')
    elif kind in ('maximum_temperature', 'minimum_temperature'):
        days = 366 if calendar.isleap(year) else 365
        if size != days * 31 * 31 * 4:
            raise ValueError(f'Temperature length mismatch: {size}')
        details['documented_grid'] = {'latitude': [7.5,37.5,1.0], 'longitude': [67.5,97.5,1.0], 'days': days, 'units':'degrees Celsius', 'missing_value':99.9}
        # Record diagnostics for both orders; do not silently choose a decoder.
        details['byte_order_diagnostics'] = {}
        for order in ('<f4','>f4'):
            values = np.fromfile(path, dtype=order)
            missing = np.isclose(values,99.9)
            finite = np.isfinite(values) & ~missing
            details['byte_order_diagnostics'][order] = {'missing_fraction':float(missing.mean()), 'finite_nonmissing_fraction':float(finite.mean()), 'min':float(values[finite].min()) if finite.any() else None, 'max':float(values[finite].max()) if finite.any() else None}
    elif kind == 'pdf':
        if not path.read_bytes().startswith(b'%PDF-') or size < 1024:
            raise ValueError('Not a valid PDF payload')
    elif kind == 'xlsx':
        import zipfile
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or 'xl/workbook.xml' not in z.namelist():
                raise ValueError('Invalid Excel workbook')
    return details

def acquire(target, kind, year, source_page, fetch):
    base = {'local_path':str(target.relative_to(root)), 'product':kind, 'year':year, 'source_page':source_page}
    try:
        if target.exists():
            record({**base, 'status':'existing_verified', **validate(target,kind,year)})
            return
        if shutil.disk_usage(root).free < 1024**3:
            raise RuntimeError('Less than 1 GiB free; stopped')
        print('Fetching ' + str(target), flush=True)
        response = fetch()
        content = response.content
        if len(content) > 512*1024**2:
            raise ValueError('Unexpectedly large provider response')
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + '.part.' + str(time.time_ns()))
        part.write_bytes(content)
        details = validate(part,kind,year)
        part.rename(target)
        record({**base,'status':'stored','resolved_url':response.url, **details})
    except Exception as e:
        record({**base,'status':'failed','error':str(e)})

def form_fetch(page, year):
    response = request('GET',page)
    soup = BeautifulSoup(response.text,'html.parser')
    for form in soup.find_all('form'):
        for select in form.find_all('select',attrs={'name':True}):
            option = next((o for o in select.find_all('option') if str(o.get('value','')).strip() == str(year)), None)
            if option is not None:
                fields = {i['name']:i.get('value','') for i in form.find_all('input',attrs={'type':'hidden','name':True})}
                fields[select['name']] = option['value']
                action = urljoin(response.url,form.get('action') or response.url)
                method = form.get('method','GET').upper()
                result = request(method,action,**({'data':fields} if method=='POST' else {'params':fields}))
                if 'text/html' in result.headers.get('Content-Type','').lower() or result.content.lstrip().startswith(b'<'):
                    links = BeautifulSoup(result.text,'html.parser').find_all('a',href=True)
                    candidates = [urljoin(result.url,a['href']) for a in links if str(year) in a['href'] and any(ext in a['href'].lower() for ext in ('.nc','.grd','.bin'))]
                    if len(candidates) != 1:
                        raise RuntimeError('Form returned HTML without exactly one matching file link')
                    result = request('GET',candidates[0])
                return result
    raise RuntimeError(f'No confirmed form option for {year}')

if __name__ == '__main__':
    for kind, page_name, template in ([] if args.skip_grids else [
        ('rainfall','Rainfall_25_NetCDF.html','RF25_ind{year}_rfp25.nc'),
        ('maximum_temperature','Max_1_Bin.html','maximum_temperature_{year}.GRD'),
        ('minimum_temperature','Min_1_Bin.html','minimum_temperature_{year}.GRD'),
    ]):
        page = 'https://imdpune.gov.in/cmpg/Griddata/' + page_name
        for year in (2021,2022,2023,2024):
            acquire(raw/kind/str(year)/template.format(year=year),kind,year,page,lambda p=page,y=year:form_fetch(p,y))
    
    track_page = 'https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=MzM%3D'
    try:
        response = request('GET',track_page)
        soup = BeautifulSoup(response.text,'html.parser')
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4 and cells[1].get_text(strip=True) in ('2021','2022','2023','2024'):
                year = int(cells[1].get_text(strip=True))
                link = row.find('a',href=True)
                url = urljoin(response.url,link['href'])
                acquire(raw/'official_events'/str(year)/f'best_tracks_{year}.pdf','pdf',year,track_page,lambda u=url:request('GET',u))
        link = next(a for a in soup.find_all('a',href=True) if 'Best Tracks Data' in a.get_text())
        url = urljoin(response.url,link['href'])
        acquire(raw/'official_events'/'best_tracks_provider_archive.xlsx','xlsx',None,track_page,lambda:request('GET',url))
    except Exception as e:
        record({'product':'best_tracks_discovery','status':'failed','error':str(e)})
    
    # Exact official URLs found in web search, not inferred filenames.
    reports = {
        2021:'https://www.imdpune.gov.in/cmpg/Product/Annual_Climate_Summary/annual_summary_2021.pdf',
        2022:'https://www.imdpune.gov.in/cmpg/Product/Annual_Climate_Summary/annual_summary_2022.pdf',
        2023:'https://www.imdpune.gov.in/cmpg/Product/Annual_Climate_Summary/annual_summary_2023.pdf',
    }
    for year,url in reports.items():
        acquire(raw/'official_events'/str(year)/f'annual_climate_summary_{year}.pdf','pdf',year,url,lambda u=url:request('GET',u))
    print('Acquisition pass finished. Consult manifest for individual failures.',flush=True)
    
