"""Retry only missing grids using previously inspected official form requests."""
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import acquire_imd_confirmed as core

lock = threading.Lock()
original_record = core.record
def safe_record(item):
    with lock:
        original_record(item)
core.record = safe_record

# Actions and field names were read from IMD's official forms in this session.
jobs = [
    ('rainfall','Rainfall_25_NetCDF.html','RF25.php','RF25','RF25_ind{year}_rfp25.nc'),
    ('maximum_temperature','Max_1_Bin.html','maxtemp.php','maxtemp','maximum_temperature_{year}.GRD'),
    ('minimum_temperature','Min_1_Bin.html','mintemp.php','mintemp','minimum_temperature_{year}.GRD'),
]
def run(job):
    kind,page,action,field,name,year = job
    source = 'https://imdpune.gov.in/cmpg/Griddata/' + page
    def fetch():
        response = requests.post('https://imdpune.gov.in/cmpg/Griddata/'+action,data={field:str(year)},timeout=(45,180),headers={'User-Agent':core.session.headers['User-Agent'],'Referer':source})
        response.raise_for_status()
        host = core.urlparse(response.url).hostname or ''
        if not (host == 'imdpune.gov.in' or host.endswith('.imdpune.gov.in') or host.endswith('.imd.gov.in')):
            raise ValueError('Non-IMD redirect')
        return response
    core.acquire(core.raw/kind/str(year)/name.format(year=year),kind,year,source,fetch)

work = [(*job,year) for job in jobs for year in (2021,2022,2023,2024)]
work = [job for job in work if not (core.raw/job[0]/str(job[-1])/job[4].format(year=job[-1])).exists()]
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run,work))
print('Bounded grid retry finished.',flush=True)
