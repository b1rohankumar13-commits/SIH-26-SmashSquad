"""Category-specific GEFS/IMD preprocessing; available inputs only, no labels."""
from __future__ import annotations
import argparse, calendar, concurrent.futures as cf, datetime as dt, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
import eccodes as ec

ROOT=Path(r'C:\Users\b1sun\OneDrive\Desktop\sih_project')
sys.path.insert(0,str(ROOT))
from src.preprocessing.align_30_day_rainfall import _overlap_weights, _cell_bounds
BASE=ROOT/'data/interim/forecasts/gefs/noaa_pds_2021_2023'
SUPP=ROOT/'data/interim/forecasts/gefs/category_supplement_2021_2023'
OUT=ROOT/'data/processed/categories'
META=ROOT/'data/metadata/category_preprocessing_available_v1'
MEMBERS=['gec00']+[f'gep{i:02d}' for i in range(1,11)]
LAT=np.arange(37.5,5,-1); LON=np.arange(65.5,100,1)
SPEC={
 'heavy_rainfall':['rainfall_24h','mslp','u850','v850','q850','height500'],
 'monsoon_depressions':['rainfall_24h','mslp','surface_pressure','u850','v850','u700','v700','q850','q700','height500'],
 'cyclones':['rainfall_24h','mslp','u10','v10','u850','v850','u500','v500','height500'],
 'heat_waves':['temperature_2m_6hourly','tmax_6hour_snapshot_approximation','tmin_6hour_snapshot_approximation','rh2m','u10','v10','soil_water_0_10cm','surface_pressure','total_cloud_cover'],
 'western_disturbances':['height500','u500','v500','u300','v300','temperature500','q700','mslp','rainfall_24h'],
 'active_break_monsoon':['rainfall_24h','u850','v850','q850','mslp','height500','total_cloud_cover'],
}
PRIMARY={c:['rainfall_24h'] for c in SPEC}
PRIMARY['heat_waves']=['tmax_6hour_snapshot_approximation','tmin_6hour_snapshot_approximation']
PRIMARY['western_disturbances']=['height500']
MAP={('prmsl','meanSea',0):'mslp',('sp','surface',0):'surface_pressure',
 ('10u','heightAboveGround',10):'u10',('10v','heightAboveGround',10):'v10',
 ('gh','isobaricInhPa',500):'height500',('t','isobaricInhPa',500):'temperature500',
 ('2r','heightAboveGround',2):'rh2m',('soilw','depthBelowLandLayer',0):'soil_water_0_10cm',
 ('tcc','atmosphere',0):'total_cloud_cover'}
for lev in [300,500,700,850]:
 for v in ['u','v']: MAP[(v,'isobaricInhPa',lev)]=f'{v}{lev}'
for lev in [700,850]: MAP[('q','isobaricInhPa',lev)]=f'q{lev}'
UNITS={v:'m s-1' for v in MAP.values() if v.startswith(('u','v'))}
UNITS.update(mslp='Pa',surface_pressure='Pa',height500='m',temperature500='degC',q700='kg kg-1',q850='kg kg-1',
 rainfall_24h='mm',temperature_2m_6hourly='degC',tmax_6hour_snapshot_approximation='degC',tmin_6hour_snapshot_approximation='degC',
 rh2m='%',soil_water_0_10cm='m3 m-3',total_cloud_cover='%')

def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def save_nc(ds,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.nc.part')
 enc={n:{'zlib':True,'complevel':1,'shuffle':True,'dtype':'float32','_FillValue':np.float32(np.nan)} for n,v in ds.data_vars.items() if v.dtype.kind=='f'}
 ds.to_netcdf(tmp,engine='netcdf4',encoding=enc)
 with xr.open_dataset(tmp,engine='netcdf4') as check:
  assert dict(check.sizes)==dict(ds.sizes),(path,check.sizes,ds.sizes)
  assert set(check.data_vars)==set(ds.data_vars),path
  for name in ds.data_vars:
   original=ds[name].values; actual=check[name].values
   if original.dtype.kind=='f': assert np.allclose(actual,original,rtol=1e-6,atol=1e-6,equal_nan=True),(path,name)
 tmp.replace(path)
 return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)}

def process_date(date,supplement_dates):
 started=time.monotonic(); token=date.replace('-',''); init=np.datetime64(date,'ns')
 checkpoint=META/'dates'/f'{token}.json'
 if checkpoint.exists():
  prior=json.loads(checkpoint.read_text())
  if all((ROOT/r['path']).exists() and (ROOT/r['path']).stat().st_size==r['bytes'] and sha(ROOT/r['path'])==r['sha256'] for r in prior['outputs']):
   return {'date':date,'status':'already_complete'}
  raise RuntimeError(f'Previously completed output changed for {date}; preserve and inspect.')
 names=sorted(set(sum(SPEC.values(),[])))
 data={n:np.full((10,11,33,35),np.nan,dtype='float32') for n in names if n!='temperature_2m_6hourly'}
 subtemp=np.full((40,11,33,35),np.nan,dtype='float32');rain=np.full_like(subtemp,np.nan)
 present={n:np.zeros((10,11),dtype='uint8') for n in data}
 subpresence=np.zeros((2,40,11),dtype='uint8'); sources=[];missing=[]
 for mi,member in enumerate(MEMBERS):
  for lead in range(6,241,6):
   files=[(BASE/token[:4]/token/member/f'f{lead:03d}_pgrb2a.grib2',False)]
   if lead%24==0:
    files.append((BASE/token[:4]/token/member/f'f{lead:03d}_pgrb2b.grib2',False))
    if token in supplement_dates: files.append((SUPP/token[:4]/token/member/f'f{lead:03d}_pgrb2a.grib2',True))
   for path,is_supplement in files:
    if not path.exists(): missing.append(str(path.relative_to(ROOT)));continue
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size})
    with path.open('rb') as f:
     first=True
     while (h:=ec.codes_grib_new_from_file(f)) is not None:
      try:
       short=ec.codes_get(h,'shortName');leveltype=ec.codes_get(h,'typeOfLevel');level=ec.codes_get(h,'level')
       if first:
        assert ec.codes_get(h,'dataDate')==int(token) and ec.codes_get(h,'dataTime')==0,path
        assert ec.codes_get(h,'Ni')==35 and ec.codes_get(h,'Nj')==33,path
        assert ec.codes_get(h,'latitudeOfFirstGridPointInDegrees')==37.5 and ec.codes_get(h,'longitudeOfFirstGridPointInDegrees')==65.5,path
        assert ec.codes_get(h,'iDirectionIncrementInDegrees')==1 and ec.codes_get(h,'jDirectionIncrementInDegrees')==1,path
        first=False
       n=MAP.get((short,leveltype,level))
       if short not in ['tp','2t'] and n is None: continue
       start,end=ec.codes_get(h,'startStep'),ec.codes_get(h,'endStep')
       assert end==lead,(path,short,start,end)
       values=np.asarray(ec.codes_get_values(h),dtype='float32').reshape(33,35)
       missing_value=ec.codes_get(h,'missingValue');values[values==missing_value]=np.nan
       if short=='tp':
        assert start==lead-6 and ec.codes_get(h,'stepType')=='accum',(path,start,end)
        assert ec.codes_get(h,'units')=='kg m**-2',path
        rain[lead//6-1,mi]=values;subpresence[0,lead//6-1,mi]=1
       elif short=='2t':
        assert start==end and ec.codes_get(h,'units')=='K',path
        subtemp[lead//6-1,mi]=values-273.15;subpresence[1,lead//6-1,mi]=1
       else:
        assert lead%24==0,(path,n)
        if n=='total_cloud_cover': assert start==lead-6 and ec.codes_get(h,'stepType')=='avg',(path,start,end)
        else: assert start==end,(path,n,start,end)
        if n=='temperature500': values=values-273.15
        data[n][lead//24-1,mi]=values;present[n][lead//24-1,mi]=1
      finally: ec.codes_release(h)
 rf=rain.reshape(10,4,11,33,35); tf=subtemp.reshape(10,4,11,33,35)
 data['rainfall_24h']=np.sum(rf,axis=1)
 data['tmax_6hour_snapshot_approximation']=np.max(tf,axis=1)
 data['tmin_6hour_snapshot_approximation']=np.min(tf,axis=1)
 present['rainfall_24h']=np.min(subpresence[0].reshape(10,4,11),axis=1)
 for n in PRIMARY['heat_waves']: present[n]=np.min(subpresence[1].reshape(10,4,11),axis=1)
 lead_hours=np.arange(24,241,24)
 coords={'lead_day':np.arange(1,11),'member':MEMBERS,'latitude':LAT,'longitude':LON,
 'init_time':init,'lead_hour':('lead_day',lead_hours),'valid_time':('lead_day',init+lead_hours.astype('timedelta64[h]'))}
 outputs=[];quality={}
 for category,selected in SPEC.items():
  ds=xr.Dataset(coords=coords,attrs={'title':f'GEFS available-input preprocessing: {category}',
   'pipeline_version':'available_v1','forecast_source':'NOAA archived operational GEFS noaa-gefs-pds',
   'forecast_input_role':'forecast_issued_at_init_time','observation_join_status':'not_joined_time_alignment_pending',
   'climatology_status':'deferred_user_years_unspecified','labels_created':'none',
   'supplement_included':int(token in supplement_dates),'source_manifest':str(checkpoint.relative_to(ROOT)),
   'grid_method':'existing native point selection at approved 1-degree centres; no additional forecast regridding'})
  for n in selected:
   if n=='temperature_2m_6hourly':
    ds[n]=(('six_hour_lead', 'member','latitude','longitude'),subtemp)
    ds=ds.assign_coords(six_hour_lead=np.arange(6,241,6),six_hour_valid_time=('six_hour_lead',init+np.arange(6,241,6).astype('timedelta64[h]')))
    ds['temperature_2m_source_present']=(('six_hour_lead','member'),subpresence[1])
   else:
    ds[n]=(('lead_day','member','latitude','longitude'),data[n])
    ds[n+'_source_present']=(('lead_day','member'),present[n])
   ds[n].attrs={'units':UNITS[n]}
   if 'snapshot_approximation' in n: ds[n].attrs['method']='Extremum of four samples at 06,12,18,24 hours in the forecast day; any missing sample gives NaN. Not a continuous daily extremum.'
   if n=='rainfall_24h': ds[n].attrs['method']='Sum of four incremental 6-hour totals; missing sample gives NaN; 00-00 UTC window.'
   if n=='total_cloud_cover': ds[n].attrs['method']='Original final-six-hour mean at daily lead; not a daily mean.'
   if n=='soil_water_0_10cm': ds[n].attrs['layer']='0 to 0.1 m below ground; missing ocean cells remain NaN.'
  ds['day_window_start']=('lead_day',init+(lead_hours-24).astype('timedelta64[h]'))
  ds['day_window_end']=('lead_day',init+lead_hours.astype('timedelta64[h]'))
  if 'total_cloud_cover' in selected:
   ds['cloud_window_start']=('lead_day',init+(lead_hours-6).astype('timedelta64[h]'))
   ds['cloud_window_end']=('lead_day',init+lead_hours.astype('timedelta64[h]'))
  for n in PRIMARY[category]:
   a=data[n];valid=np.isfinite(a);count=valid.sum(axis=1)
   # Descriptive statistics for the retained ensemble, with strict completeness.
   mean=np.sum(np.where(valid,a,0),axis=1,dtype='float64')/11
   spread=np.sqrt(np.sum(np.where(valid,(a-mean[:,None])**2,0),axis=1,dtype='float64')/11)
   mean[count!=11]=np.nan;spread[count!=11]=np.nan
   for suffix,values in [('ensemble_mean',mean),('ensemble_spread',spread)]:
    ds[n+'_'+suffix]=(('lead_day','latitude','longitude'),values.astype('float32'))
    ds[n+'_'+suffix].attrs={'units':UNITS[n],'method':'All 11 retained members required; spread is population standard deviation (ddof=0).'}
   ds[n+'_valid_member_count']=(('lead_day','latitude','longitude'),count.astype('uint8'))
  ds.latitude.attrs={'units':'degrees_north'};ds.longitude.attrs={'units':'degrees_east'}
  output=save_nc(ds,OUT/category/'forecast'/token[:4]/f'gefs_{token}_00_day01-10.nc')
  outputs.append(output)
  quality[category]={n:{'finite_values':int(np.isfinite(ds[n].values).sum()),'total_values':int(ds[n].size)} for n in selected}
  ds.close()
 result={'date':date,'status':'complete_available_inputs','outputs':outputs,'source_files':sources,'missing_source_files':missing,
 'supplement_available_at_snapshot':token in supplement_dates,'quality':quality,'elapsed_seconds':round(time.monotonic()-started,2)}
 atomic_json(checkpoint,result)
 return {k:result[k] for k in ['date','status','elapsed_seconds','missing_source_files','supplement_available_at_snapshot']}

def observations():
 status=json.loads((ROOT/'data/metadata/imd_acquisition_status.json').read_text())
 outputs=[]
 for item in status['grid_files']:
  year=item['year'];product=item['product'];source=ROOT/item['local_path']
  if product=='rainfall':
   with xr.open_dataset(source) as raw:
    field=raw.RAINFALL.rename({'TIME':'date','LATITUDE':'latitude','LONGITUDE':'longitude'}).sortby('latitude').sortby('longitude').load()
   field=field.sel(date=slice('2021-01-01','2024-01-10'))
   values=field.values.astype('float64');valid=np.isfinite(values)&(values>=0)
   wy=_overlap_weights(field.latitude.values,LAT[::-1],latitude=True);wx=_overlap_weights(field.longitude.values,LON,latitude=False)
   numerator=np.einsum('ai,tij,bj->tab',wy,np.where(valid,values,0),wx,optimize=True)
   denom=np.einsum('ai,tij,bj->tab',wy,valid.astype('float64'),wx,optimize=True)
   bounds_y=_cell_bounds(LAT[::-1]);bounds_x=_cell_bounds(LON)
   area=np.diff(np.sin(np.deg2rad(bounds_y)))[:,None]*np.diff(np.deg2rad(bounds_x))[None,:]
   mean=np.divide(numerator,denom,out=np.full_like(numerator,np.nan),where=denom>0)
   coverage=denom/area[None]
   assert np.nanmax(coverage)<1.000001 and np.nanmin(coverage)>=0
   ds=xr.Dataset({'rainfall':(('date','latitude','longitude'),mean[:,::-1].astype('float32')),
    'valid_source_area_fraction':(('date','latitude','longitude'),coverage[:,::-1].astype('float32'))},coords={'date':field.date.values,'latitude':LAT,'longitude':LON})
   ds.rainfall.attrs={'units':'mm','method':'Spherical area-weighted mean over finite nonnegative source rainfall only. Coverage records the observed source area fraction of each target cell; no ocean filling.'}
   ds['window_end_utc']=('date',field.date.values+np.timedelta64(3,'h'));ds['window_start_utc']=('date',field.date.values-np.timedelta64(21,'h'))
   categories=[c for c in SPEC if c!='heat_waves']
  else:
   n='tmax' if product=='maximum_temperature' else 'tmin'
   ndays=366 if calendar.isleap(year) else 365
   values=np.fromfile(source,dtype='<f4').reshape(ndays,31,31)
   values[np.isclose(values,99.9,rtol=0,atol=0.0001)]=np.nan
   dates=np.arange(np.datetime64(f'{year}-01-01'),np.datetime64(f'{year+1}-01-01'))
   take=dates<=np.datetime64('2024-01-10')
   ds=xr.Dataset({n:(('date','latitude','longitude'),values[take])},coords={'date':dates[take],'latitude':np.arange(7.5,38,1),'longitude':np.arange(67.5,98,1)})
   ds[n].attrs={'units':'degC','method':'Native IMD 1-degree binary decoded little-endian float32; 99.9 masked. No interpolation.'}
   ds.attrs['temperature_window_status']='Provider daily date preserved; no subdaily window or GEFS alignment inferred.'
   categories=['heat_waves','western_disturbances']
  ds.attrs.update(source=str(source.relative_to(ROOT)),source_sha256=sha(source),role='later_observation_evaluation_only',forecast_join_status='not_joined',labels_created='none')
  name='rainfall' if product=='rainfall' else n
  for category in categories:
   outputs.append(save_nc(ds,OUT/category/'observations'/f'imd_{name}_{year}.nc'))
  ds.close()
 atomic_json(META/'observations.json',{'status':'complete_available_grids','outputs':outputs})
 return len(outputs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--observations',action='store_true')
 a=p.parse_args();META.mkdir(parents=True,exist_ok=True)
 assert 1<=a.workers<=4
 # Freeze supplement membership once; no waiting for or joining partially downloaded dates.
 snapshot=META/'input_snapshot.json'
 if snapshot.exists(): frozen=json.loads(snapshot.read_text())
 else:
  cp=ROOT/'data/metadata/gefs_category_supplement_completed_dates.jsonl'
  complete=set()
  if cp.exists():
   for line in cp.read_text().splitlines():
    try:
     row=json.loads(line)
     if row.get('status')=='complete': complete.add(row['date'])
    except json.JSONDecodeError: pass
  frozen={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'supplement_complete_dates':sorted(complete),'scope':'Available data only, as requested. Later downloads excluded until an explicit refresh.','category_fields':SPEC}
  atomic_json(snapshot,frozen)
 if a.observations: print(json.dumps({'event':'observations_complete','files':observations()}),flush=True)
 start,end=dt.date.fromisoformat(a.start),dt.date.fromisoformat(a.end)
 assert dt.date(2021,1,1)<=start<=end<=dt.date(2023,12,31)
 dates=[(start+dt.timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
 # 45 GiB bounds uncompressed daily category fields + stats for the full scope.
 def tree_bytes(directory):
  total=0
  with os.scandir(directory) as entries:
   for entry in entries:
    if entry.is_dir(follow_symlinks=False): total+=tree_bytes(entry.path)
    elif entry.is_file(follow_symlinks=False): total+=entry.stat().st_size
  return total
 print(json.dumps({'event':'storage_audit_started'}),flush=True)
 retained=tree_bytes(ROOT/'data');reserve=45*1024**3
 cfg=yaml.safe_load((ROOT/'configs/acquisition_gefs_2021_2023.yaml').read_text())
 ceiling=cfg['limits']['storage_ceiling_gb']*1024**3*cfg['limits']['stop_at_fraction']
 if shutil.disk_usage(ROOT).free<reserve or retained+reserve>ceiling: raise RuntimeError('Full preprocessing reserve would exceed free space or approved storage ceiling.')
 print(json.dumps({'event':'forecast_preprocessing_started','requested_dates':len(dates),'workers':a.workers,'retained_bytes':retained,'reserve_bytes':reserve}),flush=True)
 failures=[];done=0
 with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(process_date,d,frozen['supplement_complete_dates']):d for d in dates}
  for future in cf.as_completed(futures):
   try: result=future.result();done+=1;print(json.dumps(result),flush=True)
   except Exception as error:
    failures.append({'date':futures[future],'error':repr(error)});print(json.dumps(failures[-1]),flush=True)
 summary={'start':a.start,'end':a.end,'completed':done,'requested':len(dates),'failures':failures,'status':'complete_available_inputs' if not failures else 'incomplete','labels_created':False}
 atomic_json(META/f'run_{a.start}_{a.end}.json',summary);print(json.dumps(summary),flush=True)
 if failures: raise SystemExit(1)

if __name__=='__main__': main()
