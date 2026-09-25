from pathlib import Path
import json
import numpy as np
import xarray as xr
import eccodes as ec
ROOT=Path(r'C:\Users\b1sun\OneDrive\Desktop\sih_project')
OUT=ROOT/'data/processed/categories'
def field(file,short):
 with file.open('rb') as f:
  while (h:=ec.codes_grib_new_from_file(f)) is not None:
   try:
    if ec.codes_get(h,'shortName')==short:
     v=ec.codes_get_values(h).reshape(33,35);v[v==ec.codes_get(h,'missingValue')]=np.nan
     return v
   finally:ec.codes_release(h)
 raise ValueError(short)
base=ROOT/'data/interim/forecasts/gefs/noaa_pds_2021_2023/2021/20210101/gec00'
rain=sum(field(base/f'f{lead:03d}_pgrb2a.grib2','tp') for lead in [6,12,18,24])
temps=np.stack([field(base/f'f{lead:03d}_pgrb2a.grib2','2t')-273.15 for lead in [6,12,18,24]])
with xr.open_dataset(OUT/'heavy_rainfall/forecast/2021/gefs_20210101_00_day01-10.nc') as ds:
 np.testing.assert_allclose(ds.rainfall_24h.isel(lead_day=0,member=0),rain,rtol=1e-6,atol=1e-5)
 np.testing.assert_allclose(ds.rainfall_24h_ensemble_mean,ds.rainfall_24h.mean('member'),atol=2e-5)
 np.testing.assert_allclose(ds.rainfall_24h_ensemble_spread,ds.rainfall_24h.std('member',ddof=0),atol=2e-5)
 assert ds.valid_time.values[-1]==np.datetime64('2021-01-11')
with xr.open_dataset(OUT/'heat_waves/forecast/2021/gefs_20210101_00_day01-10.nc') as ds:
 np.testing.assert_allclose(ds.tmax_6hour_snapshot_approximation.isel(lead_day=0,member=0),temps.max(0),atol=3e-5)
 np.testing.assert_allclose(ds.tmin_6hour_snapshot_approximation.isel(lead_day=0,member=0),temps.min(0),atol=3e-5)
 assert int(ds.soil_water_0_10cm.isel(lead_day=0,member=0).isnull().sum())>0
 assert float(ds.soil_water_0_10cm.max())<2
 assert ds.cloud_window_end.values[0]-ds.cloud_window_start.values[0]==np.timedelta64(6,'h')
with xr.open_dataset(OUT/'heavy_rainfall/observations/imd_rainfall_2021.nc') as ds:
 assert ds.rainfall.where(ds.valid_source_area_fraction==0).isnull().all()
 assert ds.valid_source_area_fraction.min()>=0 and ds.valid_source_area_fraction.max()<=1
 # Independently verify the first day's area mean for one complete 1-degree cell.
 src=ROOT/'data/raw/observations/imd/rainfall/2021/RF25_ind2021_rfp25.nc'
 with xr.open_dataset(src) as raw:
  s=raw.RAINFALL.isel(TIME=0)
  y=22.5;x=78.5
  lat=raw.LATITUDE.values;lon=raw.LONGITUDE.values
  yl=np.maximum(lat-.125,y-.5);yu=np.minimum(lat+.125,y+.5)
  xl=np.maximum(lon-.125,x-.5);xu=np.minimum(lon+.125,x+.5)
  wy=np.where(yu>yl,np.sin(np.deg2rad(yu))-np.sin(np.deg2rad(yl)),0)
  wx=np.maximum(xu-xl,0)*np.pi/180
  weight=wy[:,None]*wx[None,:];ok=np.isfinite(s.values)&(s.values>=0)
  expected=(np.where(ok,s.values,0)*weight).sum()/(ok*weight).sum()
  np.testing.assert_allclose(ds.rainfall.sel(latitude=y,longitude=x).isel(date=0),expected,atol=1e-5)
for n in ['tmax','tmin']:
 with xr.open_dataset(OUT/f'heat_waves/observations/imd_{n}_2024.nc') as ds:
  assert ds.sizes['date']==10 and ds.sizes['latitude']==31
  assert not np.isclose(ds[n].values,99.9,atol=0.0001,equal_nan=False).any()
print(json.dumps({'status':'passed','checks':['independent_6h_rain_sum','ensemble_mean_and_population_spread','snapshot_extrema_Celsius','soil_missing_mask','cloud_6h_window','zero_coverage_missing_rain','independent_spherical_rain_regrid','January_2024_overlap','native_temperature_missing_code']}))
