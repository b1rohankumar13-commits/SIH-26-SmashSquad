"""Extract source-grade track records and narrative rows; no event or bust labels."""
from pathlib import Path
import datetime as dt, hashlib, json, math, re
import openpyxl
ROOT=Path(r'C:\Users\b1sun\OneDrive\Desktop\sih_project')
SOURCE=ROOT/'data/raw/observations/imd/official_events/best_tracks_provider_archive.xlsx'
OUT=ROOT/'data/processed/categories'

def number(value):
 try:
  v=float(value)
  return v if math.isfinite(v) else None
 except (ValueError,TypeError): return None

def date_value(value):
 if isinstance(value,dt.datetime): return value.date()
 if isinstance(value,dt.date): return value
 if value is None: return None
 for fmt in ['%d/%m/%Y','%Y-%m-%d','%Y-%m-%d %H:%M:%S','%d-%m-%Y']:
  try: return dt.datetime.strptime(str(value).strip(),fmt).date()
  except ValueError: pass
 raise ValueError(f'Unrecognized date {value!r}')

def write_records(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.part')
 tmp.write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows),encoding='utf-8')
 assert len(tmp.read_text(encoding='utf-8').splitlines())==len(rows)
 tmp.replace(path)

def main():
 book=openpyxl.load_workbook(SOURCE,read_only=True,data_only=True)
 records=[];notes=[];issues=[];annual_counts={};source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 for year in [2021,2022,2023,2024]:
  serial=None;basin=None;name=None;day=None;count=0
  for rn,row in enumerate(book[str(year)].iter_rows(min_row=2,values_only=True),2):
   if not any(v is not None for v in row): continue
   row=list(row)+[None]*14
   if row[0] is not None:
    if number(row[0]) is None:
     issues.append({'sheet':year,'row':rn,'reason':'non_numeric_system_identifier','raw':row[:14]});continue
    serial=int(float(row[0]));name=None;basin=None;day=None
   if row[1] is not None: basin=str(row[1]).strip()
   if row[2] is not None: name=str(row[2]).strip()
   if row[3] is not None:
    try: day=date_value(row[3])
    except ValueError as e: issues.append({'sheet':year,'row':rn,'reason':str(e),'raw':row[:14]});day=None
   if day is not None and day>dt.date(2024,1,10): continue
   context={'system_id':f'IMD_{year}_{serial:02d}' if serial is not None else None,'basin':basin,'system_name':name,
    'source_sheet':str(year),'source_row':rn,'source_path':str(SOURCE.relative_to(ROOT)),'source_sha256':source_hash,
    'source_url':'https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=MzM%3D',
    'date_carried_from_prior_row':row[3] is None,'role':'later_observation_evaluation_only'}
   lat,lon=number(row[5]),number(row[6])
   t=str(row[4]).strip() if row[4] is not None else ''
   if isinstance(row[4],(int,float)): t=str(int(row[4])).zfill(4)
   if re.fullmatch(r'\d{1,4}',t): t=t.zfill(4)
   if lat is None or lon is None or not re.fullmatch(r'\d{4}',t):
    notes.append({**context,'source_date':str(day) if day else None,'raw_values':row[:14],'interpretation':'Source narrative or non-coordinate row; no landfall time/location inferred.'});continue
   if day is None or serial is None or not (-90<=lat<=90 and 0<=lon<=360):
    issues.append({**context,'reason':'missing_date_system_or_invalid_coordinate','raw':row[:14]});continue
   try: timestamp=dt.datetime.combine(day,dt.time(int(t[:2]),int(t[2:]),tzinfo=dt.timezone.utc))
   except ValueError:
    issues.append({**context,'reason':'invalid_time','raw':row[:14]});continue
   pressure=number(row[8]);wind=number(row[9]);grade=str(row[11]).strip().upper() if row[11] is not None else None
   records.append({**context,'valid_time_utc':timestamp.isoformat(),'latitude':lat,'longitude':lon,
    'grade_source':grade,'central_pressure_hpa':pressure,'maximum_sustained_wind_kt':wind,
    'central_pressure_pa':pressure*100 if pressure is not None else None,
    'maximum_sustained_wind_m_s':wind*1852/3600 if wind is not None else None,
    'pressure_drop_hpa':number(row[10]),'source_time_value':row[4]})
   count+=1
  annual_counts[str(year)]=count
 book.close()
 keys={};duplicates=[]
 for r in records:
  key=(r['system_id'],r['valid_time_utc'])
  if key in keys: duplicates.append({'system_id':key[0],'time':key[1],'rows':[keys[key],r['source_row']]})
  keys[key]=r['source_row']
 cyclonic={'CS','SCS','VSCS','ESCS','SUCS','SUPER CYCLONIC STORM'}
 cyclone_ids={r['system_id'] for r in records if r['grade_source'] in cyclonic}
 depression_records=[r for r in records if r['grade_source'] in {'D','DD'}]
 cyclone_records=[r for r in records if r['system_id'] in cyclone_ids]
 write_records(OUT/'monsoon_depressions/observations/imd_depression_stage_records.jsonl',depression_records)
 write_records(OUT/'cyclones/observations/imd_cyclone_system_tracks.jsonl',cyclone_records)
 for category in ['monsoon_depressions','cyclones']:
  write_records(OUT/category/'observations/imd_track_source_narrative_rows.jsonl',notes)
 summary={'status':'source_records_normalized_no_event_labels','source_rows_by_year':annual_counts,
  'all_coordinate_records':len(records),'depression_stage_records':len(depression_records),'cyclone_system_records':len(cyclone_records),
  'narrative_rows':len(notes),'issues':issues,'duplicate_system_times_preserved':duplicates,
  'depression_selection':'Source D/DD stages only; no monsoon season/domain classification applied.',
  'cyclone_selection':'Entire observed history of source systems with at least one CS/SCS/VSCS/ESCS/SUCS stage; no forecast matching.',
  'time_handling':'Only date values are carried within source system; all original observation hours retained; no 6-hour interpolation.',
  'landfall':'Narrative rows retained verbatim with row provenance; no structured landfall inferred.'}
 p=ROOT/'data/metadata/category_preprocessing_available_v1/best_tracks.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(summary,indent=2,default=str))
 print(json.dumps({k:v for k,v in summary.items() if k not in ['issues','duplicate_system_times_preserved']}));print(json.dumps({'issues':len(issues),'duplicates':len(duplicates)}))

if __name__=='__main__': main()
