# Category input acquisition status

The user requested category-specific preprocessing and subsequently chose to acquire missing inputs first. Bulk preprocessing and bust labelling have not been performed.

## Selected additions

GEFS: 2-m relative humidity, 0–10 cm volumetric soil water, and total cloud cover, for the same 2021–2023 daily 00 UTC runs, 11 members and ten daily leads as the existing archive. Relative humidity and soil moisture are instantaneous. Cloud is the native ending six-hour average; it must not be described as a full daily mean. The moisture and cloud alternatives already selected avoid downloading redundant dew point, total-column water or OLR.

New retained regional files: `data/interim/forecasts/gefs/category_supplement_2021_2023/YYYY/YYYYMMDD/member/fNNN_pgrb2a.grib2`.

Source ranges and hashes: `data/metadata/gefs_category_supplement_source_manifest.jsonl`. Regional hashes: `data/metadata/gefs_category_supplement_regional_manifest.jsonl`. Per-message checks: `data/metadata/gefs_category_supplement_qc/`. Completed ensemble dates: `data/metadata/gefs_category_supplement_completed_dates.jsonl`. A separate checkpoint records control-only samples; it never marks a full ensemble date complete.

ERA5: requests are prepared in `data/metadata/era5_category_requests.json`, but have not been submitted. Only selected winds at 300/500/700/850 hPa, height and temperature at 500 hPa, humidity at 700/850 hPa, and sea-level/surface pressure are included. Matching daily 00 UTC valid dates run from 2021-01-02 through 2024-01-10. The published IMDAA coverage ends in 2020, so the SOP's ERA5 fallback applies: https://rds.ncmrwf.gov.in/about.

## Inputs still needed from the user

- Path to a local Copernicus CDS credential configuration, or completion of CDS access setup. The existing TIGGE endpoint is different and its configuration is preserved. Do not paste tokens into chat. Dataset terms must be accepted through CDS: https://cds.climate.copernicus.eu/how-to-api.
- Observation climatology years and forecast training years, before requesting climatology data or calculating anomalies.
- Rainfall-window alignment decision before joining forecast and IMD daily rainfall. Existing forecasts use 00–00 UTC windows; IMD uses a different endpoint.

GPM IMERG Final ocean/coastal sensitivity data are an additional SOP supporting input, not part of the completed IMD land archive. Their acquisition is not yet complete. Weather reports remain contextual sources; objective western-disturbance and storm tracking definitions belong to subsequent processing. No event flag or bust definition is silently selected.

Category variable selections and proposed output locations are recorded in `configs/category_preprocessing_selection.yaml`. Existing raw and retained GEFS/IMD files are preserved.
