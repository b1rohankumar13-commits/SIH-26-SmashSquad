# IMD acquisition status

Checked: 2026-09-14T03:56:20.736379+00:00

Requested scope: rainfall, daily maximum/minimum temperature, cyclone/depression best tracks, and official heat-wave, active/break monsoon and western-disturbance source records matching the existing GEFS 2021-2023 initializations.

GEFS has 1,095 complete daily initialization records and leads through 240 hours. The latest forecast endpoint is 2024-01-10 00:00 UTC. Annual 2024 provider files are retained for the necessary January overlap; this does not add 2024 forecast initializations.

Acquisition complete: 12 annual gridded files and 24 official event-source files are saved. All 12 grid file hashes match their acquisition records. A separate numeric check found zero entirely missing days across these files and no unexpected nonfinite temperature values. Provider spatial missing-value masks remain intact.

## Daily grid coverage

| Product | 2021 | 2022 | 2023 | 2024 overlap source |
|---|---|---|---|---|
| rainfall | Present | Present | Present | Present |
| maximum_temperature | Present | Present | Present | Present |
| minimum_temperature | Present | Present | Present | Present |

## Official event source files

24 files are saved under data/raw/observations/imd/official_events. The JSON inventory records each filename, SHA-256, PDF page count and keyword page locations.

- data\raw\observations\imd\official_events\2021\annual_climate_summary_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\best_tracks_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\monsoon_report_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\weather_in_india_hot_weather_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\weather_in_india_monsoon_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\weather_in_india_post_monsoon_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2021\weather_in_india_winter_2021.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\annual_climate_summary_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\best_tracks_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\monsoon_report_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\weather_in_india_hot_weather_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\weather_in_india_monsoon_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\weather_in_india_post_monsoon_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2022\weather_in_india_winter_2022.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\annual_climate_summary_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\best_tracks_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\monsoon_report_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\weather_in_india_hot_weather_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\weather_in_india_monsoon_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\weather_in_india_post_monsoon_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2023\weather_in_india_winter_2023.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2024\best_tracks_2024.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\2024\weather_in_india_winter_2024.pdf — pdf_all_pages_parsed
- data\raw\observations\imd\official_events\best_tracks_provider_archive.xlsx — xlsx_zip_verified

## Validation and unresolved work

- Rainfall NetCDF validation checks exact annual dates, dimensions, coordinates, units, finite data and nonnegative rainfall. The existing 2024 file covers all 366 days. Native rainfall grid: 0.25 degrees, 66.5–100 E and 6.5–38.5 N; its missing mask must be retained.
- IMD temperature documentation specifies 31 x 31 grid points at 1 degree, 67.5–97.5 E and 7.5–37.5 N, Celsius, and missing value 99.9. Binary length is checked against the calendar; byte-order diagnostics are recorded without silently choosing a decoder.
- Native IMD grids do not exactly coincide with the GEFS target grid or cover its whole domain. No extrapolation, gap filling, regridding or replacement analysis source has been selected.
- Daily observation-window alignment is not yet performed. IMD rainfall uses a 24-hour period ending 03 UTC; the existing pilot approximation must not automatically be applied to GEFS.
- Event PDFs and the provider best-track workbook are raw source material. Keyword hits are not an event catalogue. Event-date extraction and category-specific scientific validation remain separate work.
- The best-track workbook contains sheets for 2021, 2022, 2023 and 2024, with date/time, position, central pressure, wind speed and grade columns. Some repeated metadata are blank/merged, and the 2021 sheet has a large formatted row extent; do not treat worksheet dimensions as observation counts. The 2021 track PDF is image based and was visually checked.
- Failed downloads and their errors are preserved in data/metadata/imd_acquisition_2021_2023.jsonl and data/metadata/imd_event_reports_2021_2023.jsonl. IMD Pune intermittently times out during connection/TLS setup; RSMC and MAUSAM downloads were reachable.
- Existing files were preserved. No forecast data or retained raw data were deleted. No credentials, paid sources or substitute datasets were used.

## Re-running acquisition

Run the saved scripts with the project virtual environment and an explicit --project-root. acquire_imd_confirmed.py discovers official forms; retry_imd_grids.py retries the exact previously verified form actions; acquire_imd_event_reports.py retrieves the seasonal source reports. Each preserves existing validated files.
