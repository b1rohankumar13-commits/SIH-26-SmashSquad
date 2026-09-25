# SIH Project — Complete File and Folder Structure

**Project root:** `C:\Users\b1sun\OneDrive\Desktop\sih_project`

This tree records all project-owned files and reserved directories present on 2 September 2026. Generated dependency, Git, test-cache, and Python bytecode internals are identified but deliberately not expanded because they can contain thousands of non-project files.

```text
sih_project/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── SIH_PROJECT_FILE_STRUCTURE.md
│
├── .git/                                      [generated Git metadata; contents omitted]
├── .pytest_cache/                             [generated test cache; contents omitted]
├── .venv/                                     [generated Python environment; contents omitted]
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│       ├── __init__.py
│       ├── forecast.py
│       ├── historical.py
│       └── regions.py
│
├── configs/
│   ├── bust_thresholds.yaml
│   ├── convlstm.yaml
│   ├── fusion.yaml
│   ├── grid.yaml
│   ├── sources.yaml
│   ├── variables.yaml
│   └── xgboost.yaml
│
├── dashboard/
│   ├── app.py
│   ├── assets/
│   │   └── boundaries/                        [reserved directory]
│   ├── boundaries/
│   │   └── README.md
│   ├── components/
│   │   ├── explanation_panel.py
│   │   ├── india_map.py
│   │   ├── lead_selector.py
│   │   └── probability_panel.py
│   └── pages/
│       ├── current_forecast.py
│       ├── historical_replay.py
│       └── regional_details.py
│
├── data/
│   ├── README.md
│   │
│   ├── raw/
│   │   ├── forecasts/
│   │   │   ├── gefs_operational/              [reserved directory]
│   │   │   ├── gefs_reforecast/               [reserved directory]
│   │   │   ├── ncmrwf/                        [reserved directory]
│   │   │   └── tigge/
│   │   │       ├── ncep/
│   │   │       │   └── 2024/
│   │   │       │       └── 07/
│   │   │       │           └── 01/            [reserved directory]
│   │   │       └── ncmrwf/
│   │   │           ├── 2019/
│   │   │           │   ├── _bulk/             [reserved directory]
│   │   │           │   ├── _incomplete/
│   │   │           │   │   └── tigge_ncmrwf_20190719_00_pressure_pf_uvqgh-p500-p850.grib2.individual.part
│   │   │           │   └── 07/
│   │   │           │       ├── 15/
│   │   │           │       │   ├── 00/
│   │   │           │       │   │   ├── tigge_ncmrwf_20190715_00_pressure_cf_f012-f240_12h_uvqgh_p500-p850.grib2
│   │   │           │       │   │   ├── tigge_ncmrwf_20190715_00_pressure_cf_uvqgh-p500-p850.grib2
│   │   │           │       │   │   ├── tigge_ncmrwf_20190715_00_pressure_pf_m01-m11_f012-f240_12h_uvqgh_p500-p850.grib2
│   │   │           │       │   │   ├── tigge_ncmrwf_20190715_00_pressure_pf_uvqgh-p500-p850.grib2
│   │   │           │       │   │   ├── tigge_ncmrwf_20190715_00_surface_cf_tp-mslp.grib2
│   │   │           │       │   │   └── tigge_ncmrwf_20190715_00_surface_pf_tp-mslp.grib2
│   │   │           │       │   └── 12/
│   │   │           │       │       ├── tigge_ncmrwf_20190715_12_pressure_cf_f012-f240_12h_uvqgh_p500-p850.grib2
│   │   │           │       │       └── tigge_ncmrwf_20190715_12_pressure_pf_m01-m11_f012-f240_12h_uvqgh_p500-p850.grib2
│   │   │           │       ├── 16/
│   │   │           │       │   └── 00/
│   │   │           │       │       ├── tigge_ncmrwf_20190716_00_pressure_cf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190716_00_pressure_pf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190716_00_surface_cf_tp-mslp.grib2
│   │   │           │       │       └── tigge_ncmrwf_20190716_00_surface_pf_tp-mslp.grib2
│   │   │           │       ├── 17/
│   │   │           │       │   └── 00/
│   │   │           │       │       ├── tigge_ncmrwf_20190717_00_pressure_cf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190717_00_pressure_pf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190717_00_surface_cf_tp-mslp.grib2
│   │   │           │       │       └── tigge_ncmrwf_20190717_00_surface_pf_tp-mslp.grib2
│   │   │           │       ├── 18/
│   │   │           │       │   └── 00/
│   │   │           │       │       ├── tigge_ncmrwf_20190718_00_pressure_cf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190718_00_pressure_pf_uvqgh-p500-p850.grib2
│   │   │           │       │       ├── tigge_ncmrwf_20190718_00_surface_cf_tp-mslp.grib2
│   │   │           │       │       └── tigge_ncmrwf_20190718_00_surface_pf_tp-mslp.grib2
│   │   │           │       └── 19/
│   │   │           │           └── 00/
│   │   │           │               ├── tigge_ncmrwf_20190719_00_pressure_cf_uvqgh-p500-p850.grib2
│   │   │           │               ├── tigge_ncmrwf_20190719_00_surface_cf_tp-mslp.grib2
│   │   │           │               └── tigge_ncmrwf_20190719_00_surface_pf_tp-mslp.grib2
│   │   │           └── 2024/
│   │   │               └── 07/
│   │   │                   └── 01/
│   │   │                       └── tigge_ncmrwf_20240701_00_tp_ensemble_global.grib2
│   │   │
│   │   ├── observations/
│   │   │   ├── era5/                           [reserved directory]
│   │   │   ├── gpm/                            [reserved directory]
│   │   │   ├── imdaa/                          [reserved directory]
│   │   │   └── imd/
│   │   │       ├── temperature/                [reserved directory]
│   │   │       └── rainfall/
│   │   │           ├── 2019/
│   │   │           │   └── RF25_ind2019_rfp25.nc
│   │   │           └── 2024/
│   │   │               └── RF25_ind2024_rfp25.nc
│   │   │
│   │   └── official_events/
│   │       ├── cyclone_bulletins/              [reserved directory]
│   │       ├── daily_weather_reports/          [reserved directory]
│   │       ├── heatwave_reports/               [reserved directory]
│   │       ├── monsoon_reports/                [reserved directory]
│   │       └── rsmc_best_tracks/               [reserved directory]
│   │
│   ├── interim/
│   │   ├── aligned/
│   │   │   └── rainfall/
│   │   │       ├── 2019/
│   │   │       │   └── 07/
│   │   │       │       └── 15/
│   │   │       │           └── ncmrwf_imd_20190715_00_common0p5_day01-day09.nc
│   │   │       └── 2024/
│   │   │           └── 07/
│   │   │               └── 01/
│   │   │                   └── ncmrwf_imd_20240701_00_imd_window_day01_day09.nc
│   │   ├── decoded/
│   │   │   ├── imd/
│   │   │   │   └── rainfall/
│   │   │   │       ├── 2019/
│   │   │   │       │   └── imd_rainfall_20190716_20190823.nc
│   │   │   │       └── 2024/
│   │   │   │           └── imd_rainfall_20240702_20240711.nc
│   │   │   └── tigge/
│   │   │       └── ncmrwf/
│   │   │           ├── 2019/
│   │   │           │   └── 07/
│   │   │           │       └── 15/
│   │   │           │           ├── tigge_ncmrwf_20190715_00_pressure_ensemble_day01-day10.nc
│   │   │           │           └── tigge_ncmrwf_20190715_00_surface_ensemble_day01-day10.nc
│   │   │           └── 2024/
│   │   │               └── 07/
│   │   │                   └── 01/
│   │   │                       └── tigge_ncmrwf_20240701_00_daily_rainfall_india.nc
│   │   ├── daily_accumulations/                [reserved directory]
│   │   ├── detected_weather_objects/           [reserved directory]
│   │   ├── india_subset/                       [reserved directory]
│   │   ├── region_masks/                       [reserved directory]
│   │   ├── regridded/                          [reserved directory]
│   │   └── time_aligned/                       [reserved directory]
│   │
│   ├── metadata/
│   │   ├── acquisition_log.jsonl
│   │   ├── checksums.csv
│   │   ├── processing_history.jsonl
│   │   ├── source_manifest.csv
│   │   └── variable_dictionary.yaml
│   │
│   └── processed/
│       ├── errors/
│       │   └── rainfall/
│       │       ├── 2019/
│       │       │   └── 07/
│       │       │       └── 15/
│       │       │           └── rainfall_grid_metrics_20190715_00.nc
│       │       └── 2024/
│       │           └── 07/
│       │               └── 01/
│       │                   └── rainfall_grid_metrics_20240701_00.nc
│       ├── event_catalogue/
│       │   └── rainfall_events_20240701_00.csv
│       ├── labels/
│       │   └── rainfall_bust_labels.csv
│       ├── dataset_splits/                     [reserved directory]
│       ├── gridded_sequences/                  [reserved directory]
│       └── tabular_features/                   [reserved directory]
│
├── docs/
│   └── rainfall_xgboost_tensorflow_convlstm_mvp.md
│
├── logs/
│   ├── acquisition/                            [reserved directory]
│   ├── inference/                              [reserved directory]
│   ├── preprocessing/                          [reserved directory]
│   └── training/                               [reserved directory]
│
├── models/
│   ├── calibration/                            [reserved directory]
│   ├── convlstm/                               [reserved directory]
│   ├── fusion/                                 [reserved directory]
│   ├── xgboost/                                [reserved directory]
│   └── registry/
│       ├── model_manifest.json
│       └── rainfall_bust_thresholds.json
│
├── notebooks/
│   ├── 01_inspect_forecast.ipynb
│   ├── 02_inspect_imd.ipynb
│   ├── 03_alignment_check.ipynb
│   ├── 04_bust_label_experiments.ipynb
│   └── 05_model_comparison.ipynb
│
├── outputs/
│   ├── README.md
│   ├── current_predictions/                    [reserved directory]
│   ├── evaluation/                             [reserved directory]
│   ├── explanations/                           [reserved directory]
│   ├── figures/                                [reserved directory]
│   ├── historical_predictions/                 [reserved directory]
│   └── maps/                                   [reserved directory]
│
├── scripts/
│   ├── __init__.py
│   ├── build_30_day_rainfall_metrics.py
│   ├── build_rainfall_metrics.py
│   ├── build_training_data.py
│   ├── decode_30_day_pressure_pilot.py
│   ├── decode_30_day_surface_pilot.py
│   ├── fit_rainfall_bust_thresholds.py
│   ├── label_rainfall_bust_events.py
│   ├── run_30_day_pilot.py
│   ├── run_live_forecast.py
│   ├── run_rainfall_bust_pipeline.py
│   ├── run_single_case.py
│   ├── split_tigge_bulk_remaining.py
│   ├── train_all_models.py
│   └── validate_tigge_30_day_pilot.py
│
├── src/
│   ├── __init__.py
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── download_era5.py
│   │   ├── download_gefs_operational.py
│   │   ├── download_gefs_reforecast.py
│   │   ├── download_gpm.py
│   │   ├── download_imd.py
│   │   ├── download_imd_rainfall.py
│   │   ├── download_imdaa.py
│   │   ├── download_official_events.py
│   │   ├── download_tigge.py
│   │   ├── download_tigge_30_day_pilot.py
│   │   ├── download_tigge_bulk_remaining.py
│   │   └── download_tigge_rainfall_control_canary.py
│   ├── catalogue/
│   │   ├── __init__.py
│   │   ├── build_catalogue.py
│   │   ├── classify_categories.py
│   │   ├── group_bust_records.py
│   │   └── match_official_events.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── common_errors.py
│   │   ├── cyclone.py
│   │   ├── heatwave.py
│   │   ├── heavy_rainfall.py
│   │   ├── monsoon_depression.py
│   │   ├── monsoon_phase.py
│   │   ├── overall_bust_label.py
│   │   ├── rainfall_scoring.py
│   │   └── western_disturbance.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── calibration_metrics.py
│   │   ├── classification_metrics.py
│   │   ├── evaluate_by_category.py
│   │   ├── evaluate_by_lead.py
│   │   ├── evaluate_by_region.py
│   │   └── spatial_metrics.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_gridded_sequences.py
│   │   ├── build_tabular_dataset.py
│   │   ├── ensemble_features.py
│   │   ├── event_features.py
│   │   ├── spatial_features.py
│   │   └── temporal_features.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── current_forecast_pipeline.py
│   │   ├── generate_map_output.py
│   │   └── historical_replay.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── calibrate.py
│   │   ├── convlstm.py
│   │   ├── predict.py
│   │   ├── train_convlstm.py
│   │   ├── train_fusion.py
│   │   └── train_xgboost.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── align_30_day_rainfall.py
│   │   ├── align_ncmrwf_imd_rainfall.py
│   │   ├── align_valid_time.py
│   │   ├── apply_masks.py
│   │   ├── convert_units.py
│   │   ├── daily_rainfall.py
│   │   ├── decode_grib.py
│   │   ├── decode_imd_binary.py
│   │   ├── prepare_imd_rainfall.py
│   │   ├── prepare_tigge_pressure_pilot.py
│   │   ├── prepare_tigge_rainfall.py
│   │   ├── prepare_tigge_surface_pilot.py
│   │   ├── quality_control.py
│   │   └── regrid.py
│   └── utils/
│       ├── __init__.py
│       ├── checksums.py
│       ├── logging.py
│       ├── paths.py
│       └── time_utils.py
│
└── tests/
    ├── test_30_day_alignment.py
    ├── test_bust_labels.py
    ├── test_common_errors.py
    ├── test_daily_rainfall.py
    ├── test_feature_leakage.py
    ├── test_inference.py
    ├── test_model_feature_contracts.py
    ├── test_rainfall_catalogue.py
    ├── test_rainfall_scoring.py
    ├── test_regridding.py
    └── test_valid_time_alignment.py
```

## Generated directories intentionally collapsed

- `.venv/`: installed Python interpreter links and third-party packages.
- `.git/`: Git repository metadata.
- `.pytest_cache/`: temporary pytest state.
- `**/__pycache__/`: compiled Python bytecode generated during execution.

These directories are environmental artifacts rather than maintained project source. They should not be used to understand or document the application architecture.
