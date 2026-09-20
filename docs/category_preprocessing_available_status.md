# Available-input category preprocessing

The user directed preprocessing to continue with available files and defer other data. All outputs are in `data/processed/categories/`, separated into heavy_rainfall, monsoon_depressions, cyclones, heat_waves, western_disturbances, and active_break_monsoon. Read the README in each category for semantics and pending steps.

Completed and numerically checked: 36 IMD grid files covering 2021-2023 and January 1-10, 2024; 699 source best-track coordinate records normalized into relevant category subsets; 64 narrative rows retained; the first GEFS initialization across all six categories. Two duplicate source system/timestamp pairs are preserved and recorded. No extraction issue rows were found.

The full GEFS forecast pass is running separately with four worker processes, each processing disjoint dates and serially decoding GRIB. Every new NetCDF is reopened and numerically compared with its in-memory arrays before its date is checkpointed. The successful independent sample checks cover rainfall accumulation, temperature conversion and extrema, ensemble mean/spread, soil missing cells, six-hour cloud windows, area-weighted observation rainfall, masks and January overlap.

Progress: `logs/category_preprocessing_stdout.log`; failures: `logs/category_preprocessing_stderr.log`. Final completion report: `data/metadata/category_preprocessing_available_v1/run_2021-01-01_2023-12-31.json`. Do not call the full forecast archive complete before that report confirms all 1,095 dates and zero failures.

ERA5, climatology, GPM, forecast-observation time joins, objective event tracking and bust labelling remain deferred. Raw sources and existing older pilot outputs are preserved.
