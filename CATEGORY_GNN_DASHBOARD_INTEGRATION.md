# Six-GNN dashboard integration

The UI now filters Current forecast, Regional details, and Model validation by weather category. It does **not** train or run the six GNNs. Until real outputs arrive, values and charts remain empty.

## Prediction contract

Publish one combined, long-form Parquet table to `outputs/current_predictions/` (one row per run, category, grid cell, and lead day). Required columns: `run_id`, `region_id`, `category`, `lead_day` (1–10), `latitude`, `longitude`, and **final calibrated** `category_bust_probability` (0–1). Include `model_id` and `grid_id` to identify the contributing GNN and cell. Use exactly: `Heavy rainfall`, `Monsoon depression`, `Cyclone`, `Heat wave`, `Western disturbance`, `Active / break monsoon`. The displayed forecast confidence is `1 − category_bust_probability`; do not substitute a raw GNN score. Each of the six categories needs its own rows at the same run/lead for the six-category panel to be complete.

## Files to connect

| File(s) | Integration needed |
| --- | --- |
| `api/schemas.py`, `api/services/gnn_store.py`, `api/routes/gnn.py` | Accept a category for every GNN output; identify stored files by run **and** category so six models do not overwrite one another. Publish the final calibrated category probability in the combined table without requiring the old fusion/overall score. |
| `api/services/prediction_store.py`, `api/routes/forecast.py` | Serve that table through `GET /forecast/current`, including category filtering and all six categories for a run. The basic category fields/filter are in place; check pagination if a run exceeds 50,000 records. |
| `src/inference/dashboard_output.py`, `scripts/publish_dashboard_predictions.py` | If this export path is used, replace the overall/fusion output mapping with the six category-specific final probabilities. No data-acquisition or preprocessing changes are needed here. |
| `dashboard/category_data.py`, `dashboard/pages/current_forecast.py`, `dashboard/pages/regional_details.py`, `dashboard/components/india_map.py`, `dashboard/components/probability_panel.py` | Already read/filter/display category-specific predictions; connect by supplying the table above. The lead graph changes with the Weather category selection. |
| `outputs/model_validation/*.parquet`, `dashboard/pages/model_validation.py` | Supply one row per category and lead band with `category`, `model_id`, `lead_band`, `pr_auc`, `brier_score`, `recall`, `false_alarm_ratio`. Include an `All leads` row for headline cards and separate Day 1–3, 4–7, 8–10 rows. Supply a category-specific reliability plot separately when available. |
| `dashboard/pages/historical_replay.py`, `dashboard/pages/system_status.py` | These pages still need real historical replay/status sources; they are not made model-aware by this UI change. |

The model team should define/calibrate and evaluate each GNN's probability using a chronological holdout, then publish the final outputs. `compose.yaml` already runs the API and dashboard and mounts `outputs/`; Docker Compose does not itself run or train the models.
