# SIH Forecast Bust Prediction

Prototype for predicting one overall Day 1–10 forecast-bust probability over India using:

- TIGGE/GEFS archived forecasts
- IMD observations
- XGBoost for structured features
- ConvLSTM for gridded spatial-temporal sequences
- Logistic regression for probability fusion

The six supported context categories are heavy rainfall, monsoon depressions, cyclones, heat waves, western disturbances, and active/break monsoon phases.

## Start order

1. Configure `configs/sources.yaml` and `configs/grid.yaml`.
2. Run `scripts/run_single_case.py`.
3. Verify forecast/observation alignment.
4. Run `scripts/run_30_day_pilot.py`.
5. Build training data and train the models.

Large weather files, model artifacts, logs, and generated outputs are ignored by Git.
