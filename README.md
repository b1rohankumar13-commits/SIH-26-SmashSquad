# SIH Forecast Bust Prediction

Predict the probability of a large forecast error for each grid cell and lead day over India. XGBoost uses weather summaries; ConvLSTM uses weather-map sequences. A future GNN can replace the spatial branch.

## Pipeline

**Forecasts + observations → Alignment and labels → Chronological splits**  
**↳ XGBoost + ConvLSTM or GNN → Fusion → Calibration → Evaluation → Map/API**

Observations are used for labels and evaluation. Live model inputs contain only information available when the forecast was issued.

## Files needed for training

Paths are relative to the project root.

| Component | Files to access |
|---|---|
| Data preparation | configs/sources.yaml, configs/variables.yaml, configs/grid.yaml; src/preprocessing/ |
| Label definitions | configs/bust_thresholds.yaml; src/detection/rainfall_scoring.py, src/detection/overall_bust_label.py; scripts/fit_rainfall_bust_thresholds.py, scripts/label_rainfall_bust_events.py |
| Shared time splits | src/features/temporal_features.py |
| XGBoost | src/features/build_tabular_dataset.py; src/models/train_xgboost.py; configs/xgboost.yaml |
| ConvLSTM | src/features/build_gridded_sequences.py; src/models/convlstm.py, src/models/train_convlstm.py; configs/convlstm.yaml |
| Fusion and calibration | src/models/prediction_contracts.py, src/models/train_fusion.py, src/models/calibrate.py; configs/fusion.yaml |
| Prediction and evaluation | src/models/predict.py; src/inference/generate_map_output.py; src/evaluation/ |

**XGBoost needs:** one feature row and verified binary label per initialization, lead and grid cell, with matching identifiers.

**ConvLSTM needs:** weather-map sequences, matching grid labels and training-only normalization. Its 12 channels are mean/spread pairs for rainfall, pressure, U-wind, V-wind, humidity and geopotential height—not event categories. Missing labels are masked during training.

## GNN handoff

Reuse the same aligned data, labels, chronological splits and evaluation. Treat grid cells as nodes, weather variables as node features and geographic neighbours as edges. Produce one bust probability per node and lead, retaining the same run, lead and grid identifiers.

The next trainer should create the graph-data builder, GNN model/trainer, configuration and inference adapter. Refit and recalibrate fusion when replacing ConvLSTM with GNN. No GNN implementation exists yet.

## Training rules and current limits

- Separate **training → tuning → fusion → calibration → test** periods. Keep initialization groups together and exclude overlapping verification windows. Fit preprocessing only on training data.
- The rainfall pilot currently provides **nine leads**; Day 10 needs complete forecast and observation coverage.
- Existing all-India labels cannot be copied across grid cells. Finalized grid-level labels are required, and older models need retraining for the updated spatial outputs.
- scripts/build_training_data.py and scripts/train_all_models.py remain placeholders. Full training orchestration and six-category validation are unfinished.
- Compare individual models and fusion using probability calibration, Brier score, precision–recall performance and regional/lead results. Software tests do not establish forecast skill.

The dashboard should distinguish forecast-error probability from weather-event probability and show valid time, data freshness and missing coverage.
