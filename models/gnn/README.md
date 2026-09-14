# GNN artifact handoff

This directory is reserved for externally trained GNN artifacts. Existing
ConvLSTM artifacts and code remain independent and unchanged.

The external GNN process sends keyed predictions to `POST /gnn/predictions`.
Each record must contain `grid_id`, `lead_day`, `latitude`, `longitude`, and
`gnn_probability`. Include `overall_bust_probability` only after fusion and
calibration; those records are then published for the dashboard.

Record the final artifact path and training metadata in
`models/registry/model_manifest.json` when the trained artifact is delivered.
