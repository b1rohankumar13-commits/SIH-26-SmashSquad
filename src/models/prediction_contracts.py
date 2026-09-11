"""Keyed, model-independent contracts for spatial probability fusion."""

import numpy as np
import pandas as pd

KEY_COLUMNS = ["run_id", "lead_day", "grid_id"]
FUSION_COLUMNS = ["xgboost_probability", "spatial_probability", "model_disagreement"]


def validate_keys(records):
    frame = pd.DataFrame(records).copy()
    if not set(KEY_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Prediction keys must include {KEY_COLUMNS}")
    if frame.empty or frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("Prediction keys cannot be empty or missing")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate prediction keys")
    leads = pd.to_numeric(frame["lead_day"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(leads).all() and np.all(leads == np.floor(leads))
            and np.all((leads >= 1) & (leads <= 10))):
        raise ValueError("lead_day must contain integers from 1 through 10")
    frame["lead_day"] = leads.astype(int)
    return frame


def grid_prediction_keys(run_ids, lead_days, grid_ids):
    """Keys in C-order for [run, lead, latitude, longitude, 1] outputs.

    grid_ids is a 2-D array on the exact tensor grid, including masked cells.
    Persist the accompanying coordinate-to-grid-id mapping with the data.
    """
    runs = np.asarray(run_ids)
    leads = np.asarray(lead_days)
    cells = np.asarray(grid_ids)
    if runs.ndim != 1 or leads.ndim != 1 or cells.ndim != 2:
        raise ValueError("Expected 1-D run IDs/leads and 2-D grid IDs")
    return validate_keys(pd.DataFrame({
        "run_id": np.repeat(runs, leads.size * cells.size),
        "lead_day": np.tile(np.repeat(leads, cells.size), runs.size),
        "grid_id": np.tile(cells.reshape(-1), runs.size * leads.size),
    }))


def probability_values(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return values


def positive_probability(model, features):
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Estimator classes must be binary 0/1")
    return probability_values(np.asarray(model.predict_proba(features))[:, np.flatnonzero(classes == 1)[0]])


def align_records(left, right, value_columns):
    """One-to-one join with exact key-set equality, preserving left order."""
    left = validate_keys(left)
    right = validate_keys(right)
    if not set(value_columns).issubset(right.columns):
        raise ValueError(f"Missing values: {value_columns}")
    matched = left[KEY_COLUMNS].merge(right[KEY_COLUMNS], on=KEY_COLUMNS,
                                     how="outer", indicator=True, validate="one_to_one")
    if not matched["_merge"].eq("both").all():
        raise ValueError("Prediction/label key sets do not match")
    return left.merge(right[KEY_COLUMNS + list(value_columns)], on=KEY_COLUMNS,
                      how="left", sort=False, validate="one_to_one")


def fusion_records(xgb_predictions, spatial_predictions):
    """Spatial branch may be ConvLSTM or GNN; its column is spatial_probability."""
    xgb = validate_keys(xgb_predictions)
    xgb["xgboost_probability"] = probability_values(xgb["xgboost_probability"])
    spatial = validate_keys(spatial_predictions)
    spatial["spatial_probability"] = probability_values(spatial["spatial_probability"])
    result = align_records(xgb[KEY_COLUMNS + ["xgboost_probability"]], spatial,
                           ["spatial_probability"])
    result["model_disagreement"] = abs(result["xgboost_probability"] - result["spatial_probability"])
    return result
