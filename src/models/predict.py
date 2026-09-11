"""Produce keyed grid probabilities using a calibrated fusion estimator."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted

from src.models.prediction_contracts import (
    FUSION_COLUMNS, KEY_COLUMNS, fusion_records, positive_probability,
    probability_values, validate_keys,
)


def predict_probabilities(xgb_model, convlstm_model, fusion_model, tabular, grids,
                          *, tabular_keys, grid_keys):
    """Return a DataFrame in tabular row order, joined by explicit forecast keys.

    grid_keys must follow C-order [run, lead, latitude, longitude] flattening;
    build them with grid_prediction_keys on the tensor's actual coordinates.
    Pass the final calibrated stacker, never an uncalibrated LogisticRegression.
    """
    if not isinstance(fusion_model, CalibratedClassifierCV):
        raise ValueError("Production inference requires a calibrated fusion estimator")
    check_is_fitted(fusion_model)
    xgb = validate_keys(tabular_keys)[KEY_COLUMNS].copy()
    spatial = validate_keys(grid_keys)[KEY_COLUMNS].copy()
    xgb_values = positive_probability(xgb_model, tabular)
    grids = np.asarray(grids)
    if grids.ndim != 5 or not np.isfinite(grids).all():
        raise ValueError("Grids must be finite batch×lead×latitude×longitude×channel tensors")
    predictions = np.asarray(convlstm_model.predict(grids, verbose=0), dtype=float)
    if predictions.shape != (*grids.shape[:4], 1):
        raise ValueError("ConvLSTM output must be batch×lead×latitude×longitude×1")
    if len(xgb) != xgb_values.size or len(spatial) != predictions.size:
        raise ValueError("Prediction keys must match their respective output row counts")
    xgb["xgboost_probability"] = xgb_values
    spatial["spatial_probability"] = probability_values(predictions)
    rows = fusion_records(xgb, spatial)
    rows["overall_bust_probability"] = positive_probability(fusion_model, rows[FUSION_COLUMNS])
    return rows.rename(columns={"spatial_probability": "convlstm_probability"})
