"""Calibrate a fitted model on a separate chronological validation set."""

from __future__ import annotations

import numpy as np


def calibrate(model, features, labels, method="sigmoid"):
    """Fit a probability calibrator without refitting the underlying model.

    ``features`` and ``labels`` must come from a time-separated validation
    period.  ``FrozenEstimator`` is the supported scikit-learn replacement for
    the removed ``cv='prefit'`` interface.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be sigmoid or isotonic")
    labels = np.asarray(labels).reshape(-1)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("Calibration labels must be binary and contain both classes")
    return CalibratedClassifierCV(
        FrozenEstimator(model), method=method
    ).fit(features, labels)


def calibrate_fusion(model, xgb_predictions, spatial_predictions, labels, method="sigmoid"):
    """Calibrate the frozen stacker on a separate, keyed chronological period."""
    from src.models.prediction_contracts import FUSION_COLUMNS, align_records, fusion_records

    rows = align_records(fusion_records(xgb_predictions, spatial_predictions),
                         labels, ["bust_label"])
    return calibrate(model, rows[FUSION_COLUMNS], rows["bust_label"], method=method)
