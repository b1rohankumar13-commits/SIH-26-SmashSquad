"""Fit fusion on keyed, unseen base-model predictions at the same spatial scale."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.prediction_contracts import FUSION_COLUMNS, align_records, fusion_records


def train_fusion(xgb_predictions, spatial_predictions, labels):
    """Frames contain run_id, lead_day, grid_id and their value column.

    Value columns: xgboost_probability, spatial_probability, bust_label.
    Use unseen chronological predictions; this function cannot infer their
    training provenance. Filter unverified keys from ALL three tables first.
    """
    rows = align_records(fusion_records(xgb_predictions, spatial_predictions),
                         labels, ["bust_label"])
    target = rows["bust_label"].to_numpy(dtype=float)
    if not np.isin(target, [0, 1]).all() or np.unique(target).size != 2:
        raise ValueError("Fusion labels must be finite binary values with both classes")
    return LogisticRegression(max_iter=1000).fit(rows[FUSION_COLUMNS], target)
