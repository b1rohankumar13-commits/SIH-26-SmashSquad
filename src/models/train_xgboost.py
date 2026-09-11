"""Train the tabular bust-probability model."""

import numpy as np


def positive_class_weight(labels) -> float:
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("XGBoost labels must be finite binary values")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes")
    return float(negatives / positives)


def train_xgboost(features, labels, params):
    from xgboost import XGBClassifier

    effective_params = dict(params)
    # YAML also contains descriptive metadata that is not an XGB parameter.
    for name in ("prediction_unit", "class_imbalance", "split"):
        effective_params.pop(name, None)
    if effective_params.get("objective", "binary:logistic") != "binary:logistic":
        raise ValueError("Bust prediction requires objective=binary:logistic")
    effective_params.setdefault("objective", "binary:logistic")
    effective_params.setdefault("scale_pos_weight", positive_class_weight(labels))
    model = XGBClassifier(**effective_params)
    return model.fit(features, labels)
