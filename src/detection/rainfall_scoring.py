"""Documented training-Q90 normalization and rainfall-MVP scoring."""

import numpy as np


NORMALIZED_COMPONENT_COLUMNS = {"z_mae": "area_weighted_mae_mm"}


def training_quantile_scale(values, quantile=0.90):
    """Return the training-period high-error reference from the mathematics file."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("A training quantile cannot be fitted without finite values")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    return float(np.quantile(values, quantile))


def normalize_error(value, q90_scale, epsilon=1e-8):
    """Calculate ``Z = E / (Q90(E_training) + epsilon)``."""
    value = float(value)
    q90_scale = float(q90_scale)
    if not np.isfinite(value):
        return np.nan
    if not np.isfinite(q90_scale) or q90_scale < 0:
        raise ValueError("The training Q90 scale must be finite and non-negative")
    return value / (q90_scale + epsilon)


def composite_score(normalized, event_error, weights):
    """Calculate the documented minimal rainfall-MVP composite score."""
    values = (
        normalized.get("z_mae", np.nan),
        float(event_error),
        normalized.get("fss_error", np.nan),
    )
    if not np.isfinite(values).all():
        return np.nan
    return float(
        weights["z_mae"] * normalized["z_mae"]
        + weights["event_error"] * float(event_error)
        + weights["fss_error"] * normalized["fss_error"]
    )
