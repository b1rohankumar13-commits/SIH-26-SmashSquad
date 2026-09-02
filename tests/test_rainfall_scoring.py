import numpy as np

from src.detection.rainfall_scoring import (
    composite_score,
    normalize_error,
    training_quantile_scale,
)


def test_training_quantile_normalization_uses_documented_ratio():
    scale = training_quantile_scale(np.arange(1.0, 11.0), quantile=0.90)
    assert np.isclose(normalize_error(scale, scale), 1.0)


def test_minimal_rainfall_mvp_score():
    score = composite_score(
        {"z_mae": 1.2, "fss_error": 0.4},
        event_error=1.0,
        weights={"z_mae": 0.50, "event_error": 0.30, "fss_error": 0.20},
    )
    assert np.isclose(score, 0.98)
