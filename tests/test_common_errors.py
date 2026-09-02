import numpy as np

from src.detection.common_errors import (
    absolute_error,
    area_weighted_mae,
    area_weighted_rmse,
    event_contingency,
    latitude_area_weights,
)


def test_absolute_error():
    assert absolute_error(12.0, 8.5) == 3.5


def test_latitude_area_weights_follow_cosine_rule():
    np.testing.assert_allclose(latitude_area_weights([0.0, 60.0]), [1.0, 0.5])


def test_area_weighted_errors_ignore_missing_pairs():
    forecast = np.array([[3.0, 5.0], [9.0, np.nan]])
    observed = np.array([[1.0, 1.0], [3.0, 2.0]])
    latitudes = np.array([0.0, 60.0])

    assert np.isclose(area_weighted_mae(forecast, observed, latitudes), 3.6)
    assert np.isclose(area_weighted_rmse(forecast, observed, latitudes), np.sqrt(15.2))


def test_event_contingency_has_all_four_outcomes():
    result = event_contingency(
        forecast=np.array([70.0, 10.0, 70.0, 10.0]),
        observed=np.array([70.0, 70.0, 10.0, 10.0]),
        threshold=64.5,
    )

    for outcome in ("hit", "miss", "false_alarm", "correct_negative"):
        assert int(result[outcome].sum()) == 1
