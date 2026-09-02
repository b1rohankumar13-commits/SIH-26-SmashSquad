import numpy as np
import pytest
import xarray as xr

from src.preprocessing.daily_rainfall import cumulative_to_daily


def test_cumulative_to_daily_preserves_all_ten_forecast_days():
    cumulative = xr.DataArray(
        np.array([0.0, 2.0, 5.5, 5.5, 9.0, 10.0, 12.0, 15.0, 19.0, 24.0, 30.0]),
        dims="step",
        coords={"step": np.arange(0, 241, 24).astype("timedelta64[h]")},
        attrs={"units": "kg m**-2"},
    )

    daily = cumulative_to_daily(cumulative)

    np.testing.assert_allclose(
        daily.values,
        [2.0, 3.5, 0.0, 3.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    np.testing.assert_array_equal(
        daily.step.values,
        np.arange(24, 241, 24).astype("timedelta64[h]"),
    )
    assert daily.attrs["units"] == "kg m**-2"


def test_cumulative_to_daily_rejects_input_without_zero_hour_baseline():
    cumulative = xr.DataArray(
        [2.0, 5.0],
        dims="step",
        coords={"step": np.array([24, 48]).astype("timedelta64[h]")},
    )

    with pytest.raises(ValueError, match="zero-hour"):
        cumulative_to_daily(cumulative)


def test_cumulative_to_daily_rejects_non_increasing_leads():
    cumulative = xr.DataArray(
        [0.0, 2.0, 3.0],
        dims="step",
        coords={"step": np.array([0, 24, 24]).astype("timedelta64[h]")},
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        cumulative_to_daily(cumulative)
