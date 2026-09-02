"""Shared magnitude and categorical-event error functions."""

from __future__ import annotations

import numpy as np


def absolute_error(forecast, observed):
    return abs(forecast - observed)


def latitude_area_weights(latitudes):
    """Return the documented cosine-latitude grid-cell weights."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if latitudes.ndim != 1:
        raise ValueError("latitudes must be a one-dimensional coordinate")
    if not np.isfinite(latitudes).all() or np.any(np.abs(latitudes) > 90):
        raise ValueError("latitudes must be finite values between -90 and 90 degrees")
    return np.cos(np.deg2rad(latitudes))


def _weighted_error_mean(error, latitudes):
    error = np.asarray(error, dtype=np.float64)
    if error.ndim < 2:
        raise ValueError("error must end with latitude and longitude dimensions")
    if error.shape[-2] != len(latitudes):
        raise ValueError(
            "The penultimate error dimension must match the latitude coordinate"
        )
    weights = latitude_area_weights(latitudes).reshape(
        (1,) * (error.ndim - 2) + (len(latitudes), 1)
    )
    valid = np.isfinite(error)
    numerator = np.sum(np.where(valid, error * weights, 0.0), axis=(-2, -1))
    denominator = np.sum(
        np.where(valid, np.broadcast_to(weights, error.shape), 0.0), axis=(-2, -1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def area_weighted_mae(forecast, observed, latitudes):
    """Calculate cosine-latitude-weighted MAE over the final two axes."""
    forecast, observed = np.broadcast_arrays(
        np.asarray(forecast, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
    )
    return _weighted_error_mean(np.abs(forecast - observed), latitudes)


def area_weighted_rmse(forecast, observed, latitudes):
    """Calculate cosine-latitude-weighted RMSE over the final two axes."""
    forecast, observed = np.broadcast_arrays(
        np.asarray(forecast, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
    )
    return np.sqrt(_weighted_error_mean((forecast - observed) ** 2, latitudes))


def event_contingency(forecast, observed, threshold):
    """Return hit, miss, false-alarm and correct-negative masks."""
    forecast, observed = np.broadcast_arrays(
        np.asarray(forecast, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
    )
    valid = np.isfinite(forecast) & np.isfinite(observed)
    forecast_event = forecast >= threshold
    observed_event = observed >= threshold
    return {
        "hit": valid & forecast_event & observed_event,
        "miss": valid & ~forecast_event & observed_event,
        "false_alarm": valid & forecast_event & ~observed_event,
        "correct_negative": valid & ~forecast_event & ~observed_event,
        "valid": valid,
    }
