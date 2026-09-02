"""Fractions Skill Score and neighbourhood rainfall diagnostics."""

import numpy as np
from scipy.ndimage import uniform_filter


def window_size_from_degrees(neighbourhood_degrees, grid_spacing_degrees):
    """Return an odd window spanning approximately the requested width."""
    if neighbourhood_degrees <= 0 or grid_spacing_degrees <= 0:
        raise ValueError("Neighbourhood and grid spacing must be positive.")
    points = int(round(neighbourhood_degrees / grid_spacing_degrees)) + 1
    return points if points % 2 == 1 else points + 1


def neighbourhood_fraction(binary_field, window_size, valid_mask=None):
    """Calculate the valid-cell event fraction in each square neighbourhood."""
    field = np.asarray(binary_field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("A two-dimensional field is required.")
    valid = np.isfinite(field) if valid_mask is None else np.asarray(valid_mask, bool)
    numerator = uniform_filter(
        np.where(valid, field, 0.0), size=window_size, mode="constant", cval=0.0
    )
    denominator = uniform_filter(
        valid.astype(np.float64), size=window_size, mode="constant", cval=0.0
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def fractions_skill_score(forecast_fraction, observed_fraction):
    """Return FSS in [0, 1]; one is perfect neighbourhood agreement."""
    forecast = np.asarray(forecast_fraction, dtype=np.float64)
    observed = np.asarray(observed_fraction, dtype=np.float64)
    valid = np.isfinite(forecast) & np.isfinite(observed)
    if not valid.any():
        return np.nan
    forecast = forecast[valid]
    observed = observed[valid]
    denominator = np.sum(forecast**2 + observed**2)
    if denominator == 0:
        return 1.0
    return float(1.0 - np.sum((forecast - observed) ** 2) / denominator)


def memberwise_fss(member_fields, observed, threshold, window_size):
    """Calculate FSS for every ensemble member and their mean."""
    members = np.asarray(member_fields, dtype=np.float64)
    observation = np.asarray(observed, dtype=np.float64)
    if members.ndim != 3 or observation.ndim != 2:
        raise ValueError("Expected member×latitude×longitude and latitude×longitude.")
    valid = np.isfinite(observation)
    observed_fraction = neighbourhood_fraction(
        observation >= threshold, window_size, valid_mask=valid
    )
    scores = []
    for member in members:
        member_valid = valid & np.isfinite(member)
        forecast_fraction = neighbourhood_fraction(
            member >= threshold, window_size, valid_mask=member_valid
        )
        scores.append(fractions_skill_score(forecast_fraction, observed_fraction))
    scores = np.asarray(scores, dtype=np.float64)
    return scores, float(np.nanmean(scores)), observed_fraction
