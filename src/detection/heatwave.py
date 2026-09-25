"""Heat-wave day criterion and per-cell heatwave-bust labels (miss / false alarm).

IMD plains criterion (v1, applied everywhere on the India land mask):
  heatwave day  <=>  Tmax >= 40 C  AND  (Tmax - normal >= 4.5 C  OR  Tmax >= 45 C)
The normal is a per-cell day-of-year climatology of observed Tmax.

Forecast side: each GEFS member is judged with the same criterion after removing the
model's mean Tmax bias (per cell, lead, calendar month; fitted on training years), so
a "bust" is a genuine ensemble error, not a known constant offset. Ensemble heatwave
probability = fraction of members meeting the criterion.

Bust (mirrors rainfall): MISS = P < 0.2 and a heatwave was observed;
FALSE_ALARM = P >= 0.5 and no heatwave was observed.
"""

from __future__ import annotations

import numpy as np

HW_ABS_C = 40.0
HW_DEPARTURE_C = 4.5
HW_SEVERE_ABS_C = 45.0
MISS_BELOW = 0.20
FALSE_ALARM_AT = 0.50


def heatwave_day(tmax_c: np.ndarray, normal_c: np.ndarray) -> np.ndarray:
    """Boolean heatwave-day mask; NaN inputs give False."""
    with np.errstate(invalid="ignore"):
        return (tmax_c >= HW_ABS_C) & ((tmax_c - normal_c >= HW_DEPARTURE_C) | (tmax_c >= HW_SEVERE_ABS_C))


def ensemble_heatwave_probability(members_c: np.ndarray, normal_c: np.ndarray, member_axis: int = 0) -> np.ndarray:
    """Fraction of members (bias-corrected Tmax, deg C) meeting the heatwave criterion."""
    normal = np.expand_dims(normal_c, member_axis)
    return heatwave_day(members_c, normal).mean(axis=member_axis).astype(np.float32)


def detect_heatwave_bust(probability: np.ndarray, observed_hw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Directional labels [..., 2] = (miss, false_alarm) as float32, NaN where not valid."""
    miss = (probability < MISS_BELOW) & observed_hw
    fa = (probability >= FALSE_ALARM_AT) & ~observed_hw
    out = np.stack([miss, fa], axis=-1).astype(np.float32)
    out[~valid] = np.nan
    return out
