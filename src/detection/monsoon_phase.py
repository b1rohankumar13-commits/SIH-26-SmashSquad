"""Active/break monsoon regime and regime-bust labels.

Follows the Rajeevan et al. (2010) / IMD approach: daily rainfall averaged over the
monsoon core zone (MCZ), standardised against its calendar-day climatology; an
active (break) spell is a run of >= 3 days with standardised anomaly > +1 (< -1).

Forecast side: each ensemble member's MCZ rainfall is standardised against the
*model's own* lead-dependent climatology (removes model bias), prefixed with the
observed days before initialisation (known at issue time) so the persistence rule
is well defined at lead 1. A regime bust is the ensemble being confidently wrong:
missing a spell (<MISS_BELOW of members) or calling one that did not happen
(>= FALSE_ALARM_AT of members) - same thresholds as the rainfall bust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rectangular approximation of the monsoon core zone over central India.
MCZ_BOX = {"south": 18.0, "north": 28.0, "west": 73.0, "east": 86.0}
ANOMALY_THRESHOLD = 1.0
MIN_RUN_DAYS = 3
MISS_BELOW = 0.20
FALSE_ALARM_AT = 0.50
CLIM_HALF_WINDOW_DAYS = 15


def mcz_mask(lats: np.ndarray, lons: np.ndarray, box: dict = MCZ_BOX) -> np.ndarray:
    lat_ok = (lats >= box["south"]) & (lats <= box["north"])
    lon_ok = (lons >= box["west"]) & (lons <= box["east"])
    return lat_ok[:, None] & lon_ok[None, :]


def area_mean(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NaN-aware mean over masked cells of the last two (lat, lon) dims."""
    return np.nanmean(field[..., mask], axis=-1)


def windowed_climatology(doy: np.ndarray, values: np.ndarray, targets: np.ndarray,
                         half_window: int = CLIM_HALF_WINDOW_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of `values` for each target day-of-year, pooling samples within
    +-half_window days (circular). Robust to seasons with sparse coverage."""
    means, stds = np.full(len(targets), np.nan), np.full(len(targets), np.nan)
    for i, t in enumerate(targets):
        dist = np.abs(doy - t)
        dist = np.minimum(dist, 366 - dist)
        sel = values[(dist <= half_window) & np.isfinite(values)]
        if sel.size >= 10:
            means[i], stds[i] = sel.mean(), sel.std(ddof=1)
    return means, stds


def regime_states(anomaly: np.ndarray, threshold: float = ANOMALY_THRESHOLD,
                  min_run: int = MIN_RUN_DAYS) -> np.ndarray:
    """+1 active / -1 break / 0 normal along the last axis, keeping only runs of
    >= min_run consecutive days beyond the threshold."""
    anomaly = np.asarray(anomaly, dtype=np.float64)
    out = np.zeros(anomaly.shape, dtype=np.int8)
    for sign, cond in ((1, anomaly > threshold), (-1, anomaly < -threshold)):
        flat_cond = cond.reshape(-1, cond.shape[-1])
        flat_out = out.reshape(-1, out.shape[-1])
        for row in range(flat_cond.shape[0]):
            run_start = None
            for t in range(flat_cond.shape[1] + 1):
                on = t < flat_cond.shape[1] and flat_cond[row, t]
                if on and run_start is None:
                    run_start = t
                elif not on and run_start is not None:
                    if t - run_start >= min_run:
                        flat_out[row, run_start:t] = sign
                    run_start = None
    return out


def regime_busts(p_state: np.ndarray, observed: np.ndarray,
                 miss_below: float = MISS_BELOW, false_alarm_at: float = FALSE_ALARM_AT):
    """Miss / false-alarm masks for one regime given ensemble probability and the
    observed yes/no for that regime."""
    observed = observed.astype(bool)
    miss = observed & (p_state < miss_below)
    false_alarm = ~observed & (p_state >= false_alarm_at)
    return miss, false_alarm


def detect_monsoon_phase_bust(p_active: np.ndarray, p_break: np.ndarray,
                              observed_state: np.ndarray) -> dict[str, np.ndarray]:
    """All four regime-bust masks plus their union, from ensemble regime
    probabilities and the observed state (+1 active / -1 break / 0 normal)."""
    miss_b, fa_b = regime_busts(p_break, observed_state == -1)
    miss_a, fa_a = regime_busts(p_active, observed_state == 1)
    return {"miss_break": miss_b, "fa_break": fa_b, "miss_active": miss_a,
            "fa_active": fa_a, "bust": miss_b | fa_b | miss_a | fa_a}


def to_series(dates, values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=np.float64), index=pd.DatetimeIndex(dates)).sort_index()
