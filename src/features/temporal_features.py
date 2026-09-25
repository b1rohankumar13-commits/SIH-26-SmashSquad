"""Lag, persistence, trend, season, and lead-time features."""

from __future__ import annotations

import numpy as np


def build_temporal_features(sequence):
    raise NotImplementedError("Choose history length after the pilot.")


def chronological_split_indices(
    initialization_times,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
):
    """Split unique initializations chronologically, never randomly."""
    times = np.asarray(initialization_times).astype("datetime64[ns]")
    if times.ndim != 1 or times.size < 5:
        raise ValueError("At least five initialization times are required")
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    if np.any(sorted_times[1:] == sorted_times[:-1]):
        raise ValueError("Initialization times must be unique at sequence level")
    train_end = int(np.floor(times.size * train_fraction))
    validation_end = train_end + int(np.floor(times.size * validation_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= times.size:
        raise ValueError("Fractions produce an empty chronological split")
    return {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "test": order[validation_end:],
    }


def training_stage_indices(initialization_times, verification_end_times, *,
                           train_end, tuning_end, fusion_end, calibration_end):
    """Five chronological stages, purging verification overlap at boundaries.

    Boundary timestamps are exclusive. Supply one row per initialization and
    its latest verification end; share the returned indices across all models.
    Choose boundaries between named events. Returned 'purged' rows must not be
    reassigned to another stage. Observation publication delays, if applicable,
    should be included in verification_end_times as label-availability times.
    """
    times = np.asarray(initialization_times, dtype="datetime64[ns]")
    ends = np.asarray(verification_end_times, dtype="datetime64[ns]")
    cuts = np.asarray([train_end, tuning_end, fusion_end, calibration_end],
                      dtype="datetime64[ns]")
    if times.ndim != 1 or times.size == 0 or ends.shape != times.shape:
        raise ValueError("Supply matching initialization and verification-end vectors")
    if np.isnat(times).any() or np.isnat(ends).any() or np.isnat(cuts).any():
        raise ValueError("Split timestamps cannot be missing")
    if len(np.unique(times)) != times.size or np.any(ends < times):
        raise ValueError("Initializations must be unique and precede verification ends")
    if np.any(cuts[1:] <= cuts[:-1]):
        raise ValueError("Stage boundaries must increase strictly")
    stages = np.searchsorted(cuts, times, side="right")
    result = {}
    purged = np.zeros(times.size, dtype=bool)
    for stage, name in enumerate(("train", "tuning", "fusion", "calibration", "test")):
        selected = stages == stage
        if stage < 4:
            overlap = selected & (ends >= cuts[stage])
            purged |= overlap
            selected &= ~overlap
        indices = np.flatnonzero(selected)
        if indices.size == 0:
            raise ValueError(f"Empty {name} stage after purging; supply more history or different boundaries")
        result[name] = indices[np.argsort(times[indices], kind="stable")]
    result["purged"] = np.flatnonzero(purged)
    return result
