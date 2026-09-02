"""Fit training-Q90 normalization and Q90/Q95 rainfall-score thresholds."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.rainfall_scoring import (
    NORMALIZED_COMPONENT_COLUMNS,
    composite_score,
    normalize_error,
    training_quantile_scale,
)


CONFIG_FILE = PROJECT_ROOT / "configs" / "bust_thresholds.yaml"
CATALOGUE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_catalogue"
    / "rainfall_events_20190715_20190813.csv"
)
REGISTRY_FILE = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "rainfall_bust_thresholds_20190715_20190813.json"
)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _group_key(row):
    return (row["region_id"], row["season"], int(row["lead_day"]))


def fit_thresholds():
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    minimum_samples = int(configuration["minimum_training_samples_per_group"])
    weights = configuration["rainfall"]["composite_weights"]

    if not CATALOGUE_FILE.is_file():
        raise FileNotFoundError(CATALOGUE_FILE)
    with CATALOGUE_FILE.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Rainfall event catalogue is empty: {CATALOGUE_FILE}")

    grouped = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    registry_groups = {}
    ready_count = 0
    for (region_id, season, lead_day), group_rows in sorted(grouped.items()):
        statistics = {}
        for normalized_name, column in NORMALIZED_COMPONENT_COLUMNS.items():
            q90_scale = training_quantile_scale(
                [_float(row[column]) for row in group_rows],
                quantile=configuration["primary_quantile"],
            )
            statistics[normalized_name] = {"q90_scale": q90_scale}

        scores = []
        for row in group_rows:
            normalized = {
                name: normalize_error(
                    _float(row[column]),
                    statistics[name]["q90_scale"],
                )
                for name, column in NORMALIZED_COMPONENT_COLUMNS.items()
            }
            normalized["fss_error"] = _float(row["fss_error"])
            scores.append(composite_score(normalized, _float(row["event_error"]), weights))

        enough_samples = len(group_rows) >= minimum_samples
        if enough_samples:
            ready_count += 1
        key = f"{region_id}|{season}|{lead_day}"
        registry_groups[key] = {
            "region_id": region_id,
            "season": season,
            "lead_day": lead_day,
            "sample_count": len(group_rows),
            "minimum_required": minimum_samples,
            "status": "ready" if enough_samples else "insufficient_history",
            "normalization": statistics,
            "q90": float(np.quantile(scores, configuration["primary_quantile"]))
            if enough_samples
            else None,
            "q95": float(np.quantile(scores, configuration["strict_quantile"]))
            if enough_samples
            else None,
        }

    registry = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalization_method": configuration["normalization"],
        "label_policy": "strict_q95",
        "critical_event_override": bool(configuration["critical_event_override"]),
        "weights": weights,
        "status": "ready" if ready_count == len(registry_groups) else "insufficient_history",
        "groups": registry_groups,
    }
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Catalogue rows: {len(rows)}")
    print(f"Ready groups: {ready_count}/{len(registry_groups)}")
    print(f"Registry status: {registry['status']}")
    print(f"Saved: {REGISTRY_FILE}")
    return REGISTRY_FILE


if __name__ == "__main__":
    fit_thresholds()
