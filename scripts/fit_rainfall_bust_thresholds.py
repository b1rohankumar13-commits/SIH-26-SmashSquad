"""Fit training-Q90 component normalization and a Q85 rainfall-bust cutoff."""

import argparse
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
    / "rainfall_bust_thresholds_q85_20190715_20190813.json"
)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _group_key(row):
    return (row["region_id"], row["season"], int(row["lead_day"]))


def fit_thresholds(*, catalogue_file=CATALOGUE_FILE, registry_file=REGISTRY_FILE, training_end=None):
    """Fit references on initialization dates through training_end, never validation dates."""
    if training_end is None:
        raise ValueError("A chronological training_end date is required for Q85 fitting")
    cutoff = datetime.fromisoformat(str(training_end)).date()
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    minimum_samples = int(configuration["minimum_training_samples_per_group"])
    weights = configuration["rainfall"]["composite_weights"]

    catalogue_file = Path(catalogue_file)
    registry_file = Path(registry_file)
    if not catalogue_file.is_file():
        raise FileNotFoundError(catalogue_file)
    with catalogue_file.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Rainfall event catalogue is empty: {catalogue_file}")

    training_rows = [
        row for row in rows
        if datetime.fromisoformat(row["init_time"][:10]).date() <= cutoff
    ]
    if not training_rows:
        raise ValueError("No forecast cases fall in the declared training period")

    grouped = {}
    for row in training_rows:
        grouped.setdefault(_group_key(row), []).append(row)

    registry_groups = {}
    ready_count = 0
    for (region_id, season, lead_day), group_rows in sorted(grouped.items()):
        required_columns = (*NORMALIZED_COMPONENT_COLUMNS.values(), "event_error", "fss_error")
        group_rows = [
            row for row in group_rows
            if all(np.isfinite(_float(row.get(column))) for column in required_columns)
        ]
        if not group_rows:
            raise ValueError(f"No finite scoring rows for {region_id}|{season}|{lead_day}")
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
            "q85": float(np.quantile(scores, configuration["bust_quantile"]))
            if enough_samples
            else None,
        }

    registry = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalization_method": configuration["normalization"],
        "label_policy": "composite_q85",
        "bust_quantile": float(configuration["bust_quantile"]),
        "training_end": cutoff.isoformat(),
        "critical_event_override": bool(configuration["critical_event_override"]),
        "weights": weights,
        "status": "ready" if ready_count == len(registry_groups) else "insufficient_history",
        "groups": registry_groups,
    }
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Training catalogue rows: {len(training_rows)} of {len(rows)}")
    print(f"Ready groups: {ready_count}/{len(registry_groups)}")
    print(f"Registry status: {registry['status']}")
    print(f"Saved: {registry_file}")
    return registry_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-end", required=True, help="Last training initialization date (YYYY-MM-DD)")
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE_FILE)
    parser.add_argument("--registry", type=Path, default=REGISTRY_FILE)
    arguments = parser.parse_args()
    fit_thresholds(
        catalogue_file=arguments.catalogue,
        registry_file=arguments.registry,
        training_end=arguments.training_end,
    )
