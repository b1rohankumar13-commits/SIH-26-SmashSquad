"""Apply fitted Q85 thresholds or emit honest candidate-only rainfall labels."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.rainfall_scoring import (
    NORMALIZED_COMPONENT_COLUMNS,
    composite_score,
    normalize_error,
)


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
LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labels"
    / "rainfall_bust_labels_q85_20190715_20190813.csv"
)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def label_events(*, catalogue_file=CATALOGUE_FILE, registry_file=REGISTRY_FILE, label_file=LABEL_FILE):
    catalogue_file = Path(catalogue_file)
    registry_file = Path(registry_file)
    label_file = Path(label_file)
    if not registry_file.is_file():
        raise FileNotFoundError(
            f"Threshold registry not found: {registry_file}. Run fit_rainfall_bust_thresholds.py first."
        )
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if registry.get("label_policy") != "composite_q85":
        raise ValueError("The threshold registry must use the composite Q85 policy")
    if not catalogue_file.is_file():
        raise FileNotFoundError(catalogue_file)
    with catalogue_file.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Rainfall event catalogue is empty: {catalogue_file}")

    labelled = []
    for row in rows:
        key = f"{row['region_id']}|{row['season']}|{int(row['lead_day'])}"
        group = registry["groups"].get(key)
        output = dict(row)
        critical_override = (
            registry["critical_event_override"]
            and int(row["critical_event_failure"]) == 1
        )
        if not group or group["status"] != "ready":
            bust_label = 1 if critical_override else ""
            output.update(
                {
                    "z_mae": "",
                    "fss_error_score_component": "",
                    "composite_bust_score": "",
                    "q85_threshold": "",
                    "bust_label": bust_label,
                    "label_status": (
                        "critical_override_pending_q85"
                        if critical_override
                        else "candidate_only_insufficient_history"
                    ),
                }
            )
        else:
            normalized = {}
            for name, column in NORMALIZED_COMPONENT_COLUMNS.items():
                statistics = group["normalization"][name]
                normalized[name] = normalize_error(
                    _float(row[column]), statistics["q90_scale"]
                )
            score_components = {
                **normalized,
                "fss_error": _float(row["fss_error"]),
            }
            score = composite_score(
                score_components, _float(row["event_error"]), registry["weights"]
            )
            valid_score = np.isfinite(score)
            bust_label = int(score > group["q85"] or critical_override) if valid_score else (1 if critical_override else "")
            output.update(normalized)
            output["fss_error_score_component"] = score_components["fss_error"]
            output.update(
                {
                    "composite_bust_score": score,
                    "q85_threshold": group["q85"],
                    "bust_label": bust_label,
                    "label_status": (
                        "q85_final" if valid_score else
                        "critical_override_pending_q85" if critical_override else
                        "candidate_only_invalid_metrics"
                    ),
                }
            )
        output["candidate_final_match"] = (
            int(int(output["candidate_bust"]) == int(output["bust_label"]))
            if output["bust_label"] != ""
            else ""
        )
        labelled.append(output)

    label_file.parent.mkdir(parents=True, exist_ok=True)
    fields = list(labelled[0])
    with label_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(labelled)
    q85_count = sum(row["label_status"] == "q85_final" for row in labelled)
    override_count = sum(
        row["label_status"] == "critical_override_pending_q85"
        for row in labelled
    )
    comparable = [row for row in labelled if row["candidate_final_match"] != ""]
    matches = sum(int(row["candidate_final_match"]) for row in comparable)
    print(f"Label rows: {len(labelled)}")
    print(f"Q85 final labels: {q85_count}")
    print(f"Critical-override labels pending Q85: {override_count}")
    print(f"Still awaiting Q85: {len(labelled) - q85_count - override_count}")
    print(f"Candidate/Q85 matches: {matches}/{len(comparable)}")
    print(f"Saved: {label_file}")
    return label_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE_FILE)
    parser.add_argument("--registry", type=Path, default=REGISTRY_FILE)
    parser.add_argument("--labels", type=Path, default=LABEL_FILE)
    arguments = parser.parse_args()
    label_events(
        catalogue_file=arguments.catalogue,
        registry_file=arguments.registry,
        label_file=arguments.labels,
    )
