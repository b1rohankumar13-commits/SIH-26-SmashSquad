"""Apply fitted thresholds or emit honest candidate-only rainfall labels."""

import csv
import json
from pathlib import Path
import sys


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
    / "rainfall_bust_thresholds_20190715_20190813.json"
)
LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labels"
    / "rainfall_bust_labels_20190715_20190813.csv"
)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def label_events():
    if not REGISTRY_FILE.is_file():
        raise FileNotFoundError(
            f"Threshold registry not found: {REGISTRY_FILE}. Run fit_rainfall_bust_thresholds.py first."
        )
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    if not CATALOGUE_FILE.is_file():
        raise FileNotFoundError(CATALOGUE_FILE)
    with CATALOGUE_FILE.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Rainfall event catalogue is empty: {CATALOGUE_FILE}")

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
            strict_label = 1 if critical_override else ""
            output.update(
                {
                    "z_mae": "",
                    "fss_error_score_component": "",
                    "composite_bust_score": "",
                    "strict_q95_threshold": "",
                    "strict_bust_label": strict_label,
                    "label_status": (
                        "strict_critical_override_pending_q95"
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
            strict_label = int(score > group["q95"] or critical_override)
            output.update(normalized)
            output["fss_error_score_component"] = score_components["fss_error"]
            output.update(
                {
                    "composite_bust_score": score,
                    "strict_q95_threshold": group["q95"],
                    "strict_bust_label": strict_label,
                    "label_status": "strict_q95_final",
                }
            )
        output["candidate_strict_match"] = (
            int(int(output["candidate_bust"]) == int(output["strict_bust_label"]))
            if output["strict_bust_label"] != ""
            else ""
        )
        labelled.append(output)

    LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(labelled[0])
    with LABEL_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(labelled)
    q95_count = sum(row["label_status"] == "strict_q95_final" for row in labelled)
    override_count = sum(
        row["label_status"] == "strict_critical_override_pending_q95"
        for row in labelled
    )
    comparable = [row for row in labelled if row["candidate_strict_match"] != ""]
    matches = sum(int(row["candidate_strict_match"]) for row in comparable)
    print(f"Label rows: {len(labelled)}")
    print(f"Q95 strict labels: {q95_count}")
    print(f"Critical-override strict labels: {override_count}")
    print(f"Still awaiting Q95: {len(labelled) - q95_count - override_count}")
    print(f"Candidate/strict matches: {matches}/{len(comparable)}")
    print(f"Saved: {LABEL_FILE}")
    return LABEL_FILE


if __name__ == "__main__":
    label_events()
