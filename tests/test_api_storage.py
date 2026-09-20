"""Tests for the local prediction storage service."""

import pandas as pd

from api.services.prediction_store import (
    catalogue_values,
    filter_predictions,
    load_current_snapshot,
    prediction_records,
)


def test_load_filter_and_serialize_csv(tmp_path):
    table = pd.DataFrame(
        {
            "latitude": [20.0, 21.0],
            "longitude": [85.0, 86.0],
            "lead_day": [3, 6],
            "overall_bust_probability": [0.25, 0.75],
            "region_id": ["odisha", "west_bengal"],
            "run_id": ["20260914_00", "20260914_00"],
        }
    )
    table.to_csv(tmp_path / "current.csv", index=False)

    snapshot = load_current_snapshot(tmp_path)

    assert snapshot.status == "available"
    assert catalogue_values(snapshot.frame, "region_id") == ["odisha", "west_bengal"]
    selected = filter_predictions(snapshot.frame, region_id="odisha", lead_day=3)
    records = prediction_records(selected)
    assert records[0]["overall_bust_probability"] == 0.25


def test_missing_output_is_an_honest_empty_state(tmp_path):
    snapshot = load_current_snapshot(tmp_path)
    assert snapshot.status == "not_generated"
    assert snapshot.frame is None


def test_six_category_rows_filter_and_serialize(tmp_path):
    table = pd.DataFrame(
        {
            "latitude": [20.0, 20.0],
            "longitude": [85.0, 85.0],
            "lead_day": [3, 3],
            "category": ["Heavy rainfall", "Cyclone"],
            "category_bust_probability": [0.8, 0.2],
            "model_id": ["rain-gnn", "cyclone-gnn"],
            "region_id": ["odisha", "odisha"],
            "run_id": ["20260914_00", "20260914_00"],
        }
    )
    table.to_csv(tmp_path / "categories.csv", index=False)
    snapshot = load_current_snapshot(tmp_path)
    assert snapshot.status == "available"
    cyclone = filter_predictions(snapshot.frame, category="Cyclone")
    assert len(cyclone) == 1
    assert prediction_records(cyclone)[0]["category_bust_probability"] == 0.2
