"""Guard the approved five-member grid and training-only Q85 label policy."""

import csv
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from scripts.fit_rainfall_bust_thresholds import fit_thresholds
from scripts.label_rainfall_bust_events import label_events
from src.preprocessing.five_member_reforecast import (
    LATITUDES,
    LEADS,
    LONGITUDES,
    MEMBERS,
    prepare_category_forecast,
)


def _forecast_dataset():
    coordinates = {
        "lead_day": LEADS,
        "member": list(MEMBERS),
        "latitude": LATITUDES,
        "longitude": LONGITUDES,
    }
    values = np.ones((10, 5, 66, 70), dtype=np.float32)
    fields = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "configs" / "category_preprocessing_selection.yaml").read_text(encoding="utf-8")
    )["categories"]["heavy_rainfall"]["forecast"]
    return xr.Dataset(
        {name: (("lead_day", "member", "latitude", "longitude"), values)
         for name in fields},
        coords=coordinates,
    )


def test_five_member_adapter_preserves_grid_and_strict_spread():
    output = prepare_category_forecast(_forecast_dataset(), "heavy_rainfall")
    assert output.attrs["member_count"] == 5
    assert output.sizes["latitude"] == 66
    assert output.sizes["longitude"] == 70
    assert float(output.rainfall_24h_ensemble_mean.isel(lead_day=0, latitude=0, longitude=0)) == 1.0
    assert float(output.rainfall_24h_ensemble_spread.isel(lead_day=0, latitude=0, longitude=0)) == 0.0


def test_five_member_adapter_rejects_missing_member():
    with pytest.raises(ValueError, match="control plus four"):
        prepare_category_forecast(_forecast_dataset().isel(member=slice(0, 4)), "heavy_rainfall")


def test_q85_uses_training_rows_only_and_labels_full_catalogue(tmp_path):
    catalogue = tmp_path / "catalogue.csv"
    registry = tmp_path / "q85.json"
    labels = tmp_path / "labels.csv"
    fieldnames = ["init_time", "region_id", "season", "lead_day", "area_weighted_mae_mm",
                  "event_error", "fss_error", "critical_event_failure", "candidate_bust"]
    with catalogue.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(31):
            writer.writerow({
                "init_time": (date(2010, 1, 1) + timedelta(days=index)).isoformat(),
                "region_id": "all_india", "season": "winter", "lead_day": 1,
                "area_weighted_mae_mm": index + 1 if index < 30 else 1000,
                "event_error": 0.1, "fss_error": 0.2,
                "critical_event_failure": 0, "candidate_bust": 0,
            })
    with pytest.raises(ValueError, match="training_end"):
        fit_thresholds(catalogue_file=catalogue, registry_file=registry)
    fit_thresholds(catalogue_file=catalogue, registry_file=registry, training_end="2010-01-30")
    fitted = json.loads(registry.read_text(encoding="utf-8"))
    group = fitted["groups"]["all_india|winter|1"]
    assert fitted["label_policy"] == "composite_q85"
    assert group["sample_count"] == 30
    assert group["q85"] < 1
    label_events(catalogue_file=catalogue, registry_file=registry, label_file=labels)
    with labels.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 31
    assert rows[-1]["bust_label"] == "1"
    assert rows[-1]["label_status"] == "q85_final"
