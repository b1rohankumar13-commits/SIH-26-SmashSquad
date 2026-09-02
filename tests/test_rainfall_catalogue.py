import pytest

from src.catalogue.build_catalogue import (
    RAINFALL_CATALOGUE_FIELDS,
    build_rainfall_catalogue,
)


def _record(event_id="rain-20190715-00-L01"):
    return {
        "event_id": event_id,
        "init_time": "2019-07-15T00:00:00Z",
        "valid_date": "2019-07-16",
        "lead_day": 1,
        "region_id": "all_india",
        "source_forecast": "tigge_ncmrwf",
        "source_observation": "imd_0p25_rainfall",
    }


def test_rainfall_catalogue_preserves_established_schema():
    row = build_rainfall_catalogue([_record()])[0]
    assert tuple(row) == RAINFALL_CATALOGUE_FIELDS
    assert row["event_id"] == "rain-20190715-00-L01"


def test_rainfall_catalogue_rejects_duplicate_event_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        build_rainfall_catalogue([_record(), _record()])
