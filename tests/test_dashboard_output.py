"""Integration tests for Xarray, GeoPandas and GeoParquet dashboard output."""

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from api.services.prediction_store import load_current_snapshot
from src.inference.dashboard_output import (
    assign_regions,
    publish_current_predictions,
    to_prediction_geodataframe,
)


def _dataset():
    return xr.Dataset(
        {
            "overall_bust_probability": (
                ("lead_day", "latitude", "longitude"),
                np.array([[[0.2, 0.4], [0.6, 0.8]], [[0.1, 0.3], [0.5, 0.7]]]),
            )
        },
        coords={"lead_day": [1, 2], "latitude": [20.0, 21.0], "longitude": [85.0, 86.0]},
    )


def test_xarray_to_geoparquet_to_api(tmp_path):
    predictions = to_prediction_geodataframe(_dataset(), run_id="20260914_00")
    assert len(predictions) == 8
    assert predictions.crs.to_epsg() == 4326

    output = publish_current_predictions(
        predictions, tmp_path, run_id="20260914_00"
    )
    snapshot = load_current_snapshot(tmp_path)
    assert output.suffix == ".parquet"
    assert snapshot.status == "available"
    assert len(snapshot.frame) == 8


def test_approved_boundaries_assign_region_ids():
    predictions = to_prediction_geodataframe(_dataset(), run_id="20260914_00")
    boundaries = gpd.GeoDataFrame(
        {"region_id": ["test_region"]},
        geometry=[box(84.5, 19.5, 86.5, 21.5)],
        crs="EPSG:4326",
    )
    assigned = assign_regions(predictions, boundaries)
    assert assigned["region_id"].eq("test_region").all()
