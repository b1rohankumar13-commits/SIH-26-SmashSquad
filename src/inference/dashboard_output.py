"""Convert Xarray model probabilities to API-ready GeoParquet output."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


REQUIRED_DIMS = ("lead_day", "latitude", "longitude")


def probability_grid(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    """Validate and order one forecast cycle's probability grid."""
    if variable not in dataset:
        raise ValueError(f"Dataset does not contain {variable!r}")
    values = dataset[variable].squeeze(drop=True)
    if set(values.dims) != set(REQUIRED_DIMS):
        raise ValueError(f"{variable!r} must have dimensions {REQUIRED_DIMS}")
    values = values.transpose(*REQUIRED_DIMS)

    leads = np.asarray(values.lead_day.values, dtype=float)
    if not (
        np.isfinite(leads).all()
        and np.all(leads == np.floor(leads))
        and np.all((leads >= 1) & (leads <= 10))
    ):
        raise ValueError("lead_day values must be integers from 1 through 10")

    latitudes = np.asarray(values.latitude.values, dtype=float)
    longitudes = np.asarray(values.longitude.values, dtype=float)
    probabilities = np.asarray(values.values, dtype=float)
    if not np.isfinite(latitudes).all() or np.any(np.abs(latitudes) > 90):
        raise ValueError("latitude values must be finite and within [-90, 90]")
    if not np.isfinite(longitudes).all() or np.any(np.abs(longitudes) > 180):
        raise ValueError("longitude values must be finite and within [-180, 180]")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return values.assign_coords(lead_day=leads.astype(int))


def to_prediction_geodataframe(
    dataset: xr.Dataset,
    *,
    variable: str = "overall_bust_probability",
    run_id: str,
    init_time: str | None = None,
) -> gpd.GeoDataFrame:
    """Flatten a validated Xarray probability grid into EPSG:4326 points."""
    if not run_id.strip():
        raise ValueError("run_id cannot be empty")
    values = probability_grid(dataset, variable)
    frame = values.to_dataframe(name="overall_bust_probability").reset_index()
    frame["run_id"] = run_id
    if init_time is not None:
        frame["init_time"] = pd.to_datetime(init_time, errors="raise")
    frame["grid_id"] = frame.apply(
        lambda row: f"{row.latitude:.6f}_{row.longitude:.6f}", axis=1
    )
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame.longitude, frame.latitude),
        crs="EPSG:4326",
    )


def assign_regions(
    predictions: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    *,
    region_column: str = "region_id",
) -> gpd.GeoDataFrame:
    """Attach approved polygon identifiers to prediction points."""
    if region_column not in boundaries.columns:
        raise ValueError(f"Boundary data does not contain {region_column!r}")
    if predictions.crs is None or boundaries.crs is None:
        raise ValueError("Prediction and boundary data must define a CRS")

    regions = boundaries[[region_column, "geometry"]].to_crs(predictions.crs)
    joined = gpd.sjoin(predictions, regions, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    keys = ["run_id", "lead_day", "grid_id"]
    if joined.duplicated(keys).any():
        raise ValueError("Approved region polygons overlap at prediction grid points")
    return joined


def publish_current_predictions(
    predictions: gpd.GeoDataFrame,
    output_directory: Path,
    *,
    run_id: str,
    overwrite: bool = False,
) -> Path:
    """Atomically publish a GeoParquet file for FastAPI and Streamlit."""
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_run_id = "".join(character for character in run_id if character.isalnum() or character in "-_")
    if not safe_run_id:
        raise ValueError("run_id must contain a filename-safe character")
    target = output_directory / f"bust_probabilities_{safe_run_id}.parquet"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Prediction output already exists: {target.name}")

    temporary = output_directory / f".{target.stem}.{uuid4().hex}.tmp.parquet"
    try:
        predictions.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
