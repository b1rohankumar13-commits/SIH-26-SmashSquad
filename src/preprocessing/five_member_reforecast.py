"""Validate externally supplied five-member GEFSv12 fields for category training.

This adapter does not fetch data, align observations, invent missing variables,
or make bust labels. Retained 11-member data is legacy provenance, not training input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_CONFIG = PROJECT_ROOT / "configs" / "category_preprocessing_selection.yaml"
MEMBERS = ("c00", "p01", "p02", "p03", "p04")
LEADS = np.arange(1, 11)
LATITUDES = np.arange(5.25, 38.0, 0.5)
LONGITUDES = np.arange(65.25, 100.0, 0.5)
SIX_HOURLY_LEADS = np.arange(6, 241, 6)
DAILY_DIMS = ("lead_day", "member", "latitude", "longitude")
SIX_HOURLY_DIMS = ("six_hour_lead", "member", "latitude", "longitude")


def _require_grid_and_members(dataset: xr.Dataset) -> xr.Dataset:
    missing = set(DAILY_DIMS).difference(dataset.coords)
    if missing:
        raise ValueError(f"Structured reforecast is missing coordinates: {sorted(missing)}")
    members = tuple(str(value) for value in dataset.member.values)
    if members != MEMBERS:
        raise ValueError(f"Expected control plus four perturbed members {MEMBERS}; got {members}")
    if not np.array_equal(dataset.lead_day.values, LEADS):
        raise ValueError("Expected exactly forecast lead days 1 through 10")
    if not np.array_equal(dataset.latitude.values, LATITUDES):
        if np.array_equal(dataset.latitude.values, LATITUDES[::-1]):
            dataset = dataset.sortby("latitude")
        else:
            raise ValueError("Expected the approved 0.5-degree India latitude centres")
    if not np.array_equal(dataset.longitude.values, LONGITUDES):
        raise ValueError("Expected the approved 0.5-degree India longitude centres")
    return dataset


def prepare_category_forecast(dataset: xr.Dataset, category: str) -> xr.Dataset:
    """Select required issue-time fields and compute strict five-member summaries."""
    with CATEGORY_CONFIG.open("r", encoding="utf-8") as stream:
        categories = yaml.safe_load(stream)["categories"]
    if category not in categories:
        raise ValueError(f"Unknown forecast category: {category}")
    dataset = _require_grid_and_members(dataset)
    selected = tuple(categories[category]["forecast"])
    missing = set(selected).difference(dataset.data_vars)
    if missing:
        raise ValueError(f"{category} is missing required forecast fields: {sorted(missing)}")

    output = xr.Dataset(coords={name: dataset.coords[name] for name in DAILY_DIMS})
    for name in selected:
        field = dataset[name]
        expected_dims = SIX_HOURLY_DIMS if name == "temperature_2m_6hourly" else DAILY_DIMS
        if set(field.dims) != set(expected_dims):
            raise ValueError(f"{name} must have dimensions {expected_dims}, got {field.dims}")
        if name == "temperature_2m_6hourly":
            if "six_hour_lead" not in dataset.coords:
                raise ValueError("Expected a six_hour_lead coordinate for temperature_2m_6hourly")
            if not np.array_equal(dataset.six_hour_lead.values, SIX_HOURLY_LEADS):
                raise ValueError("Expected six-hourly lead hours 6 through 240")
            output = output.assign_coords(six_hour_lead=dataset.six_hour_lead)
        field = field.transpose(*expected_dims)
        output[name] = field
        valid_count = field.notnull().sum("member")
        output[f"{name}_valid_member_count"] = valid_count.astype("uint8")
        complete = valid_count == len(MEMBERS)
        output[f"{name}_ensemble_mean"] = field.mean("member", skipna=False).where(complete)
        output[f"{name}_ensemble_spread"] = field.std("member", ddof=0, skipna=False).where(complete)

    output.attrs.update(
        forecast_source="NOAA GEFSv12 reforecast 2000-2019",
        role="historical_training_forecast_only",
        category=category,
        member_count=len(MEMBERS),
        grid_degrees=0.5,
        forecast_lead_days="1-10",
        observation_join_status="not_joined",
        labels_created="none",
    )
    return output
