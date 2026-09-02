"""Build grid metrics and a per-lead rainfall event catalogue for one pilot case."""

import csv
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.heavy_rainfall import (
    ensemble_exceedance_probability,
    event_error_masks,
    event_timing_error_days,
    extract_objects,
    match_object_displacements,
)
from src.detection.common_errors import area_weighted_mae, area_weighted_rmse
from src.catalogue.build_catalogue import (
    RAINFALL_CATALOGUE_FIELDS,
    build_rainfall_catalogue,
)
from src.evaluation.spatial_metrics import (
    memberwise_fss,
    neighbourhood_fraction,
    window_size_from_degrees,
)


CONFIG_FILE = PROJECT_ROOT / "configs" / "bust_thresholds.yaml"
ALIGNED_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aligned"
    / "rainfall"
    / "2024"
    / "07"
    / "01"
    / "ncmrwf_imd_20240701_00_imd_window_day01_day09.nc"
)
GRID_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "errors"
    / "rainfall"
    / "2024"
    / "07"
    / "01"
    / "rainfall_grid_metrics_20240701_00.nc"
)
EVENT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_catalogue"
    / "rainfall_events_20240701_00.csv"
)


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _season(month):
    if month in (6, 7, 8, 9):
        return "southwest_monsoon"
    if month in (10, 11, 12):
        return "post_monsoon"
    if month in (3, 4, 5):
        return "pre_monsoon"
    return "winter"


def build_metrics():
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    rainfall_config = configuration["rainfall"]
    thresholds = rainfall_config["thresholds_mm"]
    probability_rules = rainfall_config["probability_rules"]
    object_config = rainfall_config["objects"]

    dataset = xr.open_dataset(ALIGNED_FILE, engine="netcdf4")
    members = dataset["forecast_member_rainfall_mm"]
    observed = dataset["observed_rainfall_mm"]
    latitudes = observed.latitude.values
    longitudes = observed.longitude.values
    latitude_spacing = float(np.median(np.diff(latitudes)))
    longitude_spacing = float(np.median(np.diff(longitudes)))
    grid_spacing = float(np.mean([abs(latitude_spacing), abs(longitude_spacing)]))
    window_size = window_size_from_degrees(
        rainfall_config["fss"]["neighbourhood_degrees"], grid_spacing
    )

    times = observed.time.values
    probabilities = {
        name: np.stack(
            [
                ensemble_exceedance_probability(members.isel(time=index).values, value)
                for index in range(observed.sizes["time"])
            ]
        )
        for name, value in thresholds.items()
    }
    heavy_probability = probabilities["heavy"]
    observed_values = observed.values
    observed_heavy_masks = observed_values >= thresholds["heavy"]
    forecast_event_masks = (
        heavy_probability >= object_config["forecast_probability_threshold"]
    )
    timing_errors = event_timing_error_days(
        forecast_event_masks,
        observed_heavy_masks,
        search_days=rainfall_config["timing"]["search_days"],
    )

    miss_maps = []
    false_alarm_maps = []
    neighbourhood_error_maps = []
    candidate_maps = []
    rows = []

    for index, valid_time in enumerate(times):
        member_values = members.isel(time=index).values
        observed_day = observed_values[index]
        probability = heavy_probability[index]
        valid = np.isfinite(observed_day)
        miss, false_alarm, observed_event = event_error_masks(
            probability,
            observed_day,
            thresholds["heavy"],
            probability_rules["critical_miss_below"],
            probability_rules["false_alarm_at_or_above"],
        )
        miss_maps.append(miss)
        false_alarm_maps.append(false_alarm)
        candidate_maps.append(miss | false_alarm)

        member_scores, mean_fss, observed_fraction = memberwise_fss(
            member_values,
            observed_day,
            thresholds["heavy"],
            window_size,
        )
        forecast_neighbourhood_probability = neighbourhood_fraction(
            probability,
            window_size,
            valid_mask=valid & np.isfinite(probability),
        )
        neighbourhood_error_maps.append(
            np.abs(forecast_neighbourhood_probability - observed_fraction)
        )

        forecast_objects, forecast_labels = extract_objects(
            forecast_event_masks[index],
            latitudes,
            longitudes,
            minimum_cells=object_config["minimum_cells"],
            connectivity=object_config["connectivity"],
        )
        observed_objects, observed_labels = extract_objects(
            observed_event,
            latitudes,
            longitudes,
            minimum_cells=object_config["minimum_cells"],
            connectivity=object_config["connectivity"],
        )
        matches, displacement = match_object_displacements(
            forecast_objects, observed_objects
        )

        critical_object_miss = any(
            float(np.nanmean(probability[observed_labels == item.object_id]))
            < probability_rules["critical_miss_below"]
            for item in observed_objects
        )
        false_alarm_object = any(
            float(np.mean(observed_event[forecast_labels == item.object_id])) < 0.5
            for item in forecast_objects
        )

        ensemble_mean = np.nanmean(member_values, axis=0)
        error = ensemble_mean - observed_day
        absolute_error = np.abs(error)
        valid_count = int(valid.sum())
        observed_event_count = int((observed_event & valid).sum())
        non_event_count = int((~observed_event & valid).sum())
        miss_rate = _fraction(int(miss.sum()), observed_event_count)
        false_alarm_rate = _fraction(int(false_alarm.sum()), non_event_count)
        event_error = 0.5 * (miss_rate + false_alarm_rate)
        date_string = np.datetime_as_string(valid_time.astype("datetime64[D]"))
        lead_day = index + 1

        rows.append(
            {
                "event_id": f"rain-20240701-00-L{lead_day:02d}",
                "init_time": "2024-07-01T00:00:00Z",
                "valid_date": date_string,
                "lead_day": lead_day,
                "region_id": "all_india",
                "season": _season(int(date_string[5:7])),
                "mae_mm": float(np.nanmean(absolute_error)),
                "rmse_mm": float(np.sqrt(np.nanmean(error**2))),
                "area_weighted_mae_mm": float(
                    area_weighted_mae(ensemble_mean, observed_day, latitudes)
                ),
                "area_weighted_rmse_mm": float(
                    area_weighted_rmse(ensemble_mean, observed_day, latitudes)
                ),
                "bias_mm": float(np.nanmean(error)),
                "mean_memberwise_fss": mean_fss,
                "fss_error": float(1.0 - mean_fss),
                "member_fss_min": float(np.nanmin(member_scores)),
                "member_fss_max": float(np.nanmax(member_scores)),
                "heavy_rain_miss_rate": miss_rate,
                "heavy_rain_false_alarm_rate": false_alarm_rate,
                "event_error": event_error,
                "forecast_object_count": len(forecast_objects),
                "observed_object_count": len(observed_objects),
                "matched_object_count": len(matches),
                "mean_displacement_km": displacement,
                "timing_error_days": float(timing_errors[index]),
                "critical_object_miss": int(critical_object_miss),
                "false_alarm_object": int(false_alarm_object),
                "critical_event_failure": int(
                    critical_object_miss or false_alarm_object
                ),
                "candidate_bust": int(critical_object_miss or false_alarm_object),
                "label_status": "candidate_only",
                "strict_bust_label": "",
                "valid_grid_cells": valid_count,
                "source_forecast": "tigge_ncmrwf",
                "source_observation": "imd_0p25_rainfall",
            }
        )

    grid_metrics = xr.Dataset(
        {
            "probability_heavy": (("time", "latitude", "longitude"), probabilities["heavy"].astype("float32")),
            "probability_very_heavy": (("time", "latitude", "longitude"), probabilities["very_heavy"].astype("float32")),
            "probability_extremely_heavy": (("time", "latitude", "longitude"), probabilities["extremely_heavy"].astype("float32")),
            "heavy_rain_miss": (("time", "latitude", "longitude"), np.asarray(miss_maps, dtype="int8")),
            "heavy_rain_false_alarm": (("time", "latitude", "longitude"), np.asarray(false_alarm_maps, dtype="int8")),
            "neighbourhood_fraction_error": (("time", "latitude", "longitude"), np.asarray(neighbourhood_error_maps, dtype="float32")),
            "candidate_bust": (("time", "latitude", "longitude"), np.asarray(candidate_maps, dtype="int8")),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        attrs={
            "label_status": "candidate_only",
            "heavy_rain_threshold_mm": thresholds["heavy"],
            "fss_neighbourhood_degrees": rainfall_config["fss"]["neighbourhood_degrees"],
            "fss_window_grid_points": window_size,
            "warning": "Observation-derived fields are targets/diagnostics, never live XGBoost inputs",
        },
    )

    GRID_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    grid_metrics.to_netcdf(GRID_OUTPUT, engine="netcdf4")
    catalogue_rows = build_rainfall_catalogue(rows)
    with EVENT_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAINFALL_CATALOGUE_FIELDS)
        writer.writeheader()
        writer.writerows(catalogue_rows)

    print(f"FSS window: {window_size} grid points (~1 degree)")
    print(f"Per-lead event rows: {len(rows)}")
    print(f"Grid metrics: {GRID_OUTPUT}")
    print(f"Event catalogue: {EVENT_OUTPUT}")
    return GRID_OUTPUT, EVENT_OUTPUT


if __name__ == "__main__":
    build_metrics()
