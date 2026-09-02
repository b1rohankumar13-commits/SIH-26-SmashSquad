"""Build rainfall error grids and the established catalogue for all 30 runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.download_tigge_30_day_pilot import pilot_dates
from src.catalogue.build_catalogue import (
    RAINFALL_CATALOGUE_FIELDS,
    build_rainfall_catalogue,
)
from src.detection.common_errors import area_weighted_mae, area_weighted_rmse
from src.detection.heavy_rainfall import (
    ensemble_exceedance_probability,
    event_error_masks,
    event_timing_error_days,
    extract_objects,
    match_object_displacements,
)
from src.evaluation.spatial_metrics import (
    memberwise_fss,
    neighbourhood_fraction,
    window_size_from_degrees,
)


CONFIG_FILE = PROJECT_ROOT / "configs" / "bust_thresholds.yaml"
ALIGNED_ROOT = PROJECT_ROOT / "data" / "interim" / "aligned" / "rainfall"
GRID_ROOT = PROJECT_ROOT / "data" / "processed" / "errors" / "rainfall"
EVENT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_catalogue"
    / "rainfall_events_20190715_20190813.csv"
)


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "southwest_monsoon"
    if month in (10, 11, 12):
        return "post_monsoon"
    if month in (3, 4, 5):
        return "pre_monsoon"
    return "winter"


def _aligned_path(initialization) -> Path:
    token = initialization.strftime("%Y%m%d")
    return (
        ALIGNED_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / f"ncmrwf_imd_{token}_00_common0p5_day01-day09.nc"
    )


def _grid_output(initialization) -> Path:
    token = initialization.strftime("%Y%m%d")
    return (
        GRID_ROOT
        / initialization.strftime("%Y")
        / initialization.strftime("%m")
        / initialization.strftime("%d")
        / f"rainfall_grid_metrics_{token}_00.nc"
    )


def build_one(initialization, configuration: dict, *, overwrite: bool):
    aligned_path = _aligned_path(initialization)
    grid_output = _grid_output(initialization)
    if grid_output.exists() and not overwrite:
        raise FileExistsError(grid_output)

    rainfall_config = configuration["rainfall"]
    thresholds = rainfall_config["thresholds_mm"]
    probability_rules = rainfall_config["probability_rules"]
    object_config = rainfall_config["objects"]
    dataset = xr.open_dataset(aligned_path, engine="netcdf4").squeeze(
        "init_time", drop=True
    )
    members = dataset["forecast_member_rainfall_mm"]
    observed = dataset["observed_rainfall_mm"]
    latitudes = observed.latitude.values
    longitudes = observed.longitude.values
    grid_spacing = float(
        np.mean(
            [
                abs(np.median(np.diff(latitudes))),
                abs(np.median(np.diff(longitudes))),
            ]
        )
    )
    window_size = window_size_from_degrees(
        rainfall_config["fss"]["neighbourhood_degrees"], grid_spacing
    )

    probabilities = {
        name: np.stack(
            [
                ensemble_exceedance_probability(
                    members.isel(lead_day=index).values, value
                )
                for index in range(observed.sizes["lead_day"])
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
    valid_times = dataset.valid_time.values
    token = initialization.strftime("%Y%m%d")

    for index, valid_time in enumerate(valid_times):
        member_values = members.isel(lead_day=index).values
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
            member_values, observed_day, thresholds["heavy"], window_size
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
        valid_count = int(valid.sum())
        observed_event_count = int((observed_event & valid).sum())
        non_event_count = int((~observed_event & valid).sum())
        miss_rate = _fraction(int(miss.sum()), observed_event_count)
        false_alarm_rate = _fraction(int(false_alarm.sum()), non_event_count)
        date_string = np.datetime_as_string(valid_time.astype("datetime64[D]"))
        lead_day = index + 1
        rows.append(
            {
                "event_id": f"rain-{token}-00-L{lead_day:02d}",
                "init_time": f"{initialization.isoformat()}T00:00:00Z",
                "valid_date": date_string,
                "lead_day": lead_day,
                "region_id": "all_india",
                "season": _season(int(date_string[5:7])),
                "mae_mm": float(np.nanmean(np.abs(error))),
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
                "event_error": 0.5 * (miss_rate + false_alarm_rate),
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
                "source_observation": "imd_0p25_conservative_to_common0p5",
            }
        )

    grid_metrics = xr.Dataset(
        {
            "probability_heavy": (
                ("lead_day", "latitude", "longitude"),
                probabilities["heavy"].astype("float32"),
            ),
            "probability_very_heavy": (
                ("lead_day", "latitude", "longitude"),
                probabilities["very_heavy"].astype("float32"),
            ),
            "probability_extremely_heavy": (
                ("lead_day", "latitude", "longitude"),
                probabilities["extremely_heavy"].astype("float32"),
            ),
            "heavy_rain_miss": (
                ("lead_day", "latitude", "longitude"),
                np.asarray(miss_maps, dtype="int8"),
            ),
            "heavy_rain_false_alarm": (
                ("lead_day", "latitude", "longitude"),
                np.asarray(false_alarm_maps, dtype="int8"),
            ),
            "neighbourhood_fraction_error": (
                ("lead_day", "latitude", "longitude"),
                np.asarray(neighbourhood_error_maps, dtype="float32"),
            ),
            "candidate_bust": (
                ("lead_day", "latitude", "longitude"),
                np.asarray(candidate_maps, dtype="int8"),
            ),
        },
        coords={
            "lead_day": np.arange(1, 10, dtype="int8"),
            "valid_time": ("lead_day", valid_times),
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "forecast_initialization": f"{initialization.isoformat()}T00:00:00Z",
            "label_status": "candidate_only",
            "heavy_rain_threshold_mm": thresholds["heavy"],
            "fss_neighbourhood_degrees": rainfall_config["fss"][
                "neighbourhood_degrees"
            ],
            "fss_window_grid_points": window_size,
            "warning": "Observation-derived fields are never live model inputs",
        },
    )
    grid_output.parent.mkdir(parents=True, exist_ok=True)
    grid_metrics.to_netcdf(grid_output, engine="netcdf4")
    return rows, grid_output


def build_all(*, overwrite: bool = False) -> tuple[Path, list[Path]]:
    aligned_paths = [_aligned_path(initialization) for initialization in pilot_dates()]
    missing = [str(path) for path in aligned_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The aligned 30-day rainfall pilot is incomplete:\n" + "\n".join(missing)
        )
    if EVENT_OUTPUT.exists() and not overwrite:
        raise FileExistsError(EVENT_OUTPUT)
    with CONFIG_FILE.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)

    all_rows = []
    grid_outputs = []
    for initialization in pilot_dates():
        rows, grid_output = build_one(
            initialization, configuration, overwrite=overwrite
        )
        all_rows.extend(rows)
        grid_outputs.append(grid_output)
        print(f"Metrics {initialization}: {grid_output}")

    catalogue = build_rainfall_catalogue(all_rows)
    EVENT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAINFALL_CATALOGUE_FIELDS)
        writer.writeheader()
        writer.writerows(catalogue)
    print(f"Catalogue rows: {len(catalogue)}")
    print(f"Event catalogue: {EVENT_OUTPUT}")
    return EVENT_OUTPUT, grid_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-derived", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_all(overwrite=arguments.overwrite_derived)
