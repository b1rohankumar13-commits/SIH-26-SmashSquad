"""Heavy-rainfall probabilities, event errors, objects, and timing evidence."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RainObject:
    object_id: int
    cell_count: int
    centroid_latitude: float
    centroid_longitude: float


def ensemble_exceedance_probability(member_rainfall, threshold_mm):
    members = np.asarray(member_rainfall, dtype=np.float64)
    valid_count = np.isfinite(members).sum(axis=0)
    exceedance_count = np.where(np.isfinite(members), members >= threshold_mm, 0).sum(
        axis=0
    )
    return np.divide(
        exceedance_count,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )


def event_error_masks(
    probability,
    observed_rainfall,
    rainfall_threshold_mm,
    miss_probability_below,
    false_alarm_probability_at_or_above,
):
    probability = np.asarray(probability, dtype=np.float64)
    observed = np.asarray(observed_rainfall, dtype=np.float64)
    valid = np.isfinite(probability) & np.isfinite(observed)
    observed_event = observed >= rainfall_threshold_mm
    miss = valid & observed_event & (probability < miss_probability_below)
    false_alarm = (
        valid
        & ~observed_event
        & (probability >= false_alarm_probability_at_or_above)
    )
    return miss, false_alarm, observed_event & valid


def extract_objects(mask, latitudes, longitudes, minimum_cells=4, connectivity=8):
    """Extract connected rainfall objects and unweighted geographic centroids."""
    binary = np.asarray(mask, dtype=bool)
    structure = (
        np.ones((3, 3), dtype=int)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    )
    labelled, count = label(binary, structure=structure)
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    objects = []
    for object_id in range(1, count + 1):
        rows, columns = np.where(labelled == object_id)
        if rows.size < minimum_cells:
            labelled[labelled == object_id] = 0
            continue
        objects.append(
            RainObject(
                object_id=object_id,
                cell_count=int(rows.size),
                centroid_latitude=float(latitudes[rows].mean()),
                centroid_longitude=float(longitudes[columns].mean()),
            )
        )
    return objects, labelled


def haversine_km(latitude_1, longitude_1, latitude_2, longitude_2):
    radius_km = 6371.0088
    lat_1, lon_1, lat_2, lon_2 = np.deg2rad(
        [latitude_1, longitude_1, latitude_2, longitude_2]
    )
    delta_latitude = lat_2 - lat_1
    delta_longitude = lon_2 - lon_1
    value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_longitude / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(np.sqrt(value)))


def match_object_displacements(forecast_objects, observed_objects):
    """Optimally match available objects by centroid distance."""
    if not forecast_objects or not observed_objects:
        return [], np.nan
    distances = np.array(
        [
            [
                haversine_km(
                    forecast.centroid_latitude,
                    forecast.centroid_longitude,
                    observed.centroid_latitude,
                    observed.centroid_longitude,
                )
                for observed in observed_objects
            ]
            for forecast in forecast_objects
        ]
    )
    forecast_indices, observed_indices = linear_sum_assignment(distances)
    matches = [
        {
            "forecast_object_id": forecast_objects[f_index].object_id,
            "observed_object_id": observed_objects[o_index].object_id,
            "displacement_km": float(distances[f_index, o_index]),
        }
        for f_index, o_index in zip(forecast_indices, observed_indices)
    ]
    return matches, float(np.mean([match["displacement_km"] for match in matches]))


def event_timing_error_days(forecast_event_masks, observed_event_masks, search_days=2):
    """Find the nearest forecast-event day for every observed-event day."""
    forecast_presence = np.asarray(forecast_event_masks, bool).reshape(
        len(forecast_event_masks), -1
    ).any(axis=1)
    observed_presence = np.asarray(observed_event_masks, bool).reshape(
        len(observed_event_masks), -1
    ).any(axis=1)
    errors = np.full(len(observed_presence), np.nan, dtype=np.float64)
    for day in np.where(observed_presence)[0]:
        start = max(0, day - search_days)
        end = min(len(forecast_presence), day + search_days + 1)
        candidates = np.where(forecast_presence[start:end])[0] + start
        if candidates.size:
            errors[day] = float(np.min(np.abs(candidates - day)))
    return errors
