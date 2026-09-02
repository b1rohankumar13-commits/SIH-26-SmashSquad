"""Build the dashboard/event catalogue from labelled bust records."""

CATALOGUE_FIELDS = (
    "event_id", "init_time", "valid_time", "lead_day", "region_id",
    "latitude", "longitude", "forecast_value", "observed_value",
    "absolute_error", "bust_label", "category_tags", "source_ids",
)

# Preserve the established heavy-rainfall catalogue layout.  New mathematical
# fields are appended without renaming or removing the columns already used by
# the pilot CSV.
RAINFALL_CATALOGUE_FIELDS = (
    "event_id", "init_time", "valid_date", "lead_day", "region_id", "season",
    "mae_mm", "rmse_mm", "area_weighted_mae_mm", "area_weighted_rmse_mm",
    "bias_mm", "mean_memberwise_fss", "fss_error", "member_fss_min",
    "member_fss_max", "heavy_rain_miss_rate", "heavy_rain_false_alarm_rate",
    "event_error", "forecast_object_count", "observed_object_count",
    "matched_object_count", "mean_displacement_km", "timing_error_days",
    "critical_object_miss", "false_alarm_object", "critical_event_failure",
    "candidate_bust", "label_status", "strict_bust_label", "valid_grid_cells",
    "source_forecast", "source_observation",
)

RAINFALL_REQUIRED_IDENTITY_FIELDS = (
    "event_id", "init_time", "valid_date", "lead_day", "region_id",
    "source_forecast", "source_observation",
)

def build_catalogue(records):
    return [{field: row.get(field) for field in CATALOGUE_FIELDS} for row in records]


def build_rainfall_catalogue(records):
    """Return rows in the established rainfall-catalogue column order."""
    catalogue = []
    seen_event_ids = set()
    for index, row in enumerate(records):
        missing = [
            field
            for field in RAINFALL_REQUIRED_IDENTITY_FIELDS
            if row.get(field) in (None, "")
        ]
        if missing:
            raise ValueError(f"Rainfall record {index} is missing required fields: {missing}")
        event_id = row["event_id"]
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate rainfall event_id: {event_id}")
        seen_event_ids.add(event_id)
        catalogue.append(
            {field: row.get(field, "") for field in RAINFALL_CATALOGUE_FIELDS}
        )
    return catalogue
