"""Current forecast API routes."""

from fastapi import APIRouter, Query

from api.schemas import CatalogueResponse, CurrentForecastResponse
from api.services.prediction_store import (
    catalogue_values,
    filter_predictions,
    load_current_snapshot,
    prediction_records,
)


router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/current", response_model=CurrentForecastResponse)
def current_forecast(
    region_id: str | None = None,
    run_id: str | None = None,
    lead_day: int | None = Query(default=None, ge=1, le=10),
    limit: int = Query(default=10_000, ge=1, le=50_000),
) -> CurrentForecastResponse:
    snapshot = load_current_snapshot()
    if snapshot.frame is None:
        return CurrentForecastResponse(status=snapshot.status, message=snapshot.message)

    selected = filter_predictions(
        snapshot.frame,
        region_id=region_id,
        run_id=run_id,
        lead_day=lead_day,
    )
    limited = selected.head(limit)
    return CurrentForecastResponse(
        status="available",
        source_file=snapshot.path.name if snapshot.path else None,
        total_records=len(selected),
        returned_records=len(limited),
        records=prediction_records(limited),
    )


@router.get("/runs", response_model=CatalogueResponse)
def forecast_runs() -> CatalogueResponse:
    snapshot = load_current_snapshot()
    if snapshot.frame is None:
        return CatalogueResponse(status=snapshot.status, message=snapshot.message)
    return CatalogueResponse(
        status="available",
        items=catalogue_values(snapshot.frame, "run_id"),
    )
