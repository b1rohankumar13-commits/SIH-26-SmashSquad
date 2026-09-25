"""Historical forecast API routes."""

from fastapi import APIRouter

from api.schemas import HistoricalForecastResponse


router = APIRouter(prefix="/historical", tags=["historical"])


@router.get("/{init_date}", response_model=HistoricalForecastResponse)
def historical_forecast(init_date: str) -> HistoricalForecastResponse:
    return HistoricalForecastResponse(init_date=init_date)
