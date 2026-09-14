"""FastAPI entry point for BustSentinel."""

from fastapi import FastAPI

from api.routes import forecast, gnn, historical, regions
from api.services.prediction_store import storage_state


app = FastAPI(
    title="BustSentinel Forecast Bust API",
    version="0.2.0",
    description="Local API for dashboard-ready forecast-bust outputs.",
)
app.include_router(forecast.router)
app.include_router(gnn.router)
app.include_router(regions.router)
app.include_router(historical.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "prediction_storage": storage_state()}
