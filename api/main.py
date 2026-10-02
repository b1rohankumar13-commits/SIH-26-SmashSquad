"""FastAPI entry point."""

from fastapi import FastAPI
from api.routes import bulletin, dashboard, forecast, historical, live, regions

app = FastAPI(title="SIH Forecast Bust API", version="0.1.0")
app.include_router(forecast.router)
app.include_router(regions.router)
app.include_router(historical.router)
app.include_router(dashboard.router)
app.include_router(live.router)
app.include_router(bulletin.router)

@app.get("/health")
def health():
    return {"status": "ok"}
