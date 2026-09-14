"""Typed API response contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


DataStatus = Literal["available", "not_generated", "invalid_data"]
GNNPublishStatus = Literal["stored", "published"]


class ForecastPoint(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    lead_day: int = Field(ge=1, le=10)
    overall_bust_probability: float = Field(ge=0, le=1)
    region_id: str | None = None
    run_id: str | None = None
    init_time: datetime | None = None
    valid_time: datetime | None = None
    category_tags: list[str] = Field(default_factory=list)


class CurrentForecastResponse(BaseModel):
    status: DataStatus
    source_file: str | None = None
    total_records: int = 0
    returned_records: int = 0
    message: str | None = None
    records: list[ForecastPoint] = Field(default_factory=list)


class CatalogueResponse(BaseModel):
    status: DataStatus
    items: list[str] = Field(default_factory=list)
    message: str | None = None


class HistoricalForecastResponse(BaseModel):
    init_date: str
    status: DataStatus = "not_generated"
    message: str = "No historical prediction output is available yet."


class GNNPredictionPoint(BaseModel):
    grid_id: str = Field(min_length=1)
    lead_day: int = Field(ge=1, le=10)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    gnn_probability: float = Field(ge=0, le=1)
    overall_bust_probability: float | None = Field(default=None, ge=0, le=1)
    region_id: str | None = None
    init_time: datetime | None = None
    valid_time: datetime | None = None
    category_tags: list[str] = Field(default_factory=list)


class GNNPublishRequest(BaseModel):
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    records: list[GNNPredictionPoint] = Field(min_length=1, max_length=50_000)


class GNNPublishResponse(BaseModel):
    status: GNNPublishStatus
    run_id: str
    model_id: str
    record_count: int
    dashboard_ready: bool
    stored_file: str
    dashboard_file: str | None = None
    message: str


class GNNStatusResponse(BaseModel):
    status: DataStatus
    latest_file: str | None = None
