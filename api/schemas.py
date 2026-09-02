"""API response contracts."""

from datetime import datetime
from pydantic import BaseModel, Field

class BustForecast(BaseModel):
    init_time: datetime
    valid_time: datetime
    lead_day: int = Field(ge=1, le=10)
    overall_bust_probability: float = Field(ge=0, le=1)
    category_tags: list[str] = []
