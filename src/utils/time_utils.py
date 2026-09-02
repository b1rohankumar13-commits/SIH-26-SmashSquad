"""Forecast issue, lead, and valid-time helpers."""

from datetime import timedelta

def valid_time(init_time, lead_hours: int):
    return init_time + timedelta(hours=lead_hours)
