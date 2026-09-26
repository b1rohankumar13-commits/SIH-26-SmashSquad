"""Shared category-specific prediction and validation readers for the dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "current_predictions"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "model_validation"

CATEGORIES = (
    "Heavy rainfall",
    "Monsoon depression",
    "Cyclone",
    "Heat wave",
    "Western disturbance",
    "Active / break monsoon",
)
PROBABILITY_COLUMN = "category_bust_probability"
PREDICTION_COLUMNS = {
    "latitude", "longitude", "lead_day", "region_id", "run_id", "category",
    PROBABILITY_COLUMN,
}
VALIDATION_COLUMNS = {
    "category",
    "lead_band",
    "pr_auc",
    "brier_score",
    "recall",
    "false_alarm_ratio",
}


def _latest_tables(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [*directory.glob("*.parquet"), *directory.glob("*.csv")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


@st.cache_data(ttl=30, show_spinner=False)
def load_current_predictions() -> tuple[pd.DataFrame | None, Path | None]:
    """Read the newest dashboard-ready table through FastAPI or local storage."""
    api_url = os.getenv("BUSTSENTINEL_API_URL", "").rstrip("/")
    if api_url:
        try:
            response = requests.get(
                f"{api_url}/forecast/current", params={"limit": 50_000}, timeout=5
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "available" and payload.get("records"):
                frame = pd.DataFrame(payload["records"])
                if (PREDICTION_COLUMNS.issubset(frame.columns)
                    and payload.get("total_records") == payload.get("returned_records")):
                    return frame, Path(
                        payload.get("source_file") or "api_predictions.parquet"
                    )
        except (requests.RequestException, ValueError, TypeError):
            pass

    for path in _latest_tables(CURRENT_PREDICTIONS_DIR):
        try:
            frame = _read_table(path)
        except (OSError, ValueError):
            continue
        if PREDICTION_COLUMNS.issubset(frame.columns):
            return frame, path
    return None, None


def category_rows(frame: pd.DataFrame | None, category: str) -> pd.DataFrame:
    """Return valid final probabilities for exactly one weather category."""
    if frame is None or category not in CATEGORIES or not PREDICTION_COLUMNS.issubset(frame.columns):
        return pd.DataFrame()
    selected = frame[frame["category"].astype(str) == category].copy()
    selected[PROBABILITY_COLUMN] = pd.to_numeric(
        selected[PROBABILITY_COLUMN], errors="coerce"
    )
    selected = selected[selected[PROBABILITY_COLUMN].between(0, 1)]
    return selected


def category_summary(frame: pd.DataFrame | None, lead_day: int) -> dict[str, float | None]:
    """Mean grid-cell bust probability for each category at one lead day."""
    summary: dict[str, float | None] = {}
    for category in CATEGORIES:
        rows = category_rows(frame, category)
        if not rows.empty:
            rows = rows[pd.to_numeric(rows["lead_day"], errors="coerce") == lead_day]
        summary[category] = float(rows[PROBABILITY_COLUMN].mean()) if not rows.empty else None
    return summary


@st.cache_data(ttl=30, show_spinner=False)
def load_validation_results() -> tuple[pd.DataFrame | None, Path | None]:
    """Read completed per-category chronological-validation metrics, if present."""
    for path in _latest_tables(VALIDATION_DIR):
        try:
            frame = _read_table(path)
        except (OSError, ValueError):
            continue
        if VALIDATION_COLUMNS.issubset(frame.columns):
            return frame, path
    return None, None
