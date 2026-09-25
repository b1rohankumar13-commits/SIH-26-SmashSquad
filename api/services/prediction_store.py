"""Read dashboard-ready prediction tables from local storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "current_predictions"
REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "lead_day",
}
PROBABILITY_COLUMNS = {"overall_bust_probability", "category_bust_probability"}
API_COLUMNS = [
    "latitude",
    "longitude",
    "lead_day",
    "overall_bust_probability",
    "category",
    "category_bust_probability",
    "model_id",
    "grid_id",
    "region_id",
    "run_id",
    "init_time",
    "valid_time",
    "category_tags",
]


@dataclass(frozen=True)
class PredictionSnapshot:
    status: Literal["available", "not_generated", "invalid_data"]
    frame: pd.DataFrame | None = None
    path: Path | None = None
    message: str | None = None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_current_snapshot(directory: Path | None = None) -> PredictionSnapshot:
    """Return the newest valid prediction table without fabricating values."""
    source_dir = directory or CURRENT_PREDICTIONS_DIR
    if not source_dir.exists():
        return PredictionSnapshot(
            "not_generated", message="Current prediction directory does not exist."
        )

    candidates = sorted(
        [*source_dir.glob("*.parquet"), *source_dir.glob("*.csv")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return PredictionSnapshot(
            "not_generated", message="No prediction table has been generated."
        )

    errors: list[str] = []
    for path in candidates:
        try:
            frame = _read_table(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            errors.append(f"{path.name}: missing {', '.join(sorted(missing))}")
            continue
        if not PROBABILITY_COLUMNS.intersection(frame.columns):
            errors.append(f"{path.name}: missing a bust probability column")
            continue
        if "category_bust_probability" in frame and "category" not in frame:
            errors.append(f"{path.name}: category is required for category probabilities")
            continue

        frame = frame.copy()
        numeric_columns = REQUIRED_COLUMNS | (PROBABILITY_COLUMNS & set(frame.columns))
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
        valid_probability = pd.Series(False, index=frame.index)
        for column in PROBABILITY_COLUMNS & set(frame.columns):
            valid_probability |= frame[column].between(0, 1)
        frame = frame[
            frame["latitude"].between(-90, 90)
            & frame["longitude"].between(-180, 180)
            & frame["lead_day"].between(1, 10)
            & valid_probability
        ]
        if "category_bust_probability" in frame:
            frame = frame[
                frame["category_bust_probability"].isna()
                | frame["category"].notna()
            ]
        if frame.empty:
            errors.append(f"{path.name}: no valid forecast rows")
            continue

        frame["lead_day"] = frame["lead_day"].astype(int)
        return PredictionSnapshot("available", frame=frame, path=path)

    return PredictionSnapshot(
        "invalid_data",
        message="; ".join(errors) or "Prediction tables could not be read.",
    )


def filter_predictions(
    frame: pd.DataFrame,
    *,
    region_id: str | None = None,
    run_id: str | None = None,
    lead_day: int | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Apply optional dashboard filters to a validated prediction table."""
    selected = frame
    for column, value in (("region_id", region_id), ("run_id", run_id), ("category", category)):
        if value is not None:
            if column not in selected.columns:
                return selected.iloc[0:0]
            selected = selected[selected[column].astype(str) == value]
    if lead_day is not None:
        selected = selected[selected["lead_day"] == lead_day]
    return selected


def catalogue_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "tolist") and getattr(value, "ndim", 0) == 1:
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def prediction_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [column for column in API_COLUMNS if column in frame.columns]
    records: list[dict[str, Any]] = []
    for row in frame[columns].to_dict(orient="records"):
        clean = {key: _json_value(value) for key, value in row.items()}
        tags = clean.get("category_tags")
        if isinstance(tags, str):
            clean["category_tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
        records.append(clean)
    return records


def storage_state() -> str:
    if not CURRENT_PREDICTIONS_DIR.exists():
        return "awaiting_data"
    has_table = any(CURRENT_PREDICTIONS_DIR.glob("*.parquet")) or any(
        CURRENT_PREDICTIONS_DIR.glob("*.csv")
    )
    return "ready" if has_table else "awaiting_data"
