"""Validate and publish externally generated GNN predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "gnn_predictions"
CURRENT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "current_predictions"
REQUIRED_COLUMNS = {
    "grid_id",
    "lead_day",
    "latitude",
    "longitude",
    "gnn_probability",
}


@dataclass(frozen=True)
class GNNPublishResult:
    branch_file: Path
    dashboard_file: Path | None
    record_count: int


def _safe_token(value: str, field: str) -> str:
    token = "".join(character for character in value if character.isalnum() or character in "-_")
    if not token:
        raise ValueError(f"{field} must contain a filename-safe character")
    return token


def validate_gnn_records(records: list[dict], *, run_id: str, model_id: str) -> pd.DataFrame:
    """Validate keyed probabilities supplied by an external GNN process."""
    frame = pd.DataFrame(records).copy()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"GNN records are missing columns: {sorted(missing)}")

    frame["run_id"] = run_id
    frame["model_id"] = model_id
    required_numeric = ["lead_day", "latitude", "longitude", "gnn_probability"]
    numeric = list(required_numeric)
    if "overall_bust_probability" in frame:
        numeric.append("overall_bust_probability")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required_numeric].isna().any().any():
        raise ValueError("GNN probability and coordinate fields must be numeric")

    if frame.duplicated(["run_id", "lead_day", "grid_id"]).any():
        raise ValueError("Duplicate run_id, lead_day and grid_id keys")
    if not np.all(frame["lead_day"] == np.floor(frame["lead_day"])):
        raise ValueError("lead_day must contain integers")
    if not frame["lead_day"].between(1, 10).all():
        raise ValueError("lead_day must be between 1 and 10")
    if not frame["latitude"].between(-90, 90).all() or not frame["longitude"].between(-180, 180).all():
        raise ValueError("Coordinates are outside valid geographic bounds")
    if not frame["gnn_probability"].between(0, 1).all():
        raise ValueError("gnn_probability must be within [0, 1]")

    coordinate_counts = frame.groupby("grid_id")[["latitude", "longitude"]].nunique()
    if (coordinate_counts > 1).any().any():
        raise ValueError("Each grid_id must map to one coordinate")

    if "overall_bust_probability" in frame:
        supplied = frame["overall_bust_probability"].notna()
        if supplied.any() and not supplied.all():
            raise ValueError("overall_bust_probability must be supplied for every record or none")
        if supplied.all() and not frame["overall_bust_probability"].between(0, 1).all():
            raise ValueError("overall_bust_probability must be within [0, 1]")
    frame["lead_day"] = frame["lead_day"].astype(int)
    return frame


def _write_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.stem}.{uuid4().hex}.tmp.parquet"
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_gnn_records(
    records: list[dict],
    *,
    run_id: str,
    model_id: str,
    branch_directory: Path | None = None,
    dashboard_directory: Path | None = None,
) -> GNNPublishResult:
    """Store GNN output and publish final probabilities when supplied."""
    frame = validate_gnn_records(records, run_id=run_id, model_id=model_id)
    run_token = _safe_token(run_id, "run_id")
    branch_dir = branch_directory or GNN_PREDICTIONS_DIR
    dashboard_dir = dashboard_directory or CURRENT_PREDICTIONS_DIR
    branch_file = branch_dir / f"gnn_predictions_{run_token}.parquet"
    dashboard_ready = (
        "overall_bust_probability" in frame
        and frame["overall_bust_probability"].notna().all()
    )
    dashboard_file = (
        dashboard_dir / f"bust_probabilities_{run_token}.parquet"
        if dashboard_ready
        else None
    )
    for target in (branch_file, dashboard_file):
        if target is not None and target.exists():
            raise FileExistsError(f"Output already exists: {target.name}")

    _write_parquet(frame, branch_file)
    if dashboard_file is not None:
        _write_parquet(frame, dashboard_file)
    return GNNPublishResult(branch_file, dashboard_file, len(frame))


def latest_gnn_file(directory: Path | None = None) -> Path | None:
    source = directory or GNN_PREDICTIONS_DIR
    files = list(source.glob("*.parquet")) if source.exists() else []
    return max(files, key=lambda path: path.stat().st_mtime) if files else None
