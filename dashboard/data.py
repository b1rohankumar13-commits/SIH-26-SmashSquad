"""Read the per-hazard prediction exports written by scripts/export_dashboard_predictions.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

EXPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "dashboard"
DIRECTIONS = ("miss", "false_alarm")


@st.cache_data(show_spinner=False)
def load_index() -> dict | None:
    path = EXPORT_DIR / "index.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _arrays(hazard: str) -> dict[str, np.ndarray]:
    folder = EXPORT_DIR / hazard
    return {p.stem: np.load(p, mmap_mode="r") for p in folder.glob("*.npy")}


@st.cache_data(show_spinner=False)
def grid_centres() -> tuple[np.ndarray, np.ndarray]:
    return np.load(EXPORT_DIR / "lats.npy"), np.load(EXPORT_DIR / "lons.npy")


def grid_frame(hazard: str, init_pos: int, lead: int) -> pd.DataFrame:
    """One row per observed land cell for a grid hazard at (init, lead)."""
    a = _arrays(hazard)
    lats, lons = grid_centres()
    L = lead - 1
    obs_m = np.asarray(a["obs_miss"][init_pos, L])
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    keep = obs_m >= 0
    return pd.DataFrame({
        "latitude": lat[keep], "longitude": lon[keep],
        "p_miss": np.asarray(a["p_miss"][init_pos, L], np.float32)[keep],
        "p_fa": np.asarray(a["p_fa"][init_pos, L], np.float32)[keep],
        "obs_miss": obs_m[keep], "obs_fa": np.asarray(a["obs_fa"][init_pos, L])[keep],
        "ens_prob": np.asarray(a["ens_prob"][init_pos, L], np.float32)[keep],
    })


def grid_lead_summary(hazard: str, init_pos: int, tiers: dict) -> pd.DataFrame:
    """Per lead: cells flagged (high tier) per direction and cells that actually busted."""
    a = _arrays(hazard)
    rows = []
    for L in range(a["p_miss"].shape[1]):
        valid = np.asarray(a["obs_miss"][init_pos, L]) >= 0
        rec = {"lead_day": L + 1}
        for d, pk, ok in (("miss", "p_miss", "obs_miss"), ("false_alarm", "p_fa", "obs_fa")):
            p = np.asarray(a[pk][init_pos, L], np.float32)[valid]
            rec[f"{d}_flagged"] = int((p >= tiers[d]["high"]["threshold"]).sum())
            rec[f"{d}_observed"] = int((np.asarray(a[ok][init_pos, L])[valid] == 1).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def monsoon_arrays(init_pos: int) -> dict[str, np.ndarray]:
    a = _arrays("monsoon")
    return {k: np.asarray(v[init_pos], np.float32) for k, v in a.items()}
