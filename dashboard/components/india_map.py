"""PyDeck India map: directional bust-risk cells, GEFS event probability, observed busts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

GRID_BOUNDS = {"west": 65.25, "south": 5.25, "east": 99.75, "north": 37.75}
FRAME_PX = 560
HALF = 0.25  # half a 0.5-degree cell

# Direction palettes: miss = GEFS under-calling, the dangerous case (orange/red, draws attention);
# false alarm = over-calling, lower stakes (blue).
COLOURS = {
    "miss": {"elevated": [255, 176, 80, 160], "high": [240, 84, 40, 230], "observed": [255, 214, 120, 255]},
    "false_alarm": {"elevated": [120, 160, 255, 140], "high": [70, 110, 235, 210], "observed": [170, 205, 255, 255]},
}
MCZ_BOX = {"south": 18.0, "north": 28.0, "west": 73.0, "east": 86.0}


def _mercator_y(lat: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _fit_view(bounds: dict, size_px: int = FRAME_PX, pad: float = 0.04) -> dict:
    span_x = (bounds["east"] - bounds["west"]) / 360.0
    span_y = (_mercator_y(bounds["north"]) - _mercator_y(bounds["south"])) / (2 * math.pi)
    zoom = min(math.log2(size_px * (1 - pad) / (512 * span_x)),
               math.log2(size_px * (1 - pad) / (512 * span_y)))
    mid_y = (_mercator_y(bounds["south"]) + _mercator_y(bounds["north"])) / 2
    return {"latitude": math.degrees(2 * math.atan(math.exp(mid_y)) - math.pi / 2),
            "longitude": (bounds["west"] + bounds["east"]) / 2, "zoom": zoom}


def _square(lat: float, lon: float) -> list[list[float]]:
    return [[lon - HALF, lat - HALF], [lon + HALF, lat - HALF], [lon + HALF, lat + HALF], [lon - HALF, lat + HALF]]


def _tier(p: np.ndarray, t: dict) -> np.ndarray:
    elevated = t["elevated"]["threshold"] if t.get("elevated") else np.inf
    return np.where(p >= t["high"]["threshold"], 2, np.where(p >= elevated, 1, 0))


def classify(frame: pd.DataFrame, tiers: dict, mode: str) -> pd.DataFrame:
    """Add tier/direction/colour columns. mode in {'miss', 'false_alarm', 'both'}."""
    f = frame.copy()
    f["miss_tier"] = _tier(f["p_miss"].to_numpy(), tiers["miss"])
    f["fa_tier"] = _tier(f["p_fa"].to_numpy(), tiers["false_alarm"])
    if mode == "miss":
        f["tier"], f["direction"] = f["miss_tier"], "miss"
    elif mode == "false_alarm":
        f["tier"], f["direction"] = f["fa_tier"], "false_alarm"
    else:  # stronger tier wins; ties go to the larger lift over its own threshold
        rel_m = f["p_miss"] / tiers["miss"]["high"]["threshold"]
        rel_f = f["p_fa"] / tiers["false_alarm"]["high"]["threshold"]
        use_miss = (f["miss_tier"] > f["fa_tier"]) | ((f["miss_tier"] == f["fa_tier"]) & (rel_m >= rel_f))
        f["tier"] = np.where(use_miss, f["miss_tier"], f["fa_tier"])
        f["direction"] = np.where(use_miss, "miss", "false_alarm")
    names = {0: "", 1: "elevated", 2: "high"}
    f["tier_name"] = f["tier"].map(names)
    f["fill"] = [COLOURS[d][names[t]] if t else [0, 0, 0, 0] for d, t in zip(f["direction"], f["tier"])]
    return f


def build_deck(frame: pd.DataFrame | None, tiers: dict | None, mode: str, *,
               show_gefs: bool, show_observed: bool, region_box: dict | None = None,
               region_fill: list[int] | None = None) -> pdk.Deck:
    layers: list[pdk.Layer] = []
    tooltip = None
    if frame is not None and not frame.empty and tiers is not None:
        f = classify(frame, tiers, mode)
        f["polygon"] = [_square(a, b) for a, b in zip(f["latitude"], f["longitude"])]
        f["miss_pct"] = [f"{100 * v:.1f}" for v in f["p_miss"]]
        f["fa_pct"] = [f"{100 * v:.1f}" for v in f["p_fa"]]
        f["gefs_pct"] = [f"{100 * v:.0f}" for v in f["ens_prob"]]
        f["lat_s"] = [f"{v:.2f}" for v in f["latitude"]]
        f["lon_s"] = [f"{v:.2f}" for v in f["longitude"]]
        f["outcome"] = np.where(f["obs_miss"] == 1, "MISS (event happened, GEFS said no)",
                                np.where(f["obs_fa"] == 1, "FALSE ALARM (GEFS said yes, no event)", "forecast held"))
        if show_gefs:
            g = f[f["ens_prob"] > 0].copy()
            g["gfill"] = [[46, 170, 150, int(40 + 150 * p)] for p in g["ens_prob"]]
            layers.append(pdk.Layer("PolygonLayer", data=g, get_polygon="polygon", get_fill_color="gfill",
                                    stroked=False, pickable=False))
        flagged = f[f["tier"] > 0]
        layers.append(pdk.Layer("PolygonLayer", data=f, get_polygon="polygon", get_fill_color=[0, 0, 0, 0],
                                stroked=False, pickable=True))
        layers.append(pdk.Layer("PolygonLayer", data=flagged, get_polygon="polygon", get_fill_color="fill",
                                get_line_color=[255, 255, 255, 60], line_width_min_pixels=0.3,
                                stroked=True, pickable=True))
        if show_observed:
            dirs = ["miss", "false_alarm"] if mode == "both" else [mode]
            for d in dirs:
                col = "obs_miss" if d == "miss" else "obs_fa"
                o = f[f[col] == 1]
                if not o.empty:
                    layers.append(pdk.Layer("ScatterplotLayer", data=o, get_position="[longitude, latitude]",
                                            get_radius=11000, radius_min_pixels=2, radius_max_pixels=8,
                                            filled=False, stroked=True, get_line_color=COLOURS[d]["observed"],
                                            line_width_min_pixels=1.4, pickable=False))
        tooltip = {"html": "<b>{lat_s}N, {lon_s}E</b><br/>"
                           "Miss risk: {miss_pct}% &nbsp; False-alarm risk: {fa_pct}%<br/>"
                           "GEFS event probability: {gefs_pct}%<br/>Outcome: {outcome}",
                   "style": {"backgroundColor": "#14253d", "color": "white", "fontFamily": "Segoe UI, sans-serif"}}
    if region_box is not None:
        b = region_box
        box = pd.DataFrame({"polygon": [[[b["west"], b["south"]], [b["east"], b["south"]],
                                         [b["east"], b["north"]], [b["west"], b["north"]]]]})
        layers.append(pdk.Layer("PolygonLayer", data=box, get_polygon="polygon",
                                get_fill_color=region_fill or [120, 150, 255, 60],
                                get_line_color=[220, 230, 255, 220], line_width_min_pixels=2, stroked=True))
    view = _fit_view(GRID_BOUNDS)
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=view["latitude"], longitude=view["longitude"],
                                          zoom=view["zoom"], min_zoom=view["zoom"], max_zoom=9, pitch=0),
        layers=layers, tooltip=tooltip, parameters={"clearColor": [16, 31, 53, 255]})


def render_map(deck: pdk.Deck) -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stDeckGlJsonChart"] > div {{ max-width: {FRAME_PX}px; margin-inline: auto; }}
        [data-testid="stDeckGlJsonChart"] .mapboxgl-ctrl-group {{ display: none !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.pydeck_chart(deck, width="stretch", height=FRAME_PX)
