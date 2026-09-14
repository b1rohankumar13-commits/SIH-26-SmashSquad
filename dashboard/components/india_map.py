"""Interactive PyDeck map for gridded forecast-bust probabilities."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st


MAP_COLUMNS = {"latitude", "longitude", "overall_bust_probability"}


def _probability_colour(value: float) -> list[int]:
    """Return the dashboard risk colour for a probability in [0, 1]."""
    if value < 0.30:
        return [41, 158, 152, 185]
    if value < 0.60:
        return [233, 183, 72, 195]
    return [238, 104, 93, 205]


def build_india_deck(map_data: pd.DataFrame | None) -> pdk.Deck:
    """Build a map that is empty until valid prediction rows are supplied."""
    layers: list[pdk.Layer] = []
    tooltip = None

    if map_data is not None and not map_data.empty:
        missing = MAP_COLUMNS.difference(map_data.columns)
        if missing:
            raise ValueError(f"Map data is missing required columns: {sorted(missing)}")

        frame = map_data.copy()
        frame["overall_bust_probability"] = pd.to_numeric(
            frame["overall_bust_probability"], errors="coerce"
        )
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame = frame.dropna(subset=list(MAP_COLUMNS))
        frame = frame[frame["overall_bust_probability"].between(0, 1)]
        frame["probability_percent"] = (
            frame["overall_bust_probability"] * 100
        ).round(1)
        frame["fill_colour"] = frame["overall_bust_probability"].map(
            _probability_colour
        )

        if not frame.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=frame,
                    get_position="[longitude, latitude]",
                    get_fill_color="fill_colour",
                    get_line_color=[255, 255, 255, 70],
                    get_radius=17000,
                    radius_min_pixels=2,
                    radius_max_pixels=13,
                    line_width_min_pixels=0.4,
                    stroked=True,
                    filled=True,
                    pickable=True,
                    opacity=0.85,
                )
            )
            tooltip = {
                "html": (
                    "<b>Bust probability:</b> {probability_percent}%<br/>"
                    "<b>Location:</b> {latitude}, {longitude}"
                ),
                "style": {
                    "backgroundColor": "#14253d",
                    "color": "white",
                    "fontFamily": "Segoe UI, sans-serif",
                },
            }

    return pdk.Deck(
        # CARTO's dark basemap matches the dashboard sidebar while keeping
        # coastlines, state boundaries, labels, and risk markers legible.
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=22.4,
            longitude=80.8,
            zoom=3.55,
            min_zoom=3,
            max_zoom=9,
            pitch=0,
        ),
        layers=layers,
        tooltip=tooltip,
        parameters={"clearColor": [16, 31, 53, 255]},
    )


def render_india_map(map_data: pd.DataFrame | None) -> None:
    """Render the interactive India map."""
    st.pydeck_chart(
        build_india_deck(map_data),
        width="stretch",
        height=430,
    )
