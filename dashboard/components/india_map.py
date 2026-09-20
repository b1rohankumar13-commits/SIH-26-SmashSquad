"""Interactive PyDeck map for gridded forecast-bust probabilities."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st


MAP_COLUMNS = {"latitude", "longitude"}


def _probability_colour(value: float) -> list[int]:
    """Return the dashboard risk colour for a probability in [0, 1]."""
    if value < 0.30:
        return [79, 155, 152, 190]
    if value < 0.60:
        return [213, 161, 74, 200]
    return [188, 98, 77, 215]


def build_india_deck(
    map_data: pd.DataFrame | None,
    *,
    probability_column: str = "overall_bust_probability",
) -> pdk.Deck:
    """Build a map that is empty until valid prediction rows are supplied."""
    layers: list[pdk.Layer] = []
    tooltip = None

    if map_data is not None and not map_data.empty:
        required = MAP_COLUMNS | {probability_column}
        missing = required.difference(map_data.columns)
        if missing:
            raise ValueError(f"Map data is missing required columns: {sorted(missing)}")

        frame = map_data.copy()
        frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame = frame.dropna(subset=list(required))
        frame = frame[frame[probability_column].between(0, 1)]
        frame["probability_percent"] = (frame[probability_column] * 100).round(1)
        frame["fill_colour"] = frame[probability_column].map(
            _probability_colour
        )

        if not frame.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=frame,
                    get_position="[longitude, latitude]",
                    get_fill_color="fill_colour",
                    get_line_color=[33, 53, 71, 95],
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
                    "backgroundColor": "#213547",
                    "color": "white",
                    "fontFamily": "Segoe UI, sans-serif",
                },
            }

    return pdk.Deck(
        # CARTO Voyager's warm light basemap matches the Forecast Atlas theme.
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
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
        parameters={"clearColor": [244, 240, 232, 255]},
    )


def render_india_map(
    map_data: pd.DataFrame | None,
    *,
    probability_column: str = "overall_bust_probability",
) -> None:
    """Render the interactive India map."""
    st.pydeck_chart(
        build_india_deck(map_data, probability_column=probability_column),
        width="stretch",
        height=430,
    )
