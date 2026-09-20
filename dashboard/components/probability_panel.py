"""Interactive Plotly chart for forecast-bust probability by lead day."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOT_COLUMNS = {"lead_day"}


def build_lead_probability_figure(
    probability_data: pd.DataFrame | None,
    *,
    probability_column: str = "overall_bust_probability",
) -> go.Figure:
    """Build the Day 1–10 chart without manufacturing unavailable values."""
    figure = go.Figure()

    if probability_data is not None and not probability_data.empty:
        required = PLOT_COLUMNS | {probability_column}
        missing = required.difference(probability_data.columns)
        if missing:
            raise ValueError(
                f"Probability data is missing required columns: {sorted(missing)}"
            )

        frame = probability_data.copy()
        frame["lead_day"] = pd.to_numeric(frame["lead_day"], errors="coerce")
        frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
        frame = frame.dropna(subset=list(required))
        frame = frame[
            frame["lead_day"].between(1, 10)
            & frame[probability_column].between(0, 1)
        ]

        if not frame.empty:
            lead_summary = (
                frame.groupby("lead_day", as_index=False)[probability_column]
                .mean()
                .sort_values("lead_day")
            )
            lead_summary["probability_percent"] = (
                lead_summary[probability_column] * 100
            )
            figure.add_trace(
                go.Scatter(
                    x=lead_summary["lead_day"],
                    y=lead_summary["probability_percent"],
                    mode="lines+markers",
                    name="Mean grid-cell probability",
                    line={"color": "#ba643b", "width": 3},
                    marker={
                        "color": "#fffefa",
                        "line": {"color": "#ba643b", "width": 2},
                        "size": 8,
                    },
                    hovertemplate="Day %{x}<br>%{y:.1f}%<extra></extra>",
                )
            )

    if not figure.data:
        figure.add_annotation(
            text="No completed prediction run available",
            x=5.5,
            y=50,
            showarrow=False,
            font={"color": "#63737d", "size": 14},
            bgcolor="rgba(255,254,250,0.92)",
            borderpad=8,
        )

    figure.update_layout(
        height=285,
        margin={"l": 48, "r": 18, "t": 15, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffefa",
        hovermode="x unified",
        showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#63737d"},
        xaxis={
            "title": "Forecast lead day",
            "range": [0.7, 10.3],
            "tickmode": "linear",
            "dtick": 1,
            "gridcolor": "#e7ebe5",
            "linecolor": "#d8dedb",
            "fixedrange": False,
        },
        yaxis={
            "title": "Bust probability (%)",
            "range": [0, 100],
            "dtick": 20,
            "gridcolor": "#e7ebe5",
            "linecolor": "#d8dedb",
            "ticksuffix": "%",
            "fixedrange": False,
        },
    )
    return figure


def render_lead_probability_chart(
    probability_data: pd.DataFrame | None,
    *,
    probability_column: str = "overall_bust_probability",
) -> None:
    """Render the Plotly chart with a compact dashboard toolbar."""
    st.plotly_chart(
        build_lead_probability_figure(
            probability_data, probability_column=probability_column
        ),
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "scrollZoom": False,
        },
    )
