"""Interactive Plotly chart for forecast-bust probability by lead day."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PLOT_COLUMNS = {"lead_day", "overall_bust_probability"}


def build_lead_probability_figure(
    probability_data: pd.DataFrame | None,
) -> go.Figure:
    """Build the Day 1–10 chart without manufacturing unavailable values."""
    figure = go.Figure()

    if probability_data is not None and not probability_data.empty:
        missing = PLOT_COLUMNS.difference(probability_data.columns)
        if missing:
            raise ValueError(
                f"Probability data is missing required columns: {sorted(missing)}"
            )

        frame = probability_data.copy()
        frame["lead_day"] = pd.to_numeric(frame["lead_day"], errors="coerce")
        frame["overall_bust_probability"] = pd.to_numeric(
            frame["overall_bust_probability"], errors="coerce"
        )
        frame = frame.dropna(subset=list(PLOT_COLUMNS))
        frame = frame[
            frame["lead_day"].between(1, 10)
            & frame["overall_bust_probability"].between(0, 1)
        ]

        if not frame.empty:
            lead_summary = (
                frame.groupby("lead_day", as_index=False)["overall_bust_probability"]
                .mean()
                .sort_values("lead_day")
            )
            lead_summary["probability_percent"] = (
                lead_summary["overall_bust_probability"] * 100
            )
            figure.add_trace(
                go.Scatter(
                    x=lead_summary["lead_day"],
                    y=lead_summary["probability_percent"],
                    mode="lines+markers",
                    name="Mean grid-cell probability",
                    line={"color": "#dc4c51", "width": 3},
                    marker={
                        "color": "#ffffff",
                        "line": {"color": "#dc4c51", "width": 2},
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
            font={"color": "#63748b", "size": 14},
            bgcolor="rgba(255,255,255,0.86)",
            borderpad=8,
        )

    figure.update_layout(
        height=285,
        margin={"l": 48, "r": 18, "t": 15, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#63748b"},
        xaxis={
            "title": "Forecast lead day",
            "range": [0.7, 10.3],
            "tickmode": "linear",
            "dtick": 1,
            "gridcolor": "#edf1f6",
            "linecolor": "#dce3ec",
            "fixedrange": False,
        },
        yaxis={
            "title": "Bust probability (%)",
            "range": [0, 100],
            "dtick": 20,
            "gridcolor": "#edf1f6",
            "linecolor": "#dce3ec",
            "ticksuffix": "%",
            "fixedrange": False,
        },
    )
    return figure


def render_lead_probability_chart(
    probability_data: pd.DataFrame | None,
) -> None:
    """Render the Plotly chart with a compact dashboard toolbar."""
    st.plotly_chart(
        build_lead_probability_figure(probability_data),
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "scrollZoom": False,
        },
    )
