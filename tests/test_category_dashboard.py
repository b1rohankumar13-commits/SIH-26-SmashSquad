"""Category selection must not mix outputs from different GNNs."""

import pandas as pd

from dashboard.category_data import CATEGORIES, category_rows, category_summary
from dashboard.components.probability_panel import build_lead_probability_figure


def test_category_summary_and_chart_follow_selected_category():
    frame = pd.DataFrame(
        {
            "latitude": [20.0] * 4,
            "longitude": [85.0] * 4,
            "lead_day": [1, 2, 1, 2],
            "region_id": ["odisha"] * 4,
            "run_id": ["run-1"] * 4,
            "category": ["Heavy rainfall", "Heavy rainfall", "Cyclone", "Cyclone"],
            "category_bust_probability": [0.8, 0.6, 0.2, 0.1],
        }
    )
    summary = category_summary(frame, 1)
    assert summary["Heavy rainfall"] == 0.8
    assert summary["Cyclone"] == 0.2
    assert len(summary) == len(CATEGORIES) == 6
    assert summary["Heat wave"] is None

    cyclone = category_rows(frame, "Cyclone")
    figure = build_lead_probability_figure(
        cyclone, probability_column="category_bust_probability"
    )
    assert list(figure.data[0].y) == [20.0, 10.0]
