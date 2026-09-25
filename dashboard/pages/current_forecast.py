"""BustSentinel forecast-confidence dashboard."""

<<<<<<< HEAD
import pandas as pd
import streamlit as st

from category_data import (
    CATEGORIES,
    PROBABILITY_COLUMN,
    category_rows,
    category_summary,
    load_current_predictions,
)
from components.india_map import render_india_map
from components.probability_panel import render_lead_probability_chart

=======
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from components.india_map import render_india_map
from components.probability_panel import render_lead_probability_chart


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "current_predictions"
# Only surface the leads the model has reliable skill at; beyond Day 5 it
# systematically under-calls the late-lead bust resurgence.
MAX_LEAD_DAY = 5
PREDICTION_COLUMNS = {
    "latitude",
    "longitude",
    "lead_day",
    "overall_bust_probability",
}


@st.cache_data(ttl=30, show_spinner=False)
def load_current_predictions() -> tuple[pd.DataFrame | None, Path | None]:
    """Load predictions through FastAPI, with a local-development fallback."""
    api_url = os.getenv("BUSTSENTINEL_API_URL", "").rstrip("/")
    if api_url:
        try:
            response = requests.get(f"{api_url}/forecast/current", timeout=5)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "available" and payload.get("records"):
                return pd.DataFrame(payload["records"]), Path(
                    payload.get("source_file") or "api_predictions.parquet"
                )
            return None, None
        except (requests.RequestException, ValueError, TypeError):
            pass

    if not CURRENT_PREDICTIONS_DIR.exists():
        return None, None

    candidates = sorted(
        [
            *CURRENT_PREDICTIONS_DIR.glob("*.parquet"),
            *CURRENT_PREDICTIONS_DIR.glob("*.csv"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            frame = (
                pd.read_parquet(path)
                if path.suffix.lower() == ".parquet"
                else pd.read_csv(path)
            )
        except (OSError, ValueError):
            continue
        if PREDICTION_COLUMNS.issubset(frame.columns):
            return frame, path
    return None, None


>>>>>>> origin/main
prediction_data, prediction_path = load_current_predictions()

st.markdown(
    """
    <style>
    :root {
<<<<<<< HEAD
      --page: var(--bs-page);
      --surface: var(--bs-surface);
      --nav: var(--bs-nav);
      --nav-active: #ba643b;
      --ink: var(--bs-ink);
      --muted: var(--bs-muted);
      --line: var(--bs-line);
      --blue: var(--bs-blue);
      --blue-soft: var(--bs-soft);
=======
      --page: #f1f4f8;
      --surface: #ffffff;
      --nav: #101f35;
      --nav-active: #243b5d;
      --ink: #14253d;
      --muted: #63748b;
      --line: #dce3ec;
      --blue: #245be8;
      --blue-soft: #eaf0fd;
>>>>>>> origin/main
    }

    .stApp {
      background: var(--page);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stSidebar"] {
      background: var(--nav);
      border-right: 0;
    }
    @media (min-width: 901px) {
      section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        min-width: 14.0625rem !important;
        width: 14.0625rem !important;
      }
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapseButton"] {
        display: none !important;
      }
    }
    @media (max-width: 900px) {
      [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        position: fixed !important;
        top: .65rem !important;
        left: .65rem !important;
        z-index: 100000 !important;
<<<<<<< HEAD
        border: 1px solid rgba(230, 173, 131, .55) !important;
=======
        border: 1px solid rgba(126, 172, 255, .55) !important;
>>>>>>> origin/main
        border-radius: .5rem !important;
        background: var(--nav) !important;
        box-shadow: 0 .35rem 1rem rgba(9, 22, 40, .22) !important;
      }
      [data-testid="stSidebarCollapsedControl"] button {
        color: #ffffff !important;
      }
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.7rem; }
<<<<<<< HEAD
    [data-testid="stSidebar"] * { color: #edf3f2; }
    [data-testid="stSidebar"] hr {
      border-color: rgba(237, 243, 242, .2);
=======
    [data-testid="stSidebar"] * { color: #bdcadb; }
    [data-testid="stSidebar"] hr {
      border-color: rgba(189, 202, 219, .16);
>>>>>>> origin/main
      margin: 1.45rem 0;
    }
    .block-container {
      max-width: 1700px;
      padding: 1.2rem 2rem 3rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: .7rem;
      color: #ffffff;
      font-size: 1.38rem;
      font-weight: 700;
      letter-spacing: -.04em;
    }
    .brand-icon {
      position: relative;
      display: grid;
      place-items: center;
      width: 2.35rem;
      height: 2.35rem;
      border: 1px solid rgba(126, 172, 255, .72);
      border-radius: .7rem;
      color: #ffffff;
      background: linear-gradient(145deg, #2f66d0, #1b3f76);
      box-shadow: 0 .45rem 1.2rem rgba(5, 14, 28, .32);
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0;
    }
    .brand-icon::after {
      content: "";
      position: absolute;
      top: .38rem;
      right: .38rem;
      width: .35rem;
      height: .35rem;
      border: 2px solid #9bc1ff;
      border-radius: 50%;
    }
    .subbrand {
      margin-top: .15rem;
      color: #8a9bb1;
      font-size: .75rem;
      letter-spacing: .15em;
    }
    .nav-label {
      color: #7e91ad;
      font-size: .75rem;
      letter-spacing: .13em;
      margin-bottom: .6rem;
    }
    .nav-item {
      margin: .28rem 0;
      border-radius: .35rem;
      color: #bac9dc;
      padding: .72rem .8rem;
      font-size: .9rem;
    }
    .nav-item.active {
      border-left: 3px solid #7eacff;
      background: var(--nav-active);
      color: #ffffff;
    }
    .nav-icon { color: #8ba9d4; margin-right: .65rem; }

    .topbar {
      margin: -1.2rem -2rem 1.5rem;
      padding: 1.15rem 2rem;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    .topbar-title {
      color: var(--ink);
      font-size: .95rem;
      font-weight: 650;
    }
    .title-row { margin-bottom: 1.15rem; }
    .title-row h1 {
      color: var(--ink);
<<<<<<< HEAD
      font-family: "Lora", Georgia, "Times New Roman", serif;
=======
>>>>>>> origin/main
      font-size: 1.78rem;
      line-height: 1.25;
      letter-spacing: -.035em;
      margin: 0;
    }
    .title-row p {
      color: var(--muted);
      font-size: .96rem;
      margin: .3rem 0 0;
    }

    [data-testid="stSelectbox"] label p {
      color: var(--muted) !important;
      font-size: .74rem !important;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    [data-baseweb="select"] > div {
      background: var(--surface);
      border-color: var(--line);
      min-height: 2.6rem;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .9rem;
      margin: .3rem 0 1.1rem;
    }
    .card {
      overflow: hidden;
      border: 1px solid var(--line);
<<<<<<< HEAD
      border-radius: .5rem;
      background: var(--surface);
      box-shadow: 0 3px 11px rgba(21, 63, 67, .06);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .card:hover {
      transform: translateY(-3px);
      box-shadow: 0 10px 24px rgba(21, 63, 67, .14);
=======
      border-radius: .65rem;
      background: var(--surface);
>>>>>>> origin/main
    }
    .stat-card { min-height: 8.6rem; padding: 1rem 1.15rem; }
    .eyebrow {
      color: var(--muted);
      font-size: .73rem;
      font-weight: 600;
      letter-spacing: .07em;
      text-transform: uppercase;
    }
    .metric {
      color: var(--ink);
      font-size: 2.05rem;
      font-weight: 650;
      letter-spacing: -.04em;
      margin: .3rem 0;
    }
    .metric-copy { color: var(--muted); font-size: .8rem; line-height: 1.45; }

    .card-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: .95rem 1.1rem;
      border-bottom: 1px solid var(--line);
    }
<<<<<<< HEAD
    .card-head h2 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 1.05rem; margin: 0; }
=======
    .card-head h2 { color: var(--ink); font-size: 1.02rem; margin: 0; }
>>>>>>> origin/main
    .card-subtitle { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
    .card-body { padding: 1rem 1.1rem; }
    .status-pill {
      flex: 0 0 auto;
      border-radius: .3rem;
<<<<<<< HEAD
      background: var(--bs-soft);
      color: #874327;
=======
      background: #f3f6fc;
      color: #526c96;
>>>>>>> origin/main
      padding: .24rem .55rem;
      font-size: .72rem;
      font-weight: 650;
    }

    .map-panel {
      position: relative;
      min-height: 27rem;
      display: grid;
      place-items: center;
      padding: 2rem;
<<<<<<< HEAD
      color: #edf3f2;
      text-align: center;
      background:
        linear-gradient(rgba(237, 243, 242, .16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(237, 243, 242, .16) 1px, transparent 1px),
        #142b42;
=======
      color: #d8e5f2;
      text-align: center;
      background:
        linear-gradient(rgba(91, 121, 151, .16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(91, 121, 151, .16) 1px, transparent 1px),
        #101f32;
>>>>>>> origin/main
      background-size: 3.4rem 3.4rem;
    }
    .map-panel::after {
      content: "BAY OF BENGAL";
      position: absolute;
      right: 1.3rem;
      bottom: 1rem;
<<<<<<< HEAD
      color: #bdd0d0;
=======
      color: #627e9c;
>>>>>>> origin/main
      font-size: .85rem;
      letter-spacing: .18em;
    }
    .empty-symbol {
      width: 2.8rem;
      height: 2.8rem;
      display: grid;
      place-items: center;
      margin: 0 auto .8rem;
<<<<<<< HEAD
      border: 1px solid #68889a;
      border-radius: .55rem;
      color: #f0be96;
=======
      border: 1px solid #47627d;
      border-radius: .55rem;
      color: #8fb7df;
>>>>>>> origin/main
      background: rgba(39, 65, 91, .72);
    }
    .empty-title { color: inherit; font-size: .94rem; font-weight: 650; }
    .empty-copy {
      max-width: 27rem;
      margin: .3rem auto 0;
<<<<<<< HEAD
      color: #bdd0d0;
=======
      color: #9fb3c7;
>>>>>>> origin/main
      font-size: .82rem;
      line-height: 1.5;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      padding: .75rem 1.1rem;
      color: var(--muted);
      font-size: .75rem;
    }
    .dot {
      display: inline-block;
      width: .55rem;
      height: .55rem;
      margin-right: .35rem;
      border-radius: .12rem;
    }
    .timeline-empty {
      border-top: 1px solid var(--line);
      padding: .85rem 1.1rem;
      color: var(--muted);
      font-size: .8rem;
    }

    .diagnostic-summary {
      display: flex;
      align-items: center;
      gap: .75rem;
      margin-bottom: .85rem;
      padding: .8rem;
<<<<<<< HEAD
      border: 1px solid var(--line);
      border-radius: .5rem;
      background: #f8f4ec;
=======
      border: 1px solid #dfe7f3;
      border-radius: .5rem;
      background: #f7f9fd;
>>>>>>> origin/main
    }
    .diagnostic-summary-icon {
      display: grid;
      place-items: center;
      width: 2.2rem;
      height: 2.2rem;
      flex: 0 0 auto;
      border-radius: .45rem;
      background: var(--blue-soft);
      color: var(--blue);
      font-weight: 750;
    }
    .diagnostic-summary b { display: block; color: var(--ink); font-size: .84rem; }
    .diagnostic-summary span { color: var(--muted); font-size: .76rem; }
    .diagnostic-grid { display: grid; gap: .55rem; }
    .diagnostic-item {
      display: grid;
      grid-template-columns: 2rem minmax(0, 1fr) auto;
      align-items: center;
      gap: .6rem;
      padding: .7rem;
<<<<<<< HEAD
      border: 1px solid var(--line);
=======
      border: 1px solid #edf0f5;
>>>>>>> origin/main
      border-radius: .45rem;
    }
    .diagnostic-symbol {
      display: grid;
      place-items: center;
      width: 2rem;
      height: 2rem;
      border-radius: .4rem;
<<<<<<< HEAD
      background: var(--bs-soft);
      color: #874327;
=======
      background: #f3f6fb;
      color: #59739a;
>>>>>>> origin/main
      font-size: .82rem;
    }
    .diagnostic-title { color: var(--ink); font-size: .82rem; font-weight: 650; }
    .diagnostic-copy { color: var(--muted); font-size: .74rem; margin-top: .1rem; }
    .diagnostic-state {
<<<<<<< HEAD
      color: var(--muted);
=======
      color: #71839b;
>>>>>>> origin/main
      font-size: .68rem;
      font-weight: 650;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .empty-line {
      height: .45rem;
      border-radius: 1rem;
<<<<<<< HEAD
      margin: .8rem 0;
      background: linear-gradient(90deg, #e4dfd3 25%, #f1ede2 37%, #e4dfd3 63%);
      background-size: 400% 100%;
      animation: shimmer 1.4s ease infinite;
    }
    .empty-line.short { width: 64%; }
    .skeleton {
      display: inline-block;
      vertical-align: middle;
      height: 1.9rem;
      width: 4.5rem;
      border-radius: .3rem;
      background: linear-gradient(90deg, #e4dfd3 25%, #f1ede2 37%, #e4dfd3 63%);
      background-size: 400% 100%;
      animation: shimmer 1.4s ease infinite;
    }
    .skeleton.skeleton-sm { height: 1.05rem; width: 3.4rem; border-radius: .25rem; }
    @keyframes shimmer {
      0% { background-position: 100% 50%; }
      100% { background-position: 0 50%; }
    }
=======
      background: #edf1f7;
      margin: .8rem 0;
    }
    .empty-line.short { width: 64%; }
>>>>>>> origin/main
    .key-value {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      padding: .5rem 0;
      color: var(--muted);
      font-size: .82rem;
    }
    .key-value b { color: var(--ink); font-weight: 600; }
<<<<<<< HEAD
    .category-risk-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 4rem 4.5rem;
      align-items: center;
      gap: .4rem;
      padding: .55rem .35rem;
      border-bottom: 1px solid var(--line);
      color: var(--ink);
      font-size: .75rem;
    }
    .category-risk-row span:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
    .category-risk-row.heading { color: var(--muted); font-size: .68rem; font-weight: 650; text-transform: uppercase; }
    .category-risk-row.selected { background: var(--bs-soft); border-left: 3px solid var(--blue); }
=======
>>>>>>> origin/main
    .chart-empty {
      min-height: 10.5rem;
      display: grid;
      place-items: center;
<<<<<<< HEAD
      border-left: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
=======
      border-left: 1px solid #e7ebf1;
      border-bottom: 1px solid #e7ebf1;
>>>>>>> origin/main
      color: var(--muted);
      text-align: center;
      font-size: .82rem;
      background:
<<<<<<< HEAD
        linear-gradient(#e4e8e2 1px, transparent 1px),
        linear-gradient(90deg, #e4e8e2 1px, transparent 1px);
=======
        linear-gradient(#edf1f6 1px, transparent 1px),
        linear-gradient(90deg, #edf1f6 1px, transparent 1px);
>>>>>>> origin/main
      background-size: 3rem 2.4rem;
    }
    .section-gap { margin-top: 1rem; }

    @media (max-width: 1100px) {
      .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
      .block-container { padding: .8rem 1rem 2rem; }
      .topbar { margin: -.8rem -1rem 1.2rem; padding: 1rem; }
      .stats-grid { grid-template-columns: 1fr; }
      .title-row h1 { font-size: 1.5rem; }
      .map-panel { min-height: 20rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
      <div class="topbar-title">Forecast intelligence / Current forecast</div>
      <span class="bs-brand-mark" aria-label="BustSentinel">B</span>
    </div>
    <div class="title-row">
<<<<<<< HEAD
      <h1>Flag The Forecast Before It Fails</h1>
=======
      <h1>Where could the forecast fail?</h1>
      <p>Inspect uncertainty before the next forecast decision.</p>
>>>>>>> origin/main
    </div>
    """,
    unsafe_allow_html=True,
)

has_predictions = prediction_data is not None and not prediction_data.empty
working_data = prediction_data.copy() if has_predictions else pd.DataFrame()

region_options = (
<<<<<<< HEAD
    ["All regions", *sorted(working_data["region_id"].dropna().astype(str).unique().tolist())]
=======
    sorted(working_data["region_id"].dropna().astype(str).unique().tolist())
>>>>>>> origin/main
    if has_predictions and "region_id" in working_data.columns
    else ["All regions"]
)
run_options = (
    sorted(working_data["run_id"].dropna().astype(str).unique().tolist(), reverse=True)
    if has_predictions and "run_id" in working_data.columns
    else ([prediction_path.stem] if prediction_path else ["No completed runs available"])
)

control_columns = st.columns([1, 1, 1.25, 1.45, .7], gap="medium")
with control_columns[0]:
    selected_region = st.selectbox("Region", region_options, index=0)
with control_columns[1]:
    st.selectbox("Geographic view", ["Regional overview", "0.5° grid"], index=0)
with control_columns[2]:
<<<<<<< HEAD
    selected_category = st.selectbox("Weather category", CATEGORIES, index=0)
=======
    st.selectbox(
        "Weather category",
        [
            "All categories",
            "Heavy rainfall",
            "Monsoon depression",
            "Cyclone",
            "Heat wave",
            "Western disturbance",
            "Active / break monsoon",
        ],
        index=0,
    )
>>>>>>> origin/main
with control_columns[3]:
    selected_run = st.selectbox(
        "Forecast initialization",
        run_options,
        disabled=not has_predictions,
    )

if has_predictions and "run_id" in working_data.columns:
    working_data = working_data[working_data["run_id"].astype(str) == selected_run]
<<<<<<< HEAD
if has_predictions and "region_id" in working_data.columns and selected_region != "All regions":
    working_data = working_data[
        working_data["region_id"].astype(str) == selected_region
    ]

category_data = category_rows(working_data, selected_category)
lead_options = (
    sorted(
        pd.to_numeric(category_data["lead_day"], errors="coerce")
=======
if has_predictions and "region_id" in working_data.columns:
    working_data = working_data[
        working_data["region_id"].astype(str) == selected_region
    ]
if has_predictions:
    working_data = working_data[
        pd.to_numeric(working_data["lead_day"], errors="coerce") <= MAX_LEAD_DAY
    ]

lead_options = (
    sorted(
        pd.to_numeric(working_data["lead_day"], errors="coerce")
>>>>>>> origin/main
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
<<<<<<< HEAD
    if not category_data.empty
=======
    if has_predictions
>>>>>>> origin/main
    else [1]
)
with control_columns[4]:
    selected_lead = st.selectbox(
        "Lead day",
        lead_options,
        format_func=lambda value: f"Day {value}",
<<<<<<< HEAD
        disabled=category_data.empty,
    )

map_data = (
    category_data[
        pd.to_numeric(category_data["lead_day"], errors="coerce") == selected_lead
    ].copy()
    if not category_data.empty
    else None
)
valid_probability = (
    pd.to_numeric(map_data[PROBABILITY_COLUMN], errors="coerce")
=======
        disabled=not has_predictions,
    )

map_data = (
    working_data[
        pd.to_numeric(working_data["lead_day"], errors="coerce") == selected_lead
    ].copy()
    if has_predictions
    else None
)
valid_probability = (
    pd.to_numeric(map_data["overall_bust_probability"], errors="coerce")
>>>>>>> origin/main
    .dropna()
    .loc[lambda values: values.between(0, 1)]
    if map_data is not None and not map_data.empty
    else pd.Series(dtype=float)
)

mean_probability = valid_probability.mean() if not valid_probability.empty else None
<<<<<<< HEAD
all_category_probabilities = category_summary(working_data, selected_lead)
category_rows_html = "".join(
    '<div class="category-risk-row{}"><span>{}</span><span>{}</span><span>{}</span></div>'.format(
        " selected" if category == selected_category else "",
        category,
        f"{probability:.1%}" if probability is not None else "—",
        f"{1 - probability:.1%}" if probability is not None else "—",
    )
    for category, probability in all_category_probabilities.items()
)
_skeleton = '<span class="skeleton"></span>'
probability_text = f"{mean_probability:.1%}" if mean_probability is not None else _skeleton
confidence_text = f"{1 - mean_probability:.1%}" if mean_probability is not None else _skeleton
lead_text = f"Day {selected_lead}" if not category_data.empty else _skeleton
coverage_text = f"{len(valid_probability):,} cells" if not valid_probability.empty else _skeleton
=======
probability_text = f"{mean_probability:.1%}" if mean_probability is not None else "—"
confidence_text = f"{1 - mean_probability:.1%}" if mean_probability is not None else "—"
lead_text = f"Day {selected_lead}" if has_predictions else "—"
coverage_text = f"{len(valid_probability):,} cells" if not valid_probability.empty else "—"
>>>>>>> origin/main
map_status = "DATA LOADED" if not valid_probability.empty else "AWAITING DATA"

st.markdown(
    f"""
    <div class="stats-grid">
      <article class="card stat-card">
<<<<<<< HEAD
        <div class="eyebrow">{selected_category} bust probability</div>
        <div class="metric">{probability_text}</div>
        <div class="metric-copy">Mean grid-cell risk · selected region and lead</div>
=======
        <div class="eyebrow">Mean grid-cell bust probability</div>
        <div class="metric">{probability_text}</div>
        <div class="metric-copy">Selected region and forecast lead</div>
>>>>>>> origin/main
      </article>
      <article class="card stat-card">
        <div class="eyebrow">Mean forecast confidence</div>
        <div class="metric">{confidence_text}</div>
<<<<<<< HEAD
        <div class="metric-copy">1 − selected category bust probability</div>
=======
        <div class="metric-copy">1 − mean bust probability</div>
>>>>>>> origin/main
      </article>
      <article class="card stat-card">
        <div class="eyebrow">Selected forecast lead</div>
        <div class="metric">{lead_text}</div>
<<<<<<< HEAD
        <div class="metric-copy">Interactive Day 1–10 selection</div>
=======
        <div class="metric-copy">Interactive Day 1–5 selection</div>
>>>>>>> origin/main
      </article>
      <article class="card stat-card">
        <div class="eyebrow">Mapped coverage</div>
        <div class="metric">{coverage_text}</div>
        <div class="metric-copy">Valid prediction cells at this lead</div>
      </article>
    </div>
    """,
    unsafe_allow_html=True,
)

main_column, side_column = st.columns([1.85, 1], gap="medium")
with main_column:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="card-head">
              <div>
<<<<<<< HEAD
                <h2>{selected_category} bust outlook</h2>
=======
                <h2>Forecast bust outlook</h2>
>>>>>>> origin/main
              </div>
              <span class="status-pill">{map_status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
<<<<<<< HEAD
        render_india_map(
            map_data.assign(overall_bust_probability=map_data[PROBABILITY_COLUMN])
            if map_data is not None and not map_data.empty else None
        )
        st.markdown(
            f"""
            <div class="legend">
              <b>{selected_category} bust probability</b>
              <span><i class="dot" style="background:#4f9b98"></i>&lt;30%</span>
              <span><i class="dot" style="background:#d5a14a"></i>30–59%</span>
              <span><i class="dot" style="background:#bc624d"></i>≥60%</span>
=======
        render_india_map(map_data)
        st.markdown(
            """
            <div class="legend">
              <b>Bust probability</b>
              <span><i class="dot" style="background:#299e98"></i>&lt;30%</span>
              <span><i class="dot" style="background:#e9b748"></i>30–59%</span>
              <span><i class="dot" style="background:#ee685d"></i>≥60%</span>
>>>>>>> origin/main
            </div>
            """,
            unsafe_allow_html=True,
        )

with side_column:
    st.markdown(
        """
        <section class="card">
          <div class="card-head">
            <h2>Why is confidence low?</h2>
            <span class="status-pill">DIAGNOSTICS</span>
          </div>
          <div class="card-body">
            <div class="diagnostic-summary">
              <div class="diagnostic-summary-icon">i</div>
              <div><b>Confidence factors are not available yet</b><span>Select a completed forecast run to evaluate the supporting evidence.</span></div>
            </div>
            <div class="diagnostic-grid">
              <div class="diagnostic-item">
                <div class="diagnostic-symbol">↔</div>
                <div><div class="diagnostic-title">Ensemble spread</div><div class="diagnostic-copy">Member-track disagreement</div></div>
                <span class="diagnostic-state">Pending</span>
              </div>
              <div class="diagnostic-item">
                <div class="diagnostic-symbol">Δ</div>
                <div><div class="diagnostic-title">Run-to-run shift</div><div class="diagnostic-copy">Position and rainfall displacement</div></div>
                <span class="diagnostic-state">Pending</span>
              </div>
              <div class="diagnostic-item">
                <div class="diagnostic-symbol">◷</div>
                <div><div class="diagnostic-title">Historical regime match</div><div class="diagnostic-copy">Evidence from verified past cases</div></div>
                <span class="diagnostic-state">Pending</span>
              </div>
            </div>
          </div>
        </section>
<<<<<<< HEAD
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <section class="card section-gap">
          <div class="card-head"><h2>Six-category bust outlook</h2><span class="status-pill">DAY {selected_lead}</span></div>
          <div class="card-body">
            <div class="category-risk-row heading"><span>Category</span><span>Bust</span><span>Confidence</span></div>
            {category_rows_html}
=======
        <div class="section-gap"></div>
        <section class="card">
          <div class="card-head"><h2>Model agreement</h2></div>
          <div class="card-body">
            <div class="empty-line"></div>
            <div class="empty-line short"></div>
            <div class="empty-line"></div>
            <div class="key-value"><span>Component disagreement</span><b>—</b></div>
>>>>>>> origin/main
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

chart_column, tracker_column = st.columns([1.1, 1], gap="medium")
with chart_column:
    with st.container(border=True):
        st.markdown(
<<<<<<< HEAD
            f"""
            <div class="card-head">
              <div>
                <h2>{selected_category} bust probability across Day 1–10</h2>
                <div class="card-subtitle">Selected category · mean grid-cell probability by lead</div>
=======
            """
            <div class="card-head">
              <div>
                <h2>Bust probability across Day 1–5</h2>
                <div class="card-subtitle">Interactive Plotly lead-time profile</div>
>>>>>>> origin/main
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
<<<<<<< HEAD
        render_lead_probability_chart(
            category_data.assign(
                overall_bust_probability=category_data[PROBABILITY_COLUMN]
            ) if not category_data.empty else None
        )
=======
        render_lead_probability_chart(working_data if has_predictions else None)
>>>>>>> origin/main

with tracker_column:
    st.markdown(
        """
        <section class="card section-gap">
          <div class="card-head"><h2>Coastal system tracker</h2></div>
          <div class="card-body">
<<<<<<< HEAD
            <div class="key-value"><span>System ID</span><b><span class="skeleton skeleton-sm"></span></b></div>
            <div class="key-value"><span>Classification</span><b><span class="skeleton skeleton-sm"></span></b></div>
            <div class="key-value"><span>Forecast centre</span><b><span class="skeleton skeleton-sm"></span></b></div>
            <div class="key-value"><span>Track stage</span><b><span class="skeleton skeleton-sm"></span></b></div>
=======
            <div class="key-value"><span>System ID</span><b>—</b></div>
            <div class="key-value"><span>Classification</span><b>—</b></div>
            <div class="key-value"><span>Forecast centre</span><b>—</b></div>
            <div class="key-value"><span>Track stage</span><b>—</b></div>
>>>>>>> origin/main
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
