<<<<<<< HEAD
"""Category-specific regional forecast evidence page."""
=======
"""Regional forecast evidence page."""
>>>>>>> origin/main

import pandas as pd
import streamlit as st

<<<<<<< HEAD
from category_data import (
    CATEGORIES, PROBABILITY_COLUMN, category_rows,
    load_current_predictions,
)
from components.probability_panel import render_lead_probability_chart
=======
>>>>>>> origin/main
from page_ui import render_header


render_header(
    "Regional details",
    "Inspect a region in detail",
<<<<<<< HEAD
    "Review category-specific bust probability and regional evidence.",
)

predictions, _ = load_current_predictions()
has_predictions = predictions is not None and not predictions.empty
regions = (
    sorted(predictions["region_id"].dropna().astype(str).unique().tolist())
    if has_predictions and "region_id" in predictions else []
)
runs = (
    sorted(predictions["run_id"].dropna().astype(str).unique().tolist(), reverse=True)
    if has_predictions and "run_id" in predictions else []
)

region_col, category_col, run_col, lead_col = st.columns([1.2, 1.2, 1.2, 0.7])
with region_col:
    region = st.selectbox("Region", regions or ["No prediction regions available"], disabled=not regions)
with category_col:
    category = st.selectbox("Weather category", CATEGORIES)
with run_col:
    run = st.selectbox("Forecast initialization", runs or ["No completed runs available"], disabled=not runs)

selected = category_rows(predictions, category)
if not selected.empty and regions:
    selected = selected[selected["region_id"].astype(str) == region]
if not selected.empty and runs:
    selected = selected[selected["run_id"].astype(str) == run]
lead_values = (
    sorted(pd.to_numeric(selected["lead_day"], errors="coerce").dropna().astype(int).unique().tolist())
    if not selected.empty else []
)
with lead_col:
    lead = st.selectbox("Lead day", lead_values or ["—"], disabled=not lead_values)

lead_rows = (
    selected[pd.to_numeric(selected["lead_day"], errors="coerce") == lead]
    if lead_values else pd.DataFrame()
)
mean_probability = float(lead_rows[PROBABILITY_COLUMN].mean()) if not lead_rows.empty else None
probability_text = f"{mean_probability:.1%}" if mean_probability is not None else "—"
confidence_text = f"{1 - mean_probability:.1%}" if mean_probability is not None else "—"
lead_text = f"Day {lead}" if lead_values else "—"
coverage_text = str(len(lead_rows)) if not lead_rows.empty else "—"

st.markdown(
    f"""
    <div class="bs-grid-2 bs-space">
      <section class="bs-card">
        <div class="bs-card-head"><h2>{category} forecast evidence</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Mean grid-cell bust probability</span><b>{probability_text}</b></div>
          <div class="bs-kv"><span>Forecast confidence</span><b>{confidence_text}</b></div>
          <div class="bs-kv"><span>Selected lead</span><b>{lead_text}</b></div>
          <div class="bs-kv"><span>Valid grid cells</span><b>{coverage_text}</b></div>
          <div class="bs-note">Values are shown only for the selected category and completed run.</div>
=======
    "Review local confidence, contributing signals, and regime-aware evidence.",
)

st.selectbox("Region", ["No prediction regions available"], disabled=True)

st.markdown(
    """
    <div class="bs-grid-2 bs-space">
      <section class="bs-card">
        <div class="bs-card-head"><h2>Regional forecast evidence</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Overall bust probability</span><b>—</b></div>
          <div class="bs-kv"><span>Forecast confidence</span><b>—</b></div>
          <div class="bs-kv"><span>Selected lead</span><b>—</b></div>
          <div class="bs-kv"><span>Dominant diagnostic</span><b>Awaiting output</b></div>
          <div class="bs-empty">Regional values will appear after a compatible prediction table is available.</div>
>>>>>>> origin/main
        </div>
      </section>
      <section class="bs-card">
        <div class="bs-card-head"><h2>Regime-aware evidence</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Matched historical cases</span><b>—</b></div>
          <div class="bs-kv"><span>Regime-conditioned bust rate</span><b>—</b></div>
          <div class="bs-kv"><span>Evidence window</span><b>—</b></div>
<<<<<<< HEAD
          <div class="bs-note">Verified category-specific historical comparisons are not connected yet.</div>
        </div>
      </section>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="bs-card"><div class="bs-card-head"><h2>Lead-by-lead regional profile</h2></div></section>',
    unsafe_allow_html=True,
)
render_lead_probability_chart(
    selected.assign(overall_bust_probability=selected[PROBABILITY_COLUMN])
    if not selected.empty else None
)
=======
          <div class="bs-note">Historical comparisons remain unavailable until verified cases are connected.</div>
        </div>
      </section>
    </div>
    <section class="bs-card">
      <div class="bs-card-head"><h2>Lead-by-lead regional profile</h2></div>
      <div class="bs-card-body">
        <div class="bs-empty">Regional probability and diagnostic plots will appear here when data is available.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
>>>>>>> origin/main
