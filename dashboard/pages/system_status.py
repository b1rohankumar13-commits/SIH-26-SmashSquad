"""Local data and dashboard status page."""

from html import escape
from pathlib import Path

import streamlit as st

from page_ui import render_header


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "current_predictions"
prediction_files = []
if PREDICTIONS_DIR.exists():
    prediction_files = [
        *PREDICTIONS_DIR.glob("*.parquet"),
        *PREDICTIONS_DIR.glob("*.csv"),
    ]

latest_name = "—"
if prediction_files:
    latest_name = escape(max(prediction_files, key=lambda path: path.stat().st_mtime).name)

render_header(
    "Data & system status",
    "Check data readiness",
    "See whether the local outputs required by the dashboard are available.",
)

prediction_status = "Available" if prediction_files else "Awaiting data"
directory_status = "Ready" if PREDICTIONS_DIR.exists() else "Not created"

st.markdown(
    f"""
    <div class="bs-grid-2">
      <section class="bs-card">
        <div class="bs-card-head"><h2>Data &amp; pipeline health</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Current prediction output</span><b>{prediction_status}</b></div>
          <div class="bs-kv"><span>Local output directory</span><b>{directory_status}</b></div>
          <div class="bs-kv"><span>Latest file</span><b>{latest_name}</b></div>
          <div class="bs-kv"><span>Observation verification</span><b>Awaiting data</b></div>
          <div class="bs-note">Missing essential inputs remain visibly unavailable; they are never represented as low bust probability.</div>
        </div>
      </section>
      <section class="bs-card">
        <div class="bs-card-head"><h2>Local dashboard architecture</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Interface</span><b>Streamlit</b></div>
          <div class="bs-kv"><span>Geographic view</span><b>PyDeck</b></div>
          <div class="bs-kv"><span>Charts</span><b>Plotly</b></div>
          <div class="bs-kv"><span>Current storage</span><b>Local files</b></div>
          <div class="bs-kv"><span>Prediction tables</span><b>Parquet / CSV</b></div>
        </div>
      </section>
    </div>
    <section class="bs-card">
      <div class="bs-card-head"><h2>Expected dashboard outputs</h2></div>
      <div class="bs-card-body">
        <div class="bs-empty">Completed prediction and verification outputs will be listed here as they become available.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
