"""Model validation page."""

import streamlit as st

from page_ui import render_header


render_header(
    "Model validation",
    "Validate forecast-bust skill",
    "Track discrimination, calibration, detection, and false-alarm performance.",
)

st.markdown(
    """
    <div class="bs-grid-4">
      <div class="bs-card bs-metric"><div class="bs-eyebrow">PR-AUC</div><div class="bs-number">—</div><span>Compare with positive-class prevalence</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">Brier score</div><div class="bs-number">—</div><span>Probability quality</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">Recall / POD</div><div class="bs-number">—</div><span>Bust-event detection</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">False-alarm ratio</div><div class="bs-number">—</div><span>At the selected threshold</span></div>
    </div>
    <section class="bs-card">
      <div class="bs-card-head"><h2>Held-out model comparison</h2></div>
      <div class="bs-table-wrap">
        <table class="bs-table">
          <thead><tr><th>Model</th><th>PR-AUC</th><th>Brier</th><th>Recall / POD</th><th>False-alarm ratio</th></tr></thead>
          <tbody><tr><td colspan="5">No chronological holdout results are available.</td></tr></tbody>
        </table>
      </div>
    </section>
    <div class="bs-grid-2 bs-space">
      <section class="bs-card">
        <div class="bs-card-head"><h2>Reliability / calibration</h2></div>
        <div class="bs-card-body"><div class="bs-empty">Calibration plots will appear after evaluation.</div></div>
      </section>
      <section class="bs-card">
        <div class="bs-card-head"><h2>Required breakdowns</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Lead bands</span><b>Day 1–3 · 4–7 · 8–10</b></div>
          <div class="bs-kv"><span>Weather categories</span><b>Separately reported</b></div>
          <div class="bs-kv"><span>Validation design</span><b>Chronological holdout</b></div>
        </div>
      </section>
    </div>
    """,
    unsafe_allow_html=True,
)
