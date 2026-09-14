"""Regional forecast evidence page."""

import streamlit as st

from page_ui import render_header


render_header(
    "Regional details",
    "Inspect a region in detail",
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
        </div>
      </section>
      <section class="bs-card">
        <div class="bs-card-head"><h2>Regime-aware evidence</h2></div>
        <div class="bs-card-body">
          <div class="bs-kv"><span>Matched historical cases</span><b>—</b></div>
          <div class="bs-kv"><span>Regime-conditioned bust rate</span><b>—</b></div>
          <div class="bs-kv"><span>Evidence window</span><b>—</b></div>
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
