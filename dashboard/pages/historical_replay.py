"""Historical forecast replay page."""

import streamlit as st

from page_ui import render_header


render_header(
    "Historical replay",
    "Replay a past forecast decision",
    "Compare information available at issue time with the later verified outcome.",
)

st.selectbox("Archived forecast run", ["No replay cases available"], disabled=True)

st.markdown(
    """
    <section class="bs-card bs-space">
      <div class="bs-card-head"><h2>Archived forecast replay</h2></div>
      <div class="bs-card-body">
        <div class="bs-grid-2">
          <div>
            <div class="bs-eyebrow">Available at issuance</div>
            <div class="bs-kv"><span>Selected lead</span><b>—</b></div>
            <div class="bs-kv"><span>Forecast rainfall</span><b>—</b></div>
            <div class="bs-kv"><span>Predicted bust probability</span><b>—</b></div>
            <div class="bs-note">Only information available at forecast issuance belongs in this panel.</div>
          </div>
          <div>
            <div class="bs-eyebrow">Verified outcome</div>
            <div class="bs-empty">Select a verified historical case to display its observed outcome.</div>
          </div>
        </div>
      </div>
    </section>
    <section class="bs-card bs-space">
      <div class="bs-card-head"><h2>Similar past forecasts</h2></div>
      <div class="bs-card-body">
        <div class="bs-empty">No verified replay catalogue is connected yet.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
