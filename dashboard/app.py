"""Streamlit dashboard entry point."""

import streamlit as st

st.set_page_config(page_title="Forecast Bust Prediction", layout="wide")
st.title("India Forecast Bust Prediction")
st.caption("Overall Day 1–10 bust probability; categories are shown as diagnostics.")
st.info("Run the inference pipeline to populate the current map.")
