<<<<<<< HEAD
"""Category-specific held-out GNN validation page."""

from html import escape

import pandas as pd
import streamlit as st

from category_data import CATEGORIES, load_validation_results
=======
"""Model validation page."""

import streamlit as st

>>>>>>> origin/main
from page_ui import render_header


render_header(
    "Model validation",
    "Validate forecast-bust skill",
<<<<<<< HEAD
    "Track category-specific discrimination, calibration, detection, and false alarms.",
)

category = st.selectbox("Weather category", CATEGORIES)
validation, _ = load_validation_results()
selected = (
    validation[validation["category"].astype(str) == category].copy()
    if validation is not None else pd.DataFrame()
)
for column in ("pr_auc", "brier_score", "recall", "false_alarm_ratio"):
    if not selected.empty:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

overall = selected[selected["lead_band"].astype(str) == "All leads"] if not selected.empty else pd.DataFrame()
if not overall.empty and "model_id" in overall:
    overall = overall.sort_values("model_id")
metric_row = overall.iloc[0] if len(overall) == 1 else None


def metric_text(column: str) -> str:
    if metric_row is None or pd.isna(metric_row[column]):
        return "—"
    value = float(metric_row[column])
    return f"{value:.3f}" if column == "brier_score" else f"{value:.1%}"


st.markdown(
    f"""
    <div class="bs-grid-4">
      <div class="bs-card bs-metric"><div class="bs-eyebrow">PR-AUC</div><div class="bs-number">{metric_text('pr_auc')}</div><span>Compare with positive-class prevalence</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">Brier score</div><div class="bs-number">{metric_text('brier_score')}</div><span>Probability quality</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">Recall / POD</div><div class="bs-number">{metric_text('recall')}</div><span>Bust-event detection</span></div>
      <div class="bs-card bs-metric"><div class="bs-eyebrow">False-alarm ratio</div><div class="bs-number">{metric_text('false_alarm_ratio')}</div><span>At the evaluated threshold</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected.empty:
    st.info(f"No chronological holdout results are available for {category} yet.")
else:
    columns = ["lead_band", "pr_auc", "brier_score", "recall", "false_alarm_ratio"]
    if "model_id" in selected:
        columns.insert(0, "model_id")
    table = selected[columns].rename(
        columns={
            "model_id": "GNN model",
            "lead_band": "Lead band",
            "pr_auc": "PR-AUC",
            "brier_score": "Brier score",
            "recall": "Recall / POD",
            "false_alarm_ratio": "False-alarm ratio",
        }
    )
    st.markdown(f"### {escape(category)} · held-out results")
    st.dataframe(table, width="stretch", hide_index=True)

st.markdown(
    f"""
    <div class="bs-grid-2 bs-space">
      <section class="bs-card">
        <div class="bs-card-head"><h2>{escape(category)} reliability / calibration</h2></div>
        <div class="bs-card-body"><div class="bs-empty">Category-specific calibration plot will appear after evaluation output is connected.</div></div>
=======
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
>>>>>>> origin/main
      </section>
      <section class="bs-card">
        <div class="bs-card-head"><h2>Required breakdowns</h2></div>
        <div class="bs-card-body">
<<<<<<< HEAD
          <div class="bs-kv"><span>Selected model</span><b>{escape(str(metric_row['model_id'])) if metric_row is not None and 'model_id' in metric_row else 'Awaiting model ID'}</b></div>
          <div class="bs-kv"><span>Lead bands</span><b>Day 1–3 · 4–7 · 8–10</b></div>
=======
          <div class="bs-kv"><span>Lead bands</span><b>Day 1–3 · 4–7 · 8–10</b></div>
          <div class="bs-kv"><span>Weather categories</span><b>Separately reported</b></div>
>>>>>>> origin/main
          <div class="bs-kv"><span>Validation design</span><b>Chronological holdout</b></div>
        </div>
      </section>
    </div>
    """,
    unsafe_allow_html=True,
)
