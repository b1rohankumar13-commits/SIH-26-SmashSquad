"""Shared, data-neutral page elements for BustSentinel."""

import streamlit as st


def render_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-title">Forecast intelligence / {section}</div>
          <span class="bs-brand-mark" aria-label="BustSentinel">B</span>
        </div>
        <div class="title-row">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_panel(title: str, message: str) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.info(message, icon=":material/info:")
