"""BustSentinel multipage dashboard entry point (multi-hazard, directional)."""

from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="BustSentinel | Forecast Confidence Desk",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES_DIR = Path(__file__).resolve().parent / "pages"
PAGE_SPECS = [
    ("Forecast desk", ":material/dashboard:", "current_forecast.py", True),
]

pages = [
    st.Page(PAGES_DIR / filename, title=label, icon=icon, default=is_default)
    for label, icon, filename, is_default in PAGE_SPECS
]
selected_page = st.navigation(pages, position="hidden")

st.markdown(
    """
    <style>
    :root {
      --bs-page: #f1f4f8;
      --bs-surface: #ffffff;
      --bs-ink: #14253d;
      --bs-muted: #63748b;
      --bs-line: #dce3ec;
      --bs-blue: #245be8;
      --bs-nav: #101f35;
    }
    .stApp {
      background: var(--bs-page);
      color: var(--bs-ink);
      font-family: "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container {
      max-width: 1700px;
      padding: 1.2rem 1.875rem 3rem;
    }
    [data-testid="stSidebar"] {
      background: #101f35;
      border-right: 0;
    }
    [data-testid="stSidebar"] > div:first-child {
      padding: 2.2rem 1.35rem 1.5rem;
    }
    [data-testid="stSidebar"] * { color: #c8d4e5; }
    [data-testid="stSidebar"] hr { border-color: #2b3b51; }
    .bs-brand {
      align-items: center;
      color: #fff;
      display: flex;
      font-size: 1.38rem;
      font-weight: 750;
      gap: .65rem;
      letter-spacing: -.035em;
    }
    .bs-brand-mark {
      align-items: center;
      background:
        radial-gradient(circle at 28% 20%, rgba(255, 255, 255, .42), transparent 32%),
        linear-gradient(145deg, #78b5ff 0%, #3477f4 48%, #1746bd 100%);
      border: 1px solid rgba(183, 215, 255, .72);
      border-radius: .7rem;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, .52),
        inset 0 -3px 5px rgba(9, 42, 128, .35),
        0 3px 0 #12398f,
        0 8px 16px rgba(12, 45, 112, .32);
      color: #fff !important;
      display: inline-flex;
      font-size: .9rem;
      font-weight: 800;
      height: 2rem;
      justify-content: center;
      position: relative;
      text-shadow: 0 1px 2px rgba(6, 27, 78, .55);
      transform: translateY(-1px);
      width: 2rem;
    }
    .bs-subbrand, .bs-nav-label {
      color: #91acd0 !important;
      font-size: .69rem;
      letter-spacing: .18em;
      text-transform: uppercase;
    }
    .bs-subbrand { margin: .45rem 0 1.35rem 2.65rem; }
    .bs-nav-label { margin: 1.15rem 0 .55rem; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
      border-left: 3px solid transparent;
      border-radius: .45rem;
      color: #c8d4e5;
      margin: .18rem 0;
      padding: .72rem .85rem;
      text-decoration: none;
      transition: background .15s ease, border-color .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
      background: #1a2d49;
      color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
      background: #294569;
      border-left-color: #71a8ff;
      color: #fff;
    }
    .topbar {
      align-items: center;
      background: var(--bs-surface);
      border-bottom: 1px solid var(--bs-line);
      display: flex;
      justify-content: space-between;
      margin: -1.2rem -1.875rem 1.5rem;
      padding: 1.15rem 1.875rem;
    }
    .topbar-title {
      color: var(--bs-ink);
      font-size: .95rem;
      font-weight: 650;
    }
    .title-row { margin-bottom: 1.2rem; }
    .title-row h1 {
      color: var(--bs-ink);
      font-size: 1.78rem;
      letter-spacing: -.035em;
      line-height: 1.25;
      margin: 0;
    }
    .title-row p {
      color: var(--bs-muted);
      font-size: .96rem;
      margin: .3rem 0 0;
    }
    .bs-grid-2, .bs-grid-3, .bs-grid-4 {
      display: grid;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .bs-grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .bs-grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .bs-grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .bs-card {
      background: var(--bs-surface);
      border: 1px solid var(--bs-line);
      border-radius: .65rem;
      overflow: hidden;
    }
    .bs-card-head {
      align-items: center;
      border-bottom: 1px solid var(--bs-line);
      display: flex;
      justify-content: space-between;
      min-height: 3.7rem;
      padding: .9rem 1.15rem;
    }
    .bs-card-head h2 {
      color: var(--bs-ink);
      font-size: 1.02rem;
      margin: 0;
    }
    .bs-card-body { padding: 1rem 1.15rem; }
    .bs-kv {
      align-items: baseline;
      display: flex;
      font-size: .9rem;
      gap: 1rem;
      justify-content: space-between;
      padding: .48rem 0;
    }
    .bs-kv span { color: var(--bs-muted); }
    .bs-kv b { color: var(--bs-ink); font-weight: 600; text-align: right; }
    .bs-empty {
      border: 1px dashed #c5d1e2;
      border-radius: .5rem;
      color: var(--bs-muted);
      font-size: .9rem;
      margin-top: .8rem;
      padding: 1.75rem 1rem;
      text-align: center;
    }
    .bs-metric { padding: 1rem 1.15rem; }
    .bs-eyebrow {
      color: var(--bs-muted);
      font-size: .72rem;
      font-weight: 650;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .bs-number {
      color: var(--bs-ink);
      font-size: 2rem;
      font-weight: 650;
      letter-spacing: -.04em;
      margin: .2rem 0;
    }
    .bs-note {
      background: #f3f6fc;
      border-left: 3px solid #7595d6;
      color: #4f6480;
      font-size: .84rem;
      margin-top: .8rem;
      padding: .75rem;
    }
    .bs-table-wrap { overflow-x: auto; }
    .bs-table { border-collapse: collapse; width: 100%; }
    .bs-table th, .bs-table td {
      border-bottom: 1px solid #edf0f4;
      padding: .75rem 1rem;
      text-align: left;
      white-space: nowrap;
    }
    .bs-table th {
      background: #f8fafc;
      color: var(--bs-muted);
      font-size: .75rem;
      font-weight: 550;
    }
    .bs-table td { color: var(--bs-muted); font-size: .86rem; }
    .bs-space { margin-top: 1rem; }
    @media (min-width: 901px) {
      section[data-testid="stSidebar"] {
        display: block !important;
        min-width: 14.0625rem !important;
        transform: none !important;
        width: 14.0625rem !important;
      }
      [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    }
    @media (max-width: 900px) {
      .bs-grid-3, .bs-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px) {
      .block-container { padding: .8rem 1rem 2rem; }
      .topbar { margin: -.8rem -1rem 1.2rem; padding: 1rem; }
      .bs-grid-2, .bs-grid-3, .bs-grid-4 { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="bs-brand">
          <span class="bs-brand-mark" aria-hidden="true">B</span>
          <span>BustSentinel</span>
        </div>
        <div class="bs-subbrand">Confidence desk</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="bs-nav-label">Workspace</div>', unsafe_allow_html=True)
    for page, (label, icon, _, _) in zip(pages, PAGE_SPECS):
        st.page_link(page, label=label, icon=icon, use_container_width=True)

selected_page.run()
