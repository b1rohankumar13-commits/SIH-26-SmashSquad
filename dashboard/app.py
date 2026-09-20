"""BustSentinel multipage dashboard entry point."""

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
    ("Current forecast", ":material/dashboard:", "current_forecast.py", True),
    ("Regional details", ":material/location_on:", "regional_details.py", False),
    ("Historical replay", ":material/history:", "historical_replay.py", False),
    ("Model validation", ":material/analytics:", "model_validation.py", False),
    ("Data & system status", ":material/monitor_heart:", "system_status.py", False),
]

pages = [
    st.Page(PAGES_DIR / filename, title=label, icon=icon, default=is_default)
    for label, icon, filename, is_default in PAGE_SPECS
]
selected_page = st.navigation(pages, position="hidden")
active_page_href = "" if selected_page.title == "Current forecast" else selected_page.url_path

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;750&family=Lora:ital,wght@0,500;0,600;1,500&display=swap" rel="stylesheet">
    <style>
    :root {
      --bs-page: #f4f0e8;
      --bs-surface: #fffefa;
      --bs-ink: #213547;
      --bs-muted: #63737d;
      --bs-line: #d8dedb;
      --bs-blue: #ba643b;
      --bs-soft: #f6e8dc;
      --bs-nav: #142b42;
      --bs-font-sans: "Inter", "Segoe UI", Arial, sans-serif;
      --bs-font-serif: "Lora", Georgia, "Times New Roman", serif;
      --bs-font-base: .9rem;
    }
    .stApp {
      background: var(--bs-page);
      color: var(--bs-ink);
      font-family: var(--bs-font-sans);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container {
      max-width: 1700px;
      padding: 1.2rem 1.875rem 3rem;
    }
    [data-testid="stSidebar"] {
      background: var(--bs-nav);
      border-right: 0;
    }
    [data-testid="stSidebar"] > div:first-child {
      padding: 2.2rem 1.35rem 1.5rem;
    }
    [data-testid="stSidebar"] * { color: #edf3f2; }
    [data-testid="stSidebar"] hr { border-color: rgba(237, 243, 242, .2); }
    .bs-brand {
      align-items: center;
      color: #fff;
      display: flex;
      font-size: 1.56rem;
      font-weight: 750;
      gap: .65rem;
      letter-spacing: -.035em;
    }
    .bs-brand-mark {
      align-items: center;
      background:
        radial-gradient(circle at 28% 20%, rgba(255, 255, 255, .42), transparent 32%),
        linear-gradient(145deg, #d49269 0%, #ba643b 50%, #874327 100%);
      border: 1px solid rgba(255, 220, 185, .7);
      border-radius: .7rem;
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, .52),
        inset 0 -3px 5px rgba(85, 34, 17, .35),
        0 3px 0 #71391f,
        0 8px 16px rgba(5, 20, 32, .32);
      color: #fff !important;
      display: inline-flex;
      font-size: 1rem;
      font-weight: 800;
      height: 2.25rem;
      justify-content: center;
      position: relative;
      text-shadow: 0 1px 2px rgba(60, 24, 12, .55);
      transform: translateY(-1px);
      width: 2.25rem;
    }
    .bs-subbrand, .bs-nav-label {
      color: #bdd0d0 !important;
      font-size: .78rem;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .bs-subbrand { margin: .45rem 0 1.35rem 2.65rem; }
    .bs-nav-label { margin: 1.15rem 0 .55rem; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
      border-left: 3px solid transparent;
      border-radius: .45rem;
      color: #edf3f2;
      font-size: 1.02rem;
      margin: .18rem 0;
      padding: .72rem .85rem;
      text-decoration: none;
      transition: background .15s ease, border-color .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a p {
      font-size: 1.02rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a [data-testid="stIconMaterial"] {
      font-size: 1.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
      background: #23405a;
      color: #fff;
      transform: translateX(2px);
      box-shadow: 0 4px 10px rgba(0, 0, 0, .25);
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
      background: #ba643b;
      border-left-color: #e6ad83;
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
    .title-row h1, .bs-card-head h2 {
      font-family: var(--bs-font-serif);
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
      border-radius: .5rem;
      box-shadow: 0 3px 11px rgba(21, 63, 67, .06);
      overflow: hidden;
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .bs-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 10px 24px rgba(21, 63, 67, .14);
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
      font-size: var(--bs-font-base);
      gap: 1rem;
      justify-content: space-between;
      padding: .48rem 0;
    }
    .bs-kv span { color: var(--bs-muted); }
    .bs-kv b { color: var(--bs-ink); font-weight: 600; text-align: right; }
    .bs-empty {
      border: 1px dashed #c8d2ce;
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
      background: var(--bs-soft);
      border-left: 3px solid var(--bs-blue);
      color: var(--bs-ink);
      font-size: .84rem;
      margin-top: .8rem;
      padding: .75rem;
    }
    .bs-table-wrap { overflow-x: auto; }
    .bs-table { border-collapse: collapse; width: 100%; }
    .bs-table th, .bs-table td {
      border-bottom: 1px solid var(--bs-line);
      padding: .75rem 1rem;
      text-align: left;
      white-space: nowrap;
    }
    .bs-table th {
      background: #f6f4ed;
      color: var(--bs-muted);
      font-size: .75rem;
      font-weight: 550;
    }
    .bs-table td { color: var(--bs-muted); font-size: calc(var(--bs-font-base) - .04rem); }
    .bs-space { margin-top: 1rem; }
    .bs-skeleton {
      background: linear-gradient(90deg, #e4dfd3 25%, #f1ede2 37%, #e4dfd3 63%);
      background-size: 400% 100%;
      border-radius: .3rem;
      display: inline-block;
      animation: bs-shimmer 1.4s ease infinite;
      height: 1.8rem;
      width: 5rem;
    }
    @keyframes bs-shimmer {
      0% { background-position: 100% 50%; }
      100% { background-position: 0 50%; }
    }
    @media (min-width: 901px) {
      section[data-testid="stSidebar"] {
        display: block !important;
        min-width: 16.25rem !important;
        transform: none !important;
        width: 16.25rem !important;
      }
      /* Hide the in-sidebar "<<" toggle so the pinned sidebar can't be
         collapsed with no way back; leave stSidebarCollapsedControl (the
         reopen button) visible as a fallback in case it ever does collapse. */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseIconButton"] { display: none !important; }
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

# Streamlit page links do not expose aria-current, so mark the selected route.
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[href="{active_page_href}"] {{
      background: #ba643b !important;
      border-left-color: #e6ad83 !important;
      color: #ffffff !important;
    }}
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

# Streamlit's sidebar is user-resizable and can remember a previously
# dragged width for this browser session, which reasserts itself after our
# CSS paints (causing an expand-then-contract flash). Continuously re-lock
# the width via JS instead of relying on CSS alone.
st.components.v1.html(
    """
    <script>
    (function () {
      const TARGET_WIDTH = "16.25rem";
      function lockSidebarWidth() {
        try {
          const doc = window.parent.document;
          const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
          if (sidebar && window.parent.innerWidth >= 901) {
            sidebar.style.setProperty("width", TARGET_WIDTH, "important");
            sidebar.style.setProperty("min-width", TARGET_WIDTH, "important");
          }
        } catch (e) {
          /* cross-origin or not-yet-mounted; ignore */
        }
      }
      lockSidebarWidth();
      try {
        const doc = window.parent.document;
        const observer = new MutationObserver(lockSidebarWidth);
        observer.observe(doc.body, {
          attributes: true,
          attributeFilter: ["style"],
          subtree: true,
        });
        window.parent.addEventListener("resize", lockSidebarWidth);
      } catch (e) {
        /* ignore */
      }
    })();
    </script>
    """,
    height=0,
)