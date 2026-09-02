"""Day 1–10 forecast lead selector."""

def render_lead_selector(st):
    return st.slider("Lead day", min_value=1, max_value=10, value=1)
