"""Overall probability summary panel."""

def render_probability_panel(st, probability):
    st.metric("Overall bust probability", f"{probability:.1%}")
