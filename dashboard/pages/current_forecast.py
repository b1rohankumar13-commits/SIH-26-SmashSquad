"""Forecast desk: multi-hazard, directional (miss vs false-alarm) bust risk."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.india_map import COLOURS, MCZ_BOX, build_deck, classify, render_map  # noqa: E402
from data import grid_frame, grid_lead_summary, load_index, monsoon_arrays  # noqa: E402
from page_ui import render_header  # noqa: E402
from case_studies import CASES  # noqa: E402

HAZARD_ORDER = ("rain", "heatwave", "monsoon")
MODE_LABELS = {"both": "Both directions", "miss": "Miss risk (under-forecast)",
               "false_alarm": "False-alarm risk (over-forecast)"}
TARGET_LABELS = {"miss_break": "Missed break spell", "fa_break": "False break alarm",
                 "miss_active": "Missed active spell", "fa_active": "False active alarm"}
EVENT_WORD = {"rain": "extreme rain", "heatwave": "heatwave", "monsoon": "spell"}

st.markdown(
    """
    <style>
    .bs-sub { color: var(--bs-muted); font-size: .8rem; line-height: 1.45; }
    .bs-chip { border-radius: .3rem; font-size: .7rem; font-weight: 700; padding: .2rem .5rem;
               letter-spacing: .03em; }
    .chip-miss { background: #ffeadf; color: #c2410c; }
    .chip-fa { background: #e8eeff; color: #3552c4; }
    .chip-ok { background: #e6f5f2; color: #1f7f73; }
    .chip-na { background: #eef1f5; color: #6b7c93; }
    .legend-row { align-items: flex-start; display: flex; gap: .7rem; padding: .55rem 0;
                  border-bottom: 1px solid #eef1f5; font-size: .82rem; color: var(--bs-muted); }
    .legend-row:last-child { border-bottom: 0; }
    .sw { border-radius: .2rem; flex: 0 0 auto; height: .95rem; margin-top: .1rem; width: .95rem; }
    .ring { border: 2px solid; border-radius: 50%; flex: 0 0 auto; height: .8rem; margin-top: .15rem; width: .8rem; }
    .legend-row b { color: var(--bs-ink); }
    .bs-number.small { font-size: 1.55rem; }
    [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p {
      color: var(--bs-muted) !important; font-size: .74rem !important; font-weight: 600;
      letter-spacing: .06em; text-transform: uppercase; }
    [data-testid="stHeaderActionElements"] { display: none; }
    .case-card { border-left: 4px solid #f0542a; margin-bottom: 1rem; }
    .case-card.case-false_alarm { border-left-color: #466eeb; }
    .case-story { color: var(--bs-ink); font-size: .92rem; margin: 0 0 .9rem; }
    .case-stats { margin-bottom: 0; }
    [data-testid="stCheckbox"] label p { color: var(--bs-ink) !important; font-size: .85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def rgba(c: list[int]) -> str:
    return f"rgba({c[0]},{c[1]},{c[2]},{c[3] / 255:.2f})"


def fmt_date(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def one_in(p: float) -> str:
    if p <= 0:
        return "—"
    return f"{round(10 * p)} in 10" if p >= 0.25 else f"1 in {round(1 / p)}"


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def tier_sentence(t: dict, direction: str, event: str) -> str:
    what = (f"{article(event)} {event} that GEFS failed to forecast" if direction == "miss"
            else f"{article(event)} {event} GEFS forecast that did not happen")
    lift = f"{t['lift']:.0f}× the normal rate" if t["lift"] < 100 else "vs. well under 1% normally"
    return (f"{one_in(t['precision'])} flagged cells turned out to be {what} "
            f"({lift}); catches {t['catch_rate']:.0%} of them.")


def counts(hi: int, el: int, t: dict) -> str:
    out = f"{hi} <span class='bs-sub'>high</span>"
    return out + (f" · {el} <span class='bs-sub'>elevated</span>" if t.get("elevated") else "")


def card(eyebrow: str, number: str, sub: str, chip: str = "") -> str:
    return (f'<article class="bs-card bs-metric"><div class="bs-eyebrow">{eyebrow} {chip}</div>'
            f'<div class="bs-number small">{number}</div><div class="bs-sub">{sub}</div></article>')


index = load_index()
render_header("Forecast desk", "Where will the forecast bust — and which way?",
              "Per-cell risk that the GEFS ensemble is confidently wrong: missing an event it "
              "should have called, or calling one that never comes.")
if index is None:
    st.info("No exported predictions found. Run `python scripts/export_dashboard_predictions.py`.",
            icon=":material/info:")
    st.stop()

hazards = [h for h in HAZARD_ORDER if h in index]
cases = {k: c for k, c in CASES.items() if c["hazard"] in index and c["init"] in index[c["hazard"]]["inits"]}


def apply_case() -> None:
    """Jump the controls to the chosen case study."""
    c = cases.get(st.session_state.case)
    if c is None:
        return
    st.session_state.hazard = c["hazard"]
    st.session_state[f"init_{c['hazard']}"] = c["init"]
    st.session_state[f"lead_{c['hazard']}"] = c["lead"]
    st.session_state[f"mode_{c['hazard']}"] = c["mode"]
    st.session_state.show_obs = True


cs, _ = st.columns([2, 3])
with cs:
    case_id = st.selectbox("Case study", ["explore", *cases], key="case", on_change=apply_case,
                           format_func=lambda k: "Free exploration" if k == "explore" else "★ " + cases[k]["title"])

c1, c2, c3, c4 = st.columns([1.25, 1.5, 1.2, .8], gap="medium")
with c1:
    hazard = st.selectbox("Hazard", hazards, format_func=lambda h: index[h]["label"], key="hazard")
H = index[hazard]
inits = H["inits"]
if H["kind"] == "grid":
    notable = [i for i in H.get("notable_inits", []) if i in inits]
    options = notable + [i for i in inits if i not in notable]
    with c2:
        mode = st.selectbox("Show", list(MODE_LABELS), format_func=MODE_LABELS.get, key=f"mode_{hazard}")
    with c3:
        init = st.selectbox("Forecast issued (00 UTC)", options, key=f"init_{hazard}",
                            format_func=lambda s: fmt_date(s) + ("  ★ notable" if s in notable else ""))
else:
    notable = [i for i in H.get("notable_inits", []) if i in inits]
    options = notable + [i for i in inits if i not in notable]
    with c2:
        st.selectbox("Show", ["Regime busts (4 types)"], disabled=True)
    with c3:
        init = st.selectbox("Forecast issued (00 UTC)", options, key=f"init_{hazard}",
                            format_func=lambda s: fmt_date(s) + ("  ★ notable" if s in notable else ""))
    mode = "both"
with c4:
    lead = st.selectbox("Lead day", list(range(1, 10)), format_func=lambda d: f"Day {d}", key=f"lead_{hazard}")
init_pos = inits.index(init)
case = cases.get(case_id)
if case and (case["hazard"], case["init"], case["lead"]) != (hazard, init, lead):
    case = None
valid_day = (pd.Timestamp(fmt_date(init)) + pd.Timedelta(days=lead - 1)).strftime("%d %b %Y")

t1, t2, _ = st.columns([1, 1, 2.2])
with t1:
    st.session_state.setdefault("show_obs", True)
    show_obs = st.toggle("Show what actually happened", key="show_obs",
                         help="Historical replay: rings mark cells where the forecast really busted.")
with t2:
    show_gefs = st.toggle("Show GEFS event probability", value=False, disabled=H["kind"] != "grid",
                          help="Teal shading = share of the 5 GEFS members forecasting the event.")

# --------------------------------------------------------------------------- grid hazards
if H["kind"] == "grid":
    tiers = H["tiers"]
    frame = grid_frame(hazard, init_pos, lead)
    f = classify(frame, tiers, mode)
    n_cells = len(f)
    hi_m, el_m = int((f["miss_tier"] == 2).sum()), int((f["miss_tier"] == 1).sum())
    hi_f, el_f = int((f["fa_tier"] == 2).sum()), int((f["fa_tier"] == 1).sum())
    gefs_yes = int((f["ens_prob"] >= 0.5).sum())
    obs_m, obs_f = int((f["obs_miss"] == 1).sum()), int((f["obs_fa"] == 1).sum())
    caught_m = int(((f["obs_miss"] == 1) & (f["miss_tier"] >= 1)).sum())
    caught_f = int(((f["obs_fa"] == 1) & (f["fa_tier"] >= 1)).sum())
    ev = EVENT_WORD[hazard]
    st.markdown(
        '<div class="bs-grid-4">'
        + card("Miss risk", counts(hi_m, el_m, tiers["miss"]),
               f"Cells where GEFS may be <b>under-calling</b> {ev} (valid {valid_day})",
               '<span class="bs-chip chip-miss">UNDER</span>')
        + card("False-alarm risk", counts(hi_f, el_f, tiers["false_alarm"]),
               f"Cells where GEFS may be <b>over-calling</b> {ev}", '<span class="bs-chip chip-fa">OVER</span>')
        + card("GEFS says", f"{gefs_yes} cells",
               f"≥ 50% of members forecast {ev} (of {n_cells:,} land cells)")
        + card("What happened", f"{obs_m} missed · {obs_f} false",
               (f"Alerts (elevated+) covered {caught_m}/{obs_m} misses and {caught_f}/{obs_f} false alarms"
                if (obs_m or obs_f) else "The forecast held everywhere at this lead"),
               '<span class="bs-chip chip-ok">REPLAY</span>')
        + "</div>", unsafe_allow_html=True)

    if case:
        b = case["box"]
        R = case["region"]
        inbox = f["latitude"].between(b["south"], b["north"]) & f["longitude"].between(b["west"], b["east"])
        c_gefs = int((inbox & (f["ens_prob"] >= 0.5)).sum())
        before = f"issued {fmt_date(case['init'])}, {lead - 1} days before"
        if case["mode"] == "miss":
            cm = inbox & (f["obs_miss"] == 1)
            n_hit = int(cm.sum())
            c_caught = int((cm & (f["miss_tier"] >= 1)).sum())
            c_high = int((cm & (f["miss_tier"] == 2)).sum())
            c_flag = int((inbox & (f["miss_tier"] >= 1)).sum())
            gefs_sub = (f"with ≥ 50% of members calling {ev} — effectively <b>no warning</b>" if c_gefs == 0 else
                        f"with ≥ 50% of members calling {ev} — but <b>none</b> where it actually struck: "
                        f"not one member forecast it in any of the {n_hit} hit cells")
            stats = [(f"GEFS forecast for {R}", f"{c_gefs} cells", gefs_sub),
                     ("What happened", f"{n_hit} cells", f"of {R} got {ev} GEFS missed entirely"),
                     ("BustSentinel alert", f"{c_caught} of {n_hit} caught",
                      f"{c_high} at high alert · {c_flag} {R} cells flagged in total ({before})")]
            chip = '<span class="bs-chip chip-miss">MISS</span>'
        else:
            g = inbox & (f["ens_prob"] >= 0.5)
            fa = inbox & (f["obs_fa"] == 1)
            flag = inbox & (f["fa_tier"] >= 1)
            stats = [(f"GEFS forecast for {R}", f"{c_gefs} cells", f"with ≥ 50% of members calling {ev}"),
                     ("What happened", f"{int(fa.sum())} of {c_gefs} didn't",
                      f"{ev.capitalize()} never arrived in {int(fa.sum())} of the cells GEFS called"),
                     ("BustSentinel alert", f"{int((fa & flag).sum())} of {int(fa.sum())} flagged",
                      f"{int(flag.sum())} cells flagged as likely false alarms; {int((flag & ~fa).sum())} of those "
                      f"did get {ev} ({before})")]
            chip = '<span class="bs-chip chip-fa">FALSE ALARM</span>'
        cols = "".join(f'<div><div class="bs-eyebrow">{e}</div><div class="bs-number small">{n}</div>'
                       f'<div class="bs-sub">{sub}</div></div>' for e, n, sub in stats)
        st.markdown(
            f'<section class="bs-card case-card case-{case["mode"]}"><div class="bs-card-head"><h2>★ Case study · '
            f'{case["title"]}</h2>{chip}</div><div class="bs-card-body"><p class="case-story">{case["story"]}</p>'
            f'<div class="bs-grid-3 case-stats">{cols}</div></div></section>', unsafe_allow_html=True)

    main, side = st.columns([1.75, 1], gap="medium")
    with main:
        with st.container(border=True):
            st.markdown(f'<div class="bs-card-head"><h2>{H["label"]} bust risk · Day {lead} '
                        f'(valid {valid_day})</h2><span class="bs-chip chip-na">{MODE_LABELS[mode].upper()}</span></div>',
                        unsafe_allow_html=True)
            render_map(build_deck(frame, tiers, mode, show_gefs=show_gefs, show_observed=show_obs,
                                  region_box=case["box"] if case else None,
                                  region_fill=[0, 0, 0, 0]))
    with side:
        rows = []
        for d, label in (("miss", "Miss"), ("false_alarm", "False alarm")):
            if mode not in (d, "both"):
                continue
            for tname in ("high", "elevated"):
                if not tiers[d].get(tname):
                    continue
                rows.append(f'<div class="legend-row"><span class="sw" style="background:{rgba(COLOURS[d][tname])}"></span>'
                            f'<div><b>{label} · {tname}</b><br/>{tier_sentence(tiers[d][tname], d, ev)}</div></div>')
            if show_obs:
                rows.append(f'<div class="legend-row"><span class="ring" style="border-color:{rgba(COLOURS[d]["observed"])}"></span>'
                            f'<div><b>Observed {label.lower()}</b><br/>Where the forecast really busted this way.</div></div>')
        if show_gefs:
            rows.append('<div class="legend-row"><span class="sw" style="background:rgba(46,170,150,.7)"></span>'
                        f'<div><b>GEFS event probability</b><br/>Darker = more members forecast {ev}.</div></div>')
        st.markdown('<section class="bs-card"><div class="bs-card-head"><h2>How to read the map</h2></div>'
                    f'<div class="bs-card-body">{"".join(rows)}</div></section>', unsafe_allow_html=True)
        tm, tf = tiers["miss"], tiers["false_alarm"]
        st.markdown(
            '<section class="bs-card bs-space"><div class="bs-card-head"><h2>About this model</h2></div><div class="bs-card-body">'
            f'<div class="bs-kv"><span>Event</span><b>{H["event"]}</b></div>'
            f'<div class="bs-kv"><span>Model</span><b>{H["model"]}</b></div>'
            f'<div class="bs-kv"><span>Truth</span><b>{H["truth"]}</b></div>'
            f'<div class="bs-kv"><span>Validation</span><b>{H["val_period"]}</b></div>'
            f'<div class="bs-kv"><span>Miss skill (PR-AUC)</span><b>{tm["pr_auc"]:.3f} · {tm["pr_auc"]/tm["base_rate"]:.0f}× chance</b></div>'
            f'<div class="bs-kv"><span>False-alarm skill (PR-AUC)</span><b>{tf["pr_auc"]:.3f} · {tf["pr_auc"]/tf["base_rate"]:.0f}× chance</b></div>'
            '<div class="bs-note">Historical validation period — the model never saw these dates in training.</div>'
            + (f'<div class="bs-note">False alarms are largely visible in GEFS itself: when most members call '
               f'{ev}, it does not happen about {one_in(tf["high"]["precision"])} times. The model adds most of '
               f'its value on <b>misses</b> — events GEFS gave no warning of.</div>' if hazard == "rain" else '')
            + '</div></section>', unsafe_allow_html=True)

    summ = grid_lead_summary(hazard, init_pos, tiers)
    fig = go.Figure()
    for d, label in (("miss", "Miss"), ("false_alarm", "False alarm")):
        if mode not in (d, "both"):
            continue
        fig.add_bar(x=summ["lead_day"], y=summ[f"{d}_flagged"], name=f"{label} · high alerts",
                    marker_color=rgba(COLOURS[d]["high"]))
        if show_obs:
            fig.add_scatter(x=summ["lead_day"], y=summ[f"{d}_observed"], name=f"{label} · observed",
                            mode="lines+markers", line={"color": rgba(COLOURS[d]["high"]), "dash": "dot"})
    fig.add_vline(x=lead, line_color="#9aa9bd", line_dash="dash")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                      plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.12),
                      xaxis=dict(title="Lead day", dtick=1), yaxis=dict(title="Grid cells"))
    with st.container(border=True):
        st.markdown(f'<div class="bs-card-head"><h2>Across the 9-day forecast · issued {fmt_date(init)}</h2></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# --------------------------------------------------------------------------- monsoon (regional)
else:
    a = monsoon_arrays(init_pos)
    L = lead - 1
    targets, tiers = H["targets"], H["tiers"]
    p, obs = a["p"][L], a["obs"][L]
    level = {}
    for j, t in enumerate(targets):
        if t in tiers:
            el = tiers[t]["elevated"]["threshold"] if tiers[t].get("elevated") else np.inf
            level[t] = 2 if p[j] >= tiers[t]["high"]["threshold"] else 1 if p[j] >= el else 0
    cards = []
    for j, t in enumerate(targets):
        direction = "miss" if t.startswith("miss") else "false_alarm"
        chip = ('<span class="bs-chip chip-miss">UNDER</span>' if direction == "miss"
                else '<span class="bs-chip chip-fa">OVER</span>')
        if t in tiers:
            status = ["low", "elevated", "high"][level[t]]
            sub = f"Alert: <b>{status}</b>"
        else:
            sub = "Not scored — too few validation cases"
        if show_obs and obs[j] >= 0:
            sub += " · happened: <b>" + ("yes" if obs[j] == 1 else "no") + "</b>"
        cards.append(card(TARGET_LABELS[t], f"{p[j]:.0%}", sub, chip))
    st.markdown('<div class="bs-grid-4">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    worst = max(level.items(), key=lambda kv: (kv[1], p[targets.index(kv[0])]), default=(None, 0))
    fill = [60, 80, 110, 70]
    if worst[1] > 0:
        d = "miss" if worst[0].startswith("miss") else "false_alarm"
        fill = COLOURS[d]["high" if worst[1] == 2 else "elevated"][:3] + [110]
    main, side = st.columns([1.75, 1], gap="medium")
    with main:
        with st.container(border=True):
            st.markdown(f'<div class="bs-card-head"><h2>Monsoon core zone · Day {lead} (valid {valid_day})</h2>'
                        f'<span class="bs-chip chip-na">REGIONAL</span></div>', unsafe_allow_html=True)
            render_map(build_deck(None, None, "both", show_gefs=False, show_observed=False,
                                  region_box=MCZ_BOX, region_fill=fill))
    with side:
        pa, pb = float(a["p_active"][L]), float(a["p_break"][L])
        rows = "".join(
            f'<div class="legend-row"><div><b>{TARGET_LABELS[t]}</b><br/>{tier_sentence(tiers[t]["high"], "miss" if t.startswith("miss") else "false_alarm", "spell")}</div></div>'
            for t in targets if t in tiers)
        st.markdown(
            '<section class="bs-card"><div class="bs-card-head"><h2>GEFS regime call</h2></div><div class="bs-card-body">'
            f'<div class="bs-kv"><span>Members forecasting an active spell</span><b>{pa:.0%}</b></div>'
            f'<div class="bs-kv"><span>Members forecasting a break spell</span><b>{pb:.0%}</b></div>'
            '<div class="bs-note">Box colour = strongest alert at this lead (orange = likely missed spell, '
            'blue = likely false spell alarm).</div></div></section>'
            f'<section class="bs-card bs-space"><div class="bs-card-head"><h2>What a high alert means</h2></div>'
            f'<div class="bs-card-body">{rows}</div></section>', unsafe_allow_html=True)

    fig = go.Figure()
    palette = {"miss_break": "#f59a4a", "fa_break": "#7ea2ff", "miss_active": "#e0461e", "fa_active": "#3a5fd9"}
    for j, t in enumerate(targets):
        fig.add_scatter(x=list(range(1, 10)), y=a["p"][:, j] * 100, name=TARGET_LABELS[t], mode="lines+markers",
                        line={"color": palette[t]})
        if show_obs:
            hit = np.where(a["obs"][:, j] == 1)[0]
            if len(hit):
                fig.add_scatter(x=hit + 1, y=a["p"][hit, j] * 100, mode="markers", showlegend=False,
                                marker=dict(size=14, symbol="circle-open", color=palette[t], line=dict(width=2)))
    fig.add_vline(x=lead, line_color="#9aa9bd", line_dash="dash")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", legend=dict(orientation="h", y=1.15),
                      xaxis=dict(title="Lead day", dtick=1), yaxis=dict(title="Bust probability (%)"))
    with st.container(border=True):
        st.markdown(f'<div class="bs-card-head"><h2>Regime-bust risk across the 9-day forecast · issued {fmt_date(init)}</h2></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if show_obs:
            st.caption("Open rings = the bust actually happened on that day.")
