# Problem Statement & Our Solution

> **Note:** No official SIH problem-statement number/title was found in this repo.
> This document reconstructs the PS from `README.md` and project discussion. Replace
> the framing below with the exact official PS text/number if you have it, so the
> submission matches word-for-word.

---

## 1. The problem

Numerical weather forecasts — the ones IMD, NCMRWF, and downstream agencies act on —
are usually good, but not always. Occasionally a forecast **busts**: it materially
diverges from what actually happens. A "calm" forecast turns into a cyclone landfall
surprise; a modest-rain forecast turns into a flash flood; a monsoon-phase call flips.

The trouble isn't that forecasts fail sometimes — every forecast system does. The
trouble is that **today there is no signal, ahead of time, telling you a given
forecast is at elevated risk of busting.** Every forecast is presented with the same
implicit confidence, whether it's a routine, reliable Day-3 call or a fragile one for
a system the model is known to struggle with. Decision-makers who act on the forecast
— evacuation calls, dam releases, sowing advisories, grid-load planning — have no way
to tell those two situations apart until it's too late.

This matters most for exactly the high-impact, high-uncertainty weather that drives
disaster response in India:

- **Heavy rainfall** (flash floods, urban flooding)
- **Cyclones** (Bay of Bengal / Arabian Sea)
- **Monsoon depressions**
- **Heat waves**
- **Western disturbances**
- **Active/break monsoon phase**

These six categories were chosen because they're both high-consequence *and*
disproportionately prone to forecast busts — which is exactly where a reliability
signal is most valuable.

**In one sentence:** *India's forecasts don't come with an honest signal of when
they're about to fail — and that missing signal is what this project builds.*

---

## 2. What we're building

A system that predicts, for each Day 1–10 forecast cycle, **the probability that this
forecast will bust** — spatially, as a **gridded map over India**, and as a single
**overall probability**. Not a new weather forecast. A **confidence layer on top of
the existing forecast**, telling you where and how much to distrust it.

Outputs:
- A **gridded bust-probability map** (grid / district / state / meteorological-region
  views) — where is this forecast fragile, right now, for this cycle.
- A single **overall bust probability** per cycle.
- Category-level evaluation (which of the six hazard types is driving the risk),
  even though the model predicts one unified bust signal (see §4).

---

## 3. Why this is hard (and why it's not just "predict the weather")

- **Ground truth is derived, not observed.** "Bust" isn't a raw measurement — it has
  to be defined per hazard (rainfall exceedance, cyclone track/intensity error, heat
  wave miss, etc.) and then combined into one label. That labeling logic *is* part of
  the modelling work (`src/detection/`).
- **Busts are rare.** By definition they're the tail, not the average case — every
  design choice downstream (model size, pretraining, fusion) exists to cope with a
  small positive-label budget.
- **The failure mode is spatial and multi-scale.** A bust in Kerala doesn't mean much
  for Rajasthan. The signal has to respect geography, and the same system (e.g. a
  monsoon depression) has both a fine local footprint and a large synoptic-scale
  driver — a single fixed-resolution model doesn't capture both cleanly.

---

## 4. How we're solving it

### Two complementary models, fused

The same underlying forecast data is given to two models that represent it
differently, because they make different mistakes and cover different failure modes:

**GNN + LSTM (spatial branch)** — sees the data as a **map**. India is represented as
a **multi-resolution mesh**: every node carries both short local edges and long-range
coarse edges, so the model reasons about fine-scale features (localized heavy rain)
and large-scale drivers (synoptic circulation, monsoon flow) at once, with information
able to propagate across the whole domain in a few steps — the way real weather
systems are shaped by their surroundings, not just their own pixel. An LSTM layer
carries this representation across the Day 1–10 lead sequence. Output: the **gridded
bust map**.

**XGBoost (tabular branch)** — sees the *same* underlying data, reduced to engineered
scalar features (ensemble spread, exceedance probabilities, rainfall-area statistics,
etc.). It's the data-efficient counterpart: robust with few labelled examples, good at
threshold/interaction rules, interpretable via SHAP. Output: an **overall bust
probability**.

**Fusion** — a logistic regression combines the two models' calibrated probabilities
plus their disagreement into one **final overall bust probability**. Two models that
see the same weather differently and are wrong in different ways make a stronger,
better-calibrated combined signal than either alone — and the disagreement itself is
informative (large disagreement flags cases the system itself is unsure about).

### One backbone, six-category labeling — not six separate models

The six hazard categories are used to **define and evaluate** the bust label, not to
split the model. Six detectors (`src/detection/cyclone.py`, `heavy_rainfall.py`,
`heatwave.py`, `monsoon_depression.py`, `western_disturbance.py`, `monsoon_phase.py`)
each decide whether *their* hazard busted; these are OR-combined into one overall
label (`overall_bust_label.py`). Both models predict that single unified target.
Splitting into six separate models would starve the rarest categories (cyclones:
~4–5 events/year) of enough labels to learn anything — pooling all hazards into one
target is what makes the label budget workable. Per-category performance is still
reported at evaluation time (`evaluate_by_category.py`), so the six-hazard structure
isn't lost — it's just not baked into six separate models.

### Coping with scarce labels

Because bust events are rare, the spatial branch is **pretrained** on an unsupervised
pretext task — predicting the observed field (or the forecast-error field) from the
forecast, using ~20 years of reforecast data with no labels required — before
fine-tuning the small labelled bust-detection head. This lets a spatially-aware model
learn real atmospheric structure without needing thousands of labelled bust events.
Fusing with the data-efficient XGBoost branch is the second line of defence against
overfitting on scarce labels.

*(Architecture, mesh design, environment, training plan: see
`docs/graphnet_design.md`.)*

---

## 5. Who benefits

The value isn't a new forecast — India already has forecasts. The value is knowing
**when to trust the one you have.**

- **IMD / NCMRWF (forecasters)** — see where and when their operational forecast is
  fragile; surfaces systematic model weaknesses for future improvement.
- **NDMA / State DMAs / district collectors** — the highest-stakes user. A flagged
  high-bust-risk forecast means hedging toward the worse-case outcome instead of
  trusting a falsely calm one — directly relevant to evacuation and resource
  pre-positioning decisions.
- **Sectors riding on each hazard category:** farmers and agromet advisories
  (monsoon phase), dam/reservoir operators and urban flood managers (heavy rainfall),
  coastal communities and fisheries (cyclones), power-grid load planners and public
  health (heat waves), north-India agriculture/transport (western disturbances).
- **Insurance / parametric weather products** (e.g. crop insurance) — sharper
  forecast-reliability information improves risk pricing and payout triggers.

**In one line:** *every operational forecast in India comes with no honest signal of
when it's about to fail — this project adds that missing layer, so the people who act
on forecasts know when to prepare for worse.*
