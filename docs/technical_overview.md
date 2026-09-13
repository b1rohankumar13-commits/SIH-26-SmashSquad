# Technical Overview — How This System Works

This document explains the project end-to-end for someone joining without prior
context on the codebase. It's self-contained — no need to have looked at the repo
first.

---

## 1. What the system computes

For a weather forecast issued at some initialization time, covering **Day 1 through
Day 10** of lead time, over a **fixed grid covering India** (roughly 5–38°N,
65–100°E, at 0.5° resolution — about 4,600 grid cells, including a margin over the
surrounding seas), the system outputs:

- a **gridded probability**, per cell, per lead day, that *this forecast's prediction
  for that cell and day will turn out to be wrong enough to count as a "bust"*
- a single **overall probability** summarizing the whole forecast cycle

This is not a new weather forecast. It's a **reliability layer on top of an existing
forecast** — it doesn't say what the weather will be, it says how much to trust the
forecast that's already been issued.

"Bust" is not something you can observe directly. It's a **derived label**, computed
after the fact by comparing what was forecast against what actually happened. Defining
that label, per weather hazard, is a real piece of engineering in its own right,
separate from the prediction models that come later.

---

## 2. The six hazards

The system focuses on six categories of high-impact weather, because these are both
the most consequential and the most prone to forecast failure:

1. **Heavy rainfall**
2. **Monsoon depressions**
3. **Cyclones**
4. **Heat waves**
5. **Western disturbances**
6. **Active/break monsoon phase**

Each hazard needs its own definition of what "bust" means, because they fail in
different ways: a rainfall forecast busts by missing or over-predicting how much rain
falls where; a cyclone forecast busts by getting the track or intensity wrong; a heat
wave forecast busts by missing a sustained temperature threshold over a region.

**Important design point:** these six hazards are used to *define* and *evaluate* the
bust label — they are **not** six separate prediction models. All six hazard
detectors feed into **one combined label**: a forecast cycle is marked as a "bust" if
*any* of the six hazards busted (a logical OR across categories). The prediction
models (§6–8) then predict that one combined label, not six separate ones.

This is a deliberate choice driven by data scarcity. Cyclones, for example, happen
only a handful of times a year in the North Indian Ocean — nowhere near enough
labelled examples to train a dedicated model per hazard. Pooling all six hazards into
one target gives the models enough positive examples to learn from. You still get
hazard-level insight, but as a **reporting breakdown at evaluation time** (how well did
we do specifically on cyclone-driven busts?), not as six separately-trained models.

---

## 3. Why this is hard

- **The label has to be built, not just read off a sensor.** Every hazard needs its
  own error-detection logic before you even have something to train a model on.
- **Bust events are rare.** By definition they're the tail behavior, not the typical
  case. This shapes almost every downstream modelling decision — model size, how much
  data is needed, how training is structured — around not overfitting to a small
  number of positive examples.
- **The failure mode is spatial and multi-scale.** A bust over Kerala doesn't imply
  anything about Rajasthan. And the same weather system can have both a small local
  footprint (localized heavy rain) and a large synoptic-scale driver (the monsoon flow
  steering it) — a model needs to reason at both scales at once, not just one fixed
  window size.

---

## 4. Stage 1 — Getting and cleaning the data

Raw forecast data (ensemble weather model output, archived and operational) and raw
observation data (India Meteorological Department gridded rainfall/temperature,
cyclone best-track records) come from external sources in inconsistent raw formats
(GRIB, NetCDF, custom binary grids).

This stage:
- **Downloads** the raw files from their sources.
- **Decodes** them into a common array format.
- **Regrids** everything onto the same fixed 0.5° grid, since different sources
  natively use different resolutions/projections.
- **Aligns time**: a forecast's "Day 3" and an observation's "24 hours ending 03:00
  UTC" don't line up trivially — getting this wrong silently corrupts every label
  downstream, so this alignment logic is treated as a first-class, carefully-tested
  part of the pipeline rather than an afterthought.
- **Converts units** and applies **quality-control masks** (invalid/missing cells).
- Converts **cumulative** forecast rainfall (a running total from the forecast's
  start) into **daily increments** — this requires validating that the data actually
  starts from a zero baseline and that lead times are strictly increasing, otherwise a
  malformed file could silently produce wrong daily rainfall values instead of erroring
  out.

Of the six hazards, **heavy rainfall's data pipeline is the one that's been built out
fully** as a proof of concept before generalizing to the other five. It has downloaders
for the relevant forecast archive, real alignment logic against IMD rainfall, and is
covered by tests. The other five hazards' acquisition/preprocessing paths are
scaffolded (the interfaces exist) but not yet filled in.

---

## 5. Stage 2 — Detection: turning clean data into a bust label

This stage answers: *given what was forecast and what actually happened, did this
forecast bust?* — per hazard, then combined.

**How rainfall detection works (the completed reference case), broken into steps:**

1. **Ensemble exceedance probability.** Weather forecasts are usually run as an
   *ensemble* — many slightly different simulations. For each grid cell, compute what
   fraction of ensemble members predicted rainfall at or above a given threshold. This
   turns a spread of raw forecasts into one probability per cell, handled carefully so
   that cells with missing/invalid ensemble members don't silently bias the result.
2. **Miss / false-alarm classification.** Compare that forecast probability against
   what was actually observed. A cell where rain occurred but the forecast said it was
   unlikely is a **miss**. A cell where the forecast said rain was likely but it didn't
   occur is a **false alarm**. Thresholds for "unlikely" and "likely" are configured
   values, not model-internal — tunable, and reviewed as part of what defines a bust.
3. **Object extraction and matching.** Rather than just scoring cell-by-cell, group
   connected rainy cells into discrete rain "objects" (contiguous rain systems), both
   in the forecast and in the observations, then match forecast objects to observed
   objects by geographic distance (using an optimal assignment algorithm, not just
   nearest-neighbor). This captures a different, arguably more important, kind of
   failure: the rain system happened, but shifted 50 km from where it was forecast —
   distinct from "no rain system was predicted at all."
4. **Timing error.** Similarly, if an observed rain event's nearest matching forecast
   event was on the wrong day (within a small search window), that's a *timing* bust,
   separate from a spatial miss.
5. **Composite scoring.** All of these error signals (magnitude error, event
   miss/false-alarm rate, spatial-skill-score error) get **normalized** against a
   reference value computed from the training period specifically — because raw error
   magnitudes vary a lot by region, season, and lead day, and without normalizing, a
   naturally-noisier region would dominate the score even when it's not actually
   performing worse. The normalized components are then combined with configured
   weights into one composite bust score, and a threshold on that score (with a looser
   "candidate" cutoff for exploration and a stricter final cutoff) decides whether a
   given case is officially labelled a bust.

The other five hazards will each need an equivalent detector, tailored to what "bust"
means for that hazard (e.g. cyclone track/intensity error against the official best
track; a temperature-threshold miss sustained over days for heat waves). The rainfall
detector's structure — probability → error classification → object/timing matching →
normalized composite score — is the template to follow, not something each hazard
reinvents from scratch. Right now, only rainfall's detector is implemented; the other
five are scaffolded but empty.

**Combining into one label:** each hazard detector produces its own bust/no-bust
verdict; these get combined with a logical OR into the single overall bust label the
prediction models will actually be trained on (see §2 for why).

**Cataloguing:** after detection, individual bust records get grouped (so one event
spanning several days/cells becomes one catalogue entry rather than many duplicates)
and, where possible, cross-checked against official recorded weather events for
validation.

---

## 6. Stage 3 — Building model inputs

The same underlying clean data gets turned into **two different representations**,
because the two prediction models (§7) consume different shapes of input:

- **A tabular representation** — one row of hand-engineered scalar numbers per case
  (ensemble spread, exceedance probabilities, rain-area size, spatial gradients, etc.)
  for the tree-based model.
- **A gridded sequence representation** — a tensor of per-grid-cell feature vectors,
  one for each of the 10 lead days, for the spatial neural network model.

Both are built from the same clean data produced in Stage 1; the tabular one is a
compressed numeric summary, the gridded one preserves the full spatial field.

---

## 7. Stage 4a — The tabular model (XGBoost)

A gradient-boosted tree ensemble trained on the tabular features to output a bust
probability. It's configured for binary classification, evaluated with a metric
suited to rare/imbalanced events (precision-recall based, rather than plain accuracy,
which would be misleading when busts are a small minority class), with settings tuned
toward not overfitting (limited tree depth, subsampling of both rows and columns per
tree).

**Why this model exists alongside a more sophisticated spatial model:** it's
**data-efficient**. With relatively few labelled bust events, a tree ensemble on
hand-engineered features is much less prone to overfitting than a large neural
network, and it comes with built-in interpretability (it's straightforward to explain
*which features* drove a given prediction) — useful both for debugging and for
explaining a prediction to a human decision-maker downstream.

---

## 8. Stage 4b — The spatial model (graph neural network + LSTM)

This is the model that produces the **gridded** bust map, and it's the piece currently
being actively designed/built.

**Why a graph neural network, specifically:** the grid cells aren't independent —
the weather systems that cause busts (cyclones, monsoon depressions, organized
rainfall) span many cells and are shaped by their surroundings. A GNN represents India
as a **mesh of connected nodes** rather than a rigid fixed-size window (which is what
a standard convolutional network would use). Crucially, the mesh is
**multi-resolution**: every node carries both short-range local edges and long-range
coarse edges *at the same time*, so the model can capture fine local detail
(localized heavy rain) and large-scale structure (synoptic circulation, monsoon flow)
simultaneously, with information able to propagate across the whole domain in just a
handful of processing steps. This mirrors how weather actually behaves — local events
are shaped by their broader environment — which is exactly the kind of spatial
reasoning this problem needs.

**How data flows through it, for a single lead day:**
1. **Encode**: the raw grid-cell features are passed onto the mesh nodes via a graph
   built between grid cells and mesh nodes.
2. **Process**: several rounds of message-passing across the multi-resolution mesh,
   letting each node's representation absorb information from both nearby and distant
   parts of the domain.
3. This produces a mesh-node embedding for that lead day.

**Handling all 10 lead days together:** since this happens once per lead day, an
**LSTM** (a recurrent network) runs across the sequence of 10 embeddings, per mesh
node, with the same weights shared across all nodes — this lets the model account for
how the situation evolves and is expected to change over the forecast window. Because
the full 10-day forecast is already fully known upfront (this system evaluates an
already-issued forecast, it doesn't generate the weather itself step by step), the
recurrence can run in **both directions** — using information from later lead days when
assessing an earlier one — unlike a model that has to predict the future one step at a
time.

**Decoding back to the grid:** the temporally-processed mesh-node embeddings are
mapped back onto the original grid cells (the reverse of the encoding step), producing
a bust probability for every grid cell, for every lead day — the gridded bust map.

**Handling the domain edge:** the grid intentionally extends a bit past India's
coastline into the surrounding seas. Cells over India (plus a coastal margin) are the
ones actually scored — this is what the model is trained and evaluated against. Cells
over the sea act as a **boundary region**: they're given as input so the model can see
systems (cyclones, moisture, western disturbances) approaching from outside the
domain, but they're never themselves scored or included in the loss. This is a
lightweight version of a technique used in regional weather models: because the model
already has the full future forecast as input rather than generating it step by step,
it doesn't need the more complex boundary-forcing machinery those models use — a
simple input-only boundary halo is enough.

**The scarce-label problem, and how it's addressed:** a network built this way has
real capacity, and training it from scratch on only the (rare) labelled bust events
would overfit badly. The approach: first **pretrain** the encoder and message-passing
layers on a much larger amount of *unlabeled* data, using a self-supervised task —
predicting the actual observed weather (or the forecast's error) from the forecast
input. This doesn't need bust labels at all, just historical forecast/observation
pairs, of which there are many years' worth. Only after that pretraining does the
model get **fine-tuned** — training just the temporal (LSTM) and final prediction
layers — on the actual, scarce bust labels. Combined with the separate tabular model
(§7), which is inherently more label-efficient, this is the primary defense against
overfitting on a small number of positive examples.

---

## 9. Stage 5 — Fusion: combining the two models

Each of the two models — tabular (XGBoost) and spatial (GNN+LSTM) — sees the *same*
underlying weather data, but represented completely differently: one as a table of
summary numbers, one as a spatial field. Because they represent the problem
differently, **they tend to make different kinds of mistakes**, which is exactly what
makes combining them worthwhile — an ensemble of models that fail in different ways is
more reliable than either alone.

Before combining, each model's raw output probability is **calibrated** (adjusted so
that, say, "70% probability" actually corresponds to observed frequency of 70%,
independently for each model — a raw model score doesn't automatically have this
property).

The two calibrated probabilities are then combined using a **logistic regression**: a
simple model that learns how to weigh the two inputs against each other, given the
correct answer during training. Alongside the two probabilities, the **disagreement
between the two models** is also given as an input — how far apart their predictions
are is itself informative, since large disagreement flags cases the overall system is
genuinely unsure about, which is useful to know downstream. The output of this final
step is the single **overall bust probability** for the forecast cycle.

A critical rule for training this fusion step correctly: it must be trained only on
predictions the two underlying models made on data **they were not trained on**
(out-of-fold or held-out validation predictions). Training it on in-sample predictions
would let it learn to trust the two models' training-set overconfidence rather than
their real-world reliability.

---

## 10. Stage 6 — Putting it into use

- A **live pipeline** takes a newly issued forecast, runs it through acquisition,
  preprocessing, feature building, both models, and fusion, producing the gridded and
  overall bust probabilities for that cycle.
- A **historical replay** path reruns the same pipeline over past forecast cycles, used
  for backtesting and evaluation rather than live use.
- **Evaluation** computes standard classification and calibration metrics, and — this
  is where the six-hazard structure becomes visible again — breaks those metrics down
  by lead day, by geographic region, and **by hazard category**, even though the
  underlying model predicts one unified bust probability rather than six separate
  ones. This is how you'd answer "how good is this specifically for cyclones?" without
  needing a dedicated cyclone model.
- A **web API** exposes forecast results, historical results, and regional
  breakdowns.
- A **dashboard** presents the gridded bust map over India, lets a user step through
  lead days, view a region's details, and see an explanation of what drove a given
  prediction (leveraging the tabular model's interpretability).

---

## 11. Current state, honestly

**Fully built:** the rainfall data pipeline (acquisition through preprocessing), and
rainfall's bust detector and composite scoring — this is the one hazard implemented
completely, and it's covered by tests.

**Scaffolded but not yet implemented:** the other five hazard detectors, both
feature-building paths (tabular and gridded), both prediction models in their trained
form, the fusion step's actual training run, the live/replay inference pipelines, and
most of the API and dashboard. The interfaces and file structure for all of these
exist; the logic inside them does not yet.

**Practical takeaway:** the rainfall detection code is the reference standard for how
this codebase is meant to be written — careful input validation, NaN-safe numeric
handling, normalization that accounts for regional/seasonal variation, and real test
coverage of both the normal case and the failure/edge cases. When building out any of
the remaining pieces, match that standard rather than the placeholder scaffolding
that's there now.
