# GraphNet Spatial Branch — Design

Status: **design** (pre-implementation). Author-facing spec; solo project.

Replaces the stubbed ConvLSTM spatial branch (`src/models/convlstm.py`) with a
**GraphCast-style multi-mesh GNN encoder + bi-LSTM + GNN decoder**, producing a
**gridded Day 1–N forecast-bust probability map** over India. The existing
**XGBoost tabular branch stays**; the two are fused as they are today
(`configs/fusion.yaml`).

---

## 0. Why a GNN? (rationale)

Weather is a spatial field, and the systems that cause forecast busts — cyclones,
monsoon depressions, western disturbances, organised rainfall — are coherent
structures that span many grid cells and are shaped by their surroundings. A graph
neural network represents India as a mesh of connected nodes, letting information flow
along physically meaningful spatial relationships rather than being confined to a
fixed pixel window. Crucially, the mesh is **multi-resolution**: each node carries both
short local edges and long-range coarse edges, so the model captures fine-scale
features like localized heavy rainfall *and* large-scale drivers like synoptic
circulation and monsoon flow at the same time, and influence can propagate across the
entire domain in only a few steps. This mirrors how real weather behaves — local events
are governed by the broader environment — which is exactly the spatial reasoning
forecast-bust prediction requires.

---

## 1. Scope & non-goals

- **In scope:** the spatial branch (grid → mesh → temporal → grid), its training
  and pretraining, and how its output enters fusion.
- **Not changing:** acquisition, preprocessing, rainfall detection/scoring,
  evaluation, dashboard, API. They consume the new branch unchanged.
- **Not doing:** autoregressive rollout. Unlike GraphCast we already *have* all
  lead days of forecast as input, so there is no state to roll forward.

---

## 2. Environment & versions (recommended — nothing is currently pinned)

`requirements.txt` is fully unpinned and there is no `pyproject`, `.python-version`,
or conda env in the repo. Local interpreter is **Python 3.13.15**, which is too new
for this stack. Recommendation:

| Component | Recommended | Why |
|---|---|---|
| **Python** | **3.11.x** (dedicated env) | Sweet spot for PyTorch 2.x, PyG, `cfgrib`/`eccodes`, `geopandas`. 3.13 lacks wheels for several of these. |
| **PyTorch** | **≥ 2.4, CUDA 12.1 build** (`cu121`; `cu124` also fine) | RTX 4070 is Ada (sm_89), fully supported on CUDA 12.x. BF16 tensor cores. |
| **GNN lib** | **PyTorch Geometric (PyG)** | **Neural-LAM is built on PyG** — we reuse its graph-construction code rather than reimplement mesh generation. Install `torch-geometric` + matching `torch-scatter`/`torch-sparse` wheels for the exact torch+CUDA build. |
| **GRIB/geo stack** | install from **conda-forge** (miniforge), not pip | `eccodes`, `cfgrib`, `geopandas`, `rasterio`, `gdal` are unreliable via pip wheels **on Windows**. This is the biggest environment gotcha. |

**Recommended setup on the 4070 (Windows 11):**
1. Miniforge → `conda create -n sih python=3.11`.
2. `conda install -c conda-forge eccodes cfgrib geopandas rasterio gdal xarray netCDF4 zarr dask`.
3. `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
4. `pip install torch-geometric` (+ scatter/sparse wheels matched to torch/CUDA).
5. `pip install` the rest (xgboost, scikit-learn, shap, fastapi, streamlit, plotly, pytest…).

> Action item: once validated, produce a **pinned** `requirements-lock.txt` +
> `environment.yml` so runs are reproducible across the 4070 and Kaggle.

---

## 3. Domain, grid, interior vs. boundary

From `configs/grid.yaml`:

- Domain box: **5–38°N, 65–100°E**, **0.5°** resolution.
- Cell centres: 5.25–37.75°N × 65.25–99.75°E → **66 × 70 = 4,620 grid cells**.
- Retained lead days: **9** for rainfall (alignment drops one); sequence length is
  configurable (nominal 10). Treat `sequence_length` as a config value, not a constant.

**Interior vs. boundary (Neural-LAM style):**
- The box already extends ~2–3° past India's coastline into the Arabian Sea, Bay of
  Bengal, and the NW — so a **boundary halo largely exists already**.
- **Interior** = India land + coastal margin → these grid nodes get a bust score.
- **Boundary** = the remaining sea/edge nodes → **input-only, never scored, never in
  the loss**. They feed context inward (cyclones, western disturbances entering the domain).
- Optional later: widen the box west/south to give entering systems a longer lead-in.
  This is a data-domain decision, deferred.

---

## 4. Mesh design

**Construction:** regional **multi-scale quadrilateral mesh** (Neural-LAM's LAM
construction), **not** a clipped global icosahedron. Finest node set carries a
*superposition* of edges from all coarsening levels (multi-mesh): long edges from
coarse levels give a domain-wide receptive field in few message-passing steps; short
edges from fine levels keep local detail.

**The 0.5° ceiling — read this before setting mesh density.**
At 0.5° the input data already caps spatial detail at ~55 km. A mesh *finer* than the
grid cannot invent sub-55 km structure. So:

- "Mesoscale heavy rainfall" at true meso scale (< 25 km) is **not reachable from 0.5°
  data** — that would require acquiring **0.25°** inputs (a separate acquisition
  decision, out of scope here).
- Given we *do* want maximum rain-map detail and we are memory-rich on the 4070/16 GB
  Kaggle cards, set the **finest mesh near grid density**: **~2,500–4,000 nodes**
  (~55–75 km spacing). Node ratio approaches ~1:1 with the grid; that is acceptable —
  the multi-mesh still earns its keep through the **multi-scale edge structure**, not
  through node compression.

**Levels:** coarsen ~4× per level from the finest set:
```
~3500 → ~875 → ~220 → ~55 → ~14
```
≈ **5 levels (0–4)** to span mesoscale → domain-wide. `n_levels` is a hyperparam;
adding a 6th level buys nothing below the finest resolution.

**Two bipartite graphs, built once:**
- **grid → mesh** (encoder edges) by radius/kNN.
- **mesh → grid** (decoder edges) by radius/kNN.

---

## 5. Node inputs / features

Per **grid node**, per lead:
- **Ensemble-reduced fields:** mean, spread, and **exceedance probabilities** at the
  rainfall thresholds — reuse `ensemble_exceedance_probability` in
  `src/detection/heavy_rainfall.py`.
- **Other-hazard fields:** MSLP, winds, thickness, temperatures (cyclone, depression,
  western disturbance, heat wave, monsoon phase).
- **Static:** orography, land–sea mask, lat/lon positional encoding.
- **Lead-time encoding** appended per lead.

Ensemble → per-cell statistics happens **before** the graph (cheap, and keeps the
graph feature dim fixed regardless of ensemble size).

---

## 6. Architecture

```
                per lead day t = 1..N
grid features ──[Encoder GNN: grid→mesh]──▶ mesh embeddings(t)
                                              │
                          [Processor: ⚙N multi-mesh message-passing layers]
                                              │
   stack over t ─────────────────────────────┘
        │
        ▼
[bi-LSTM over t, per mesh node, shared weights]  ──▶ temporal mesh embedding(t)
        │
        ▼
[Decoder GNN: mesh→grid]  ──▶ per-interior-grid-node bust logit(t)
        │
        ▼
   gridded bust-probability map, Day 1..N   (interior only)
```

- **Bidirectional LSTM** is valid because all lead days are available at once (no
  causality constraint on inputs). Weights shared across mesh nodes.
- Processor depth `N` and hidden width are the main capacity knobs (§9).

---

## 7. Fusion (unchanged contract — `configs/fusion.yaml`)

The fusion layer already expects three inputs:
`calibrated_xgboost_probability`, `calibrated_convlstm_probability`,
`model_disagreement` → logistic regression → `final_overall_bust_probability`,
trained on **out-of-fold / validation-only** predictions.

- The **spatial branch replaces the `convlstm` input** — either keep the config key
  name for compatibility or rename to `graphnet`/`spatial` (decision pending; prefer
  rename + update the config in one commit).
- For the README's single **overall** probability, **pool** the gridded map (e.g.
  area-weighted mean / calibrated aggregation) before fusion. The gridded map itself is
  a first-class output for the dashboard's grid/district/state/region views.
- `model_disagreement` (XGBoost vs. spatial) stays a fusion feature — cheap signal,
  already anticipated.
- **Calibrate each branch before fusion** (isotonic/Platt — `src/models/calibrate.py`).

---

## 8. Loss & labels

- Per-interior-node **binary cross-entropy**, with **class weighting or focal loss**
  (busts are rare — this is the dominant modelling risk).
- **cos-lat area weighting**; **mask** boundary nodes and NaN/invalid cells.
- Labels come from the existing detection stack (`src/detection/…`,
  `overall_bust_label` and the per-category detectors as they're built).

---

## 9. Data-frugality: pretrain then fine-tune

Bust labels are scarce; a graph net this size will overfit if trained from scratch.

- **Pretext task (abundant, unlabeled):** train encoder+processor+decoder to predict
  the **observed field (or the forecast-error field) from the forecast**. This is
  literally "where does the forecast diverge from reality" — the raw material of a
  bust — so it transfers almost directly.
- **Fine-tune:** freeze/warm-start the encoder+processor, train the **bi-LSTM + bust
  head** on the labelled bust events.
- Second line of defence: **fusion with XGBoost** (data-efficient) already in place.

---

## 9a. Data volume — how many years

**Target: ~20 years, anchored to a *reforecast* rather than an operational archive.**
The binding constraints are consistency and rare-event count, not raw length.

- **Consistency > length.** Operational archives (TIGGE, ~2007→) change model version
  every year, so a "bust" is not the same measurement across years. A **reforecast**
  (fixed model version over history) avoids this. **GEFSv12 reforecast = 2000–2019, 20
  clean years**, and doubles as the unlabeled pretraining corpus. Do **not** reach
  pre-2000 for length — it adds model heterogeneity and worse obs.
- **Rare hazards set the floor.** Cyclones (~4–5/yr N. Indian Ocean) give ~80–100 cases
  over 20 years, fewer once restricted to busts. 10 years (~40 cases) is genuinely thin
  for that category; this argues against < ~15 years.
- **Climate regimes.** ENSO/IOD strongly modulate monsoon and cyclones. ~20 years spans
  2–3 full ENSO cycles and mixed IOD phases, so the model doesn't overfit one regime —
  the main reason 20 beats 10.

| Level | Years | Verdict |
|---|---|---|
| Floor | ~10 | Pipeline works; thin on cyclones/WDs, few climate regimes |
| **Target** | **~20** (GEFSv12 reforecast 2000–2019) | Multiple ENSO/IOD cycles, workable rare-event counts, large pretraining set |
| Beyond | >20 | Diminishing returns unless model consistency is preserved |

**Two things that matter as much as the number:**
- **Split temporally, hold out whole years/seasons** — never random split (heavy
  autocorrelation; cf. `tests/test_feature_leakage.py`). E.g. train 2000–2014 /
  val 2015–2016 / test 2017–2019.
- **Compute is not the limiter.** At 0.5° over a 66×70 box, 20 years of daily inits ×
  ~10 leads is a modest tensor volume — trim years for *consistency*, not compute.

**Staged ingest:** validate the pipeline on **2–3 years** first (the repo's "30-day
pilot" is reaching for this), then pretrain + fine-tune on the full ~20. The
pretraining (§9) consumes all 20 years *unlabeled*, so the scarce bust labels only
fine-tune a head — this is what makes ~20 years sufficient despite few labels.

---

## 10. Training on the RTX 4070 (8 GB) — and Kaggle fallback

**Primary: 4070, 8 GB, Ada.**
- **BF16** mixed precision (Ada native; `configs` `mixed_precision: true` already set).
- At ~2,500–4,000 nodes × ~5 edge scales × 9–10 leads, memory is **comfortable**;
  gradient checkpointing likely **not needed for fine-tune**.
- Memory levers, in priority order if it ever gets tight: gradient checkpointing on
  message-passing layers → smaller hidden dim → per-timestep checkpointing of the LSTM
  rollout → batch 1–2 with gradient accumulation.
- Edge count (not node count) drives GNN activation memory — watch it when raising
  `n_levels` or finest density.

**Fallback: Kaggle (only if pretraining needs more VRAM/time).**
- P100 or 2×T4, **16 GB** each — more VRAM than the 4070, but older and **no fast BF16**
  → use **FP16 AMP** (P100 gains little).
- **9-hour session cap, ~30 h/week** → pretraining **must checkpoint & resume** across
  sessions. Build resumable checkpointing into `train_graphnet.py` from day one.
- **Data is the real friction:** preprocess to compact tensors **locally**, upload only
  those as a Kaggle Dataset (mind size limits). Don't try to run raw GRIB decode there.
- **Split:** develop/debug/fine-tune on the 4070; offload long pretraining to Kaggle.

---

## 11. Repo landing

| Now (stub) | Becomes |
|---|---|
| `src/models/convlstm.py` | `src/models/graphnet/` — `mesh.py`, `encoder.py`, `processor.py`, `temporal.py`, `decoder.py`, `model.py` |
| `src/models/train_convlstm.py` | `src/models/train_graphnet.py` (+ resumable checkpointing) |
| `src/features/build_gridded_sequences.py` | grid + mesh tensor builder (feeds the graph) |
| `configs/convlstm.yaml` | `configs/graphnet.yaml` (see §12) |
| `src/models/train_fusion.py`, `calibrate.py` | unchanged contract; consume new branch |

Detection, evaluation, dashboard, API: unchanged.

---

## 12. Hyperparameters (`configs/graphnet.yaml`)

| Key | Default | Notes |
|---|---|---|
| `n_levels` | 5 | multi-mesh coarsening levels (0–4). Tune to data/compute. |
| `finest_nodes` | ~3000 | 2,500–4,000. Near 0.5° grid density; higher = more rain detail, more memory. |
| `processor_depth` | ⚙ | main capacity knob; start shallow, grow if underfitting. |
| `hidden_dim` | 96–128 | not GraphCast's 512 — domain is small. |
| `halo_width` | from grid box | interior/boundary split. |
| `sequence_length` | 9–10 | rainfall retains 9; keep configurable. |
| `precision` | `bf16` (4070) / `fp16` (Kaggle) | |
| `grad_checkpointing` | false (fine-tune) / true (pretrain) | |
| `loss` | focal or weighted BCE | class imbalance. |

---

## 13. Build order (dependencies first)

1. **Mesh builder + grid↔mesh graphs** (`mesh.py`) — fixes all tensor shapes; everything
   downstream depends on it. Reuse Neural-LAM's PyG graph code.
2. **Grid tensor builder** (`build_gridded_sequences.py`) — ensemble-reduced node
   features per lead.
3. **Encoder / processor / decoder** modules → assemble in `model.py` (no temporal yet;
   validate a single-lead forward + backward pass fits in 8 GB).
4. **Bi-LSTM temporal layer** → full sequence forward/backward.
5. **Loss + labels + masking**; overfit a tiny subset to prove correctness.
6. **Pretraining task** (forecast → observed/error) with resumable checkpoints.
7. **Fine-tune** on bust labels; **calibrate**; wire into **fusion**.
8. Pin environment (`requirements-lock.txt` / `environment.yml`).

---

## 14. Open decisions

- Rename fusion input `convlstm` → `graphnet`/`spatial`? (prefer rename + config update).
- Finest node count 2,500 vs 4,000 — pick after the first 8 GB memory measurement.
- Acquire 0.25° inputs later to actually reach mesoscale rain? (data decision, deferred).
- Widen the domain box for longer boundary lead-in? (deferred).
