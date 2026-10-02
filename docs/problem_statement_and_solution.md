# SIH26079 — AI-Based Forecast Bust Detection: what we built

**Problem statement (MoES / NCMRWF):** detect, ahead of time, where and when an operational
ensemble forecast is likely to fail ("bust") for high-impact Indian weather, and explain why.

**Our answer, BustSentinel:** a reliability layer on top of an existing ensemble forecast. It does
not make a new weather forecast; for every grid cell, IMD subdivision and lead day it says how
likely the forecast already issued is to be confidently wrong, in which direction, and which
meteorological signals drive that risk.

---

## 1. Expected outcomes — where we stand

| PS expected outcome | Status | Where |
|---|---|---|
| Region-wise forecast confidence map, Day 1–10 | Done for **Days 1–9** (Day 10 out of scope, see §8) | Forecast Desk → *Confidence* view; per-subdivision table D1–D9 |
| Bust probability per region | Done: per 0.5° cell and per IMD subdivision (36), for every hazard and lead | Desk → *Cells* / *Subdivisions*; `/api/regions/{hazard}`; CSV bulletin |
| Error-prone area detection | Done: flagged cells and subdivisions, largest flagged areas ranked by expected busts | Map + "Where" cards; subdivision ranking |
| Explainable output (key meteorological reasons) | Done: top drivers per flagged area and per subdivision, with physical values | "Why" cards; hover any driver for its meaning |
| Prototype dashboard / API for operational use | Done: Next.js dashboard + FastAPI; live runs on today's GEFS or NCMRWF NEPS; CSV bulletin | `/` desk, `/case-studies`, `/live`; `/api/*` |

---

## 2. Hazards covered

| Hazard | Truth | Event | Season |
|---|---|---|---|
| Extreme rainfall | IMD 0.25° gridded rain | ≥ 64.5 mm/day (IMD "heavy") | all year |
| Heatwave | ERA5 daily Tmax | IMD rule: ≥ 40 °C and ≥ 4.5 °C above normal | Mar–Jun |
| Low-pressure systems (lows, depressions, cyclones) | ERA5 + our own tracker (850 hPa vorticity ≥ 4e-5 s⁻¹ and MSLP dip ≥ 2 hPa; 97% IBTrACS hit rate) | system centre within 300 km | all year |
| Western disturbances | ERA5 500 hPa vorticity tracker + IMD rain | ≥ 35.6 mm/day over NW India with a tracked WD within 1200 km | Dec–Mar |
| Active / break monsoon | IMD rain over the monsoon core zone (18–28N, 73–86E) | Rajeevan-style index beyond ±1σ for ≥ 3 days | Jun–Sep |

Forecast: **NOAA GEFSv12** (5-member reforecast 2000–2019 for training/validation; operational
2022 for the held-out test; operational today for live runs). NCMRWF **NEPS-G** is supported for
live runs via ECMWF TIGGE (§6).

---

## 3. What a "bust" is

A bust is a forecast that was **confidently wrong**, split by direction, because the two have
different causes and different costs:

- **Miss:** the ensemble gave the event < 20% probability (P < 0.2), and it happened.
- **False alarm (FA):** the ensemble gave ≥ 50% (P ≥ 0.5), and it did not happen.

P is the fraction of GEFS members forecasting the event (heatwave: after removing GEFS's warm Tmax
bias per lead, month and cell; storms: tracker thresholds matched to ERA5 frequency per lead).
Between 0.2 and 0.5 the forecast is "uncertain", not wrong, so it cannot bust. Skill is therefore
scored **only on eligible cells** (P < 0.2 for misses, P ≥ 0.5 for FAs); scoring over all cells
inflates AUROC to near 1 and was rejected.

Each hazard has one model with two heads, P(miss) and P(FA), per cell and lead.

---

## 4. Models

**Inputs (per cell, per lead):** ensemble mean and spread of the forecast fields, plus derived
signals: exceedance probabilities, neighbourhood probabilities (near-miss in position), moisture
flux, low-level convergence and vorticity, 500 hPa gradient; for storms and WDs the per-member
tracks (share of members with a system nearby), MJO, seasonal climatology and an upstream ERA5
trough at issue time; for heat the observed anomaly of the last two days.

**Architecture — wide-and-deep, the key design choice.** Bust labels are built from the ensemble
probability, so an ensemble-only logistic regression is already a strong baseline; plain deep
models failed to beat it on rare labels. We fit that logistic first, freeze it as the *wide* path,
and train a spatial encoder (the *deep* path) to learn only the correction. The deep path is:

| Hazard | Deep path |
|---|---|
| Rainfall | GridGraphNet: GraphSAGE over the 0.5° grid + temporal conv across leads (24 channels) |
| Heatwave | GNN + CNN, 2 seeds each, 4 checkpoints averaged (errors are complementary) |
| Low-pressure systems | Two GNNs averaged (one adds MJO + storm climatology), trained 2000–2015 |
| Western disturbances | GNN + CNN, 4 checkpoints averaged, trained Dec–Mar 2000–2015 |
| Active / break | GNN with frozen ensemble-baseline head, pooled over the core zone |

Bake-offs we ran and rejected: multi-resolution mesh GNN (no gain, 2× slower), ConvLSTM (last on
heat), XGBoost fusion (redundant once exceedance channels were inputs), self-supervised
pretraining (same), lagged ensembles (+0.002), a GEFS storm-track correction model (systematic
position error is < 1.2% of the total).

**Calibration:** isotonic regression per lead, fitted on the validation years and frozen.

**Subdivisions:** IMD's 36 meteorological subdivisions (rebuilt from Survey of India state
boundaries). Each subdivision gets the **expected number of busting cells**, Σ pᵢ over its cells,
calibrated per lead into a probability that the subdivision busts. **Confidence** that the
forecast holds = (1 − P(miss)) (1 − P(FA)), graded Low / Medium / High at the validation 90th / 70th
percentile of risk with floors of 25% / 10% bust chance. A subdivision is shaded on the map only
when the grid itself flags ≥ 3 of its cells, so one hotspot does not paint a whole state.

---

## 5. Results

Frozen models, eligible cells, **model vs GEFS alone** (the ensemble-only baseline).

**Miss detection — the headline.**

| Hazard | Validation AUROC | Validation PR-AUC | 2022 held-out PR-AUC |
|---|---|---|---|
| Extreme rainfall (val 2016–19) | 0.941 vs 0.894 | 0.122 vs 0.041 (3.0×) | 0.139 vs 0.051 (AUROC 0.940 vs 0.898) |
| Heatwave | 0.978 vs 0.904 | 0.142 vs 0.018 (7.7×) | **0.204 vs 0.045** (AUROC 0.935 vs 0.857) |
| Low-pressure systems (val 2016–19) | 0.956 vs 0.890 | 0.117 vs 0.041 (2.8×) | 0.119 vs 0.051 |
| Western disturbances (val 2016–19) | 0.982 vs 0.907 | 0.171 vs 0.016 (11×) | 0.156 vs 0.022 (AUROC 0.977 vs 0.924) |
| Active monsoon (val 2008–09) | 0.943 vs 0.842 | 0.347 vs 0.158 | — |

Base rates are 0.1–0.5% of eligible cells, so a PR-AUC of 0.12–0.20 is 25–60× chance.
On 2022, coloured cells bust 12% (rain) and 15% (heat) of the time vs 0.3–0.5% elsewhere.

**False alarms.** Heatwave FA has real skill (AUROC 0.717 vs 0.316; 2022: 0.653 vs 0.439), and
the monsoon FA targets improve slightly. For rain, storms and WDs the model adds nothing over the
ensemble: when GEFS gives ≥ 50% for these, it busts 75–84% of the time regardless. We report this
as a calibration finding about GEFS (overconfident on extremes), not as model skill.

**Per subdivision (miss AUROC, validation):** rain 0.922 vs 0.883; heatwave 0.946 vs 0.824;
storms 0.910 vs 0.849; WD 0.979 vs 0.910.

**Comparison.** The closest public SIH26079 solution scores a combined miss+FA label at
subdivision-day level, Days 3–7: AUROC 0.840, +0.032 over the IFS ensemble spread. Ours, miss-only
on 0.5° cells, Days 1–9: rain AUROC 0.940, +0.042 over GEFS alone on the same held-out year.

**Live inputs.** Two heatwave inputs (observed Tmax of the last two days) are only available
after the fact in the reforecast. In real time we substitute a stand-in; on 2022 this drops heatwave
miss PR-AUC from 0.204 to 0.126 (still 2.8× GEFS alone). With an operational observed-Tmax feed
(IMD), the 0.204 figure applies. WD is unaffected (0.158 vs 0.167).

---

## 6. Explainability

For every flagged area, **gradient × input** on the calibrated model's logit attributes the bust
risk to each input channel; channels are grouped into named meteorological drivers (e.g. "850 hPa
moisture", "WD approaching from the west"), and each area shows its top 4 drivers with their share
of the push towards a bust and their physical value (mm, °C, %, or σ from normal). Subdivisions
inherit the drivers of the flagged areas inside them, weighted by expected busts. Monsoon
explanations attribute over the core zone. Hovering any driver in the dashboard shows a one-line
definition.

None of this needs the observed weather: risk maps, explanations, subdivisions and bulletins are
computed from the forecast alone. Observations are only used to train, to fit the frozen
calibration, and optionally to *Validate* a past forecast (rings on the cells that actually busted).

---

## 7. Operational prototype

- **Dashboard (Next.js):** Forecast Desk (cells / subdivisions / confidence views, Days 1–9,
  miss / FA / both, "does it beat GEFS alone?" skill card, reasons), Case Studies (Kerala 2018,
  cyclones Ockhi / Fani / Bulbul, 2009 heatwaves incl. one that never came, WDs Feb and Dec 2019), Live Forecast.
- **API (FastAPI):** `/api/grid`, `/api/regions/{hazard}`, `/api/explain/{hazard}`,
  `/api/bulletin/{hazard}.csv` (per subdivision and day: confidence, grade, bust chance, expected
  busting cells, main reasons), `/api/live/*`.
- **Live runs:** downloads today's **GEFS** (NOAA, ~6 h after 00 UTC) or **NCMRWF NEPS-G** (ECMWF
  TIGGE, 48 h embargo; control + 4 members regridded to 0.5°), runs all in-season hazards in
  about a minute, optionally every day automatically. NEPS runs zero-shot through the GEFS-trained
  models; comparing NEPS against GEFS on the same date works, but NEPS needs its own tracker
  thresholds and Tmax bias correction before its storm and heat risks are trustworthy (needs NEPS
  hindcasts). A demo mode replays forecasts already on disk.

---

## 8. Limitations

- **Day 10:** our training data and models cover leads 1–9; Day 10 was left out of scope for
  time. Adding it means one more lead in the downloads and a retrain, not a design change.
- **False alarms** carry skill only for heat (see §5).
- **2022 heatwave** was a record year (GEFS bias +2.1 °C vs +1.4 °C in training); the model still held.
- **Monsoon active/break** is validated on 2008–09 only and not yet available live.
- **Five members:** with GEFS's 5 members, a miss means no member forecast the event at all; a
  larger ensemble (NEPS, 12 members) would give finer probabilities once the models are adapted.

---

## 9. Who uses it

- **IMD / NCMRWF forecasters:** where and why today's ensemble is fragile, before issuing warnings.
- **NDMA / state disaster authorities:** a "low confidence" subdivision is a cue to prepare for the
  worse outcome rather than trust a calm forecast.
- **Sector users:** reservoir and urban-flood managers (rain), coastal and fisheries (storms),
  health and power (heat), north-India agriculture and transport (WDs), agromet advisories (monsoon).

Repository map: `src/detection/` (bust labels, trackers), `src/features/` (inputs),
`scripts/train_*` / `export_*` (models, dashboard exports), `scripts/live_forecast.py` (live
pipeline), `api/` (FastAPI), `web/` (dashboard).
