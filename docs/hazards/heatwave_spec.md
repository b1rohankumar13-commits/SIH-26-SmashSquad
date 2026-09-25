# Heatwave Bust Model — Spec (v1)

Per-cell forecast-**bust** prediction for heatwaves: flag grid cells where the GEFS
ensemble is *confidently wrong* about heatwave occurrence (miss or false alarm),
at 1–9 day lead. Same "bust" framing and per-cell map output as extreme rainfall;
different variables, label, and observation source.

Archetype: **B — smooth persistent field** (see `docs/hazards/README` archetype map).
Backbone bake-off: **apt-GNN (primary) vs ConvLSTM (existing baseline)**.

---

## 1. Data

### Predictors — GEFS reforecast, 2000–2009, 5 members (c00,p01–p04), 9 lead days
Subset of the 29-var multivar set **+ one new field**:

| # | Channel | Source | Notes |
|---|---|---|---|
| 1–2 | `tmax_2m` mean, spread | **NEW download** | PRIMARY. `tmp_2m` alone is 00Z-instant (~05:30 IST ≈ daily *min*), useless for heatwave. |
| 3–4 | `tmp_2m` mean, spread | have | 00Z temp = diurnal/min context |
| 5–6 | `tmp_850` mean, spread | have | lower-trop warmth (less surface noise) |
| 7–8 | `hgt_500` mean, spread | have | mid-trop ridge (heat dome) |
| 9–10 | `hgt_300` mean, spread | have | upper ridge |
| 11–12 | `spfh_2m` mean, spread | have | humid-heat / apparent temp |
| 13–14 | `pres_msl` mean, spread | have | subsidence / anticyclone |
| 15 | P(`tmax_2m` ≥ 40 °C) | derived | ensemble exceedance fraction |
| 16 | P(Tmax departure ≥ 4.5 °C) | derived | IMD heatwave criterion, needs climatology |
| 17 | Tmax climatological normal | static | per-cell day-of-year normal (for departures) |

~17 channels (optionally +land mask, +sin/cos DoY → 19). **Cache ≈ 11 GB** for 10 yr
(n≈3604, 17×9×66×70 float32).

### Observations (labels) — IMD gridded daily **Tmax** — **MISSING, must download**
We currently hold IMD **rainfall** only (`data/raw/observations/imd/rainfall/`).
Need IMD 1°×1° daily max-temperature (`Maxtemp` .grd, IMD Pune), 2000–2009, regridded
to our 0.5° India grid. Tiny (~MBs/yr). `download_imd.py` header says it already
covers "rainfall and temperature" — likely just a config entry.

---

## 2. Target — heatwave bust label `Y[n, lead, H, W] ∈ {0,1}`

**Observed heatwave day** at cell *c*, valid day *d* (IMD plains criterion, v1 simple):
`Tmax(c,d) ≥ 40 °C` AND `departure(c,d) ≥ 4.5 °C`, occurring in a **≥2-day spell**
(persistence). (v2: region-aware thresholds — coastal ≥37, hills ≥30.)

**Ensemble heatwave prob** at (c, valid day d, lead L):
fraction of the 5 members meeting the criterion at that cell/lead.

**Bust** (mirrors rain thresholds, tune on calibration):
- **MISS**: ensemble prob `< 0.20` but heatwave observed → bust = 1
- **FALSE_ALARM**: ensemble prob `≥ 0.50` but no heatwave observed → bust = 1
- else → bust = 0

5 members ⇒ coarse prob grid {0,.2,.4,.6,.8,1}, same as rain. Fine.

---

## 3. Models (same cache, same target, swap backbone)

**A. apt-GNN (primary).** 66×70 grid graph (8-neighbour) **+ region supernodes**;
node features = 17ch; **TCN over the 9 leads** (persistence is the signal). Two heads
off the shared encoder:
- per-cell bust logit `[n, lead, H, W]`
- region-band bust logit (supernodes) — joint auxiliary loss, coarse-scale regularizer.
Reuse `src/models/graphnet/model.py`; use `multimesh_model.py` if wider receptive
field helps (heatwave is smooth, so local likely suffices — supernodes mainly for the
region head).

**B. ConvLSTM (baseline).** `src/models/convlstm.py` + `train_convlstm.py`, input
`[ch, lead, H, W]`, per-cell logit. Near-zero extra engineering — record one number.
This is the honest test of "does message-passing beat a conv baseline on a smooth field?"

---

## 4. Cache layout — `D:\sih-data\caches\heatwave_10yr\`
```
X.dat           float32 [n, ch, lead, H, W]      standardized predictors
Y.dat           float32 [n, lead, H, W]          bust 0/1
P.dat           float32 [n, lead, H, W]          ensemble heatwave prob (calib/eval)
mask.dat        uint8   [H, W]                    valid India-land cells
meta.json       {n, ch, channel_names, leads, H, W, dates[], thresholds}
standardizer.npz per-channel mean/std
```
Point `SIH_CACHE_DIR=D:\sih-data\caches\heatwave_10yr`.

---

## 5. Wiring (fill existing slots — no new top-level structure)
- **Label:** implement `src/detection/heatwave.py::detect_heatwave_bust(...)`
- **Cache:** generalize `src/features/build_gridded_sequences.py` to take (var-list, label fn)
- **Registry:** add `heatwave` entry to new `configs/hazards.yaml` (vars, detector, backbones, bands)
- **Train:** hazard-parametrized trainer `--hazard heatwave --backbone {gnn,convlstm}`
- **Eval:** `src/evaluation/evaluate_by_{category,lead,region}.py` (category = hazard)

## 6. Eval
PR-AUC primary (low base rate) vs base-rate no-skill line; per-lead, per-region;
reliability curve; catch-rate/precision operating point. Same harness as rain.

---

## 7. Build order
1. Add IMD Tmax obs download (2000–2009) — small
2. Add GEFS `tmax_2m` download (2000–2009, 5 members) — moderate (~a few hrs)
3. Fill `detect_heatwave_bust`
4. Build cache → D:
5. `hazards.yaml` entry
6. Train GNN + ConvLSTM, calibrate per-lead, eval

## 8. Open decisions
1. **Approve two extra downloads** — IMD Tmax obs (tiny) + GEFS `tmax_2m` (moderate).
2. **Threshold scheme** — v1 simple (Tmax≥40 & anom≥4.5 everywhere) vs region-aware.
3. **Region bands** — IMD subdivisions (~36) vs latitude bands (simpler).
