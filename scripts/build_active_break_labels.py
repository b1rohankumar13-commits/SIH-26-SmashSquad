"""Build active/break regime-bust labels for monsoon-season (JJAS) inits, 2000-2009.

Output: data/processed/active_break_labels.npz - one row per (init, lead) whose valid
day falls in JJAS, with ensemble regime probabilities, observed state, the four bust
masks, and `cache_row` (row of the multivar rich cache, which uses the same sorted
file list). Prints base rates and an ensemble-only logistic baseline.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.detection.monsoon_phase import (
    area_mean, detect_monsoon_phase_bust, mcz_mask, regime_states, windowed_climatology,
)
from src.features.rainfall_labels import regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MV_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019_mv"
OBS_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
OUT = PROJECT_ROOT / "data" / "processed" / "active_break_labels.npz"
JJAS = (6, 7, 8, 9)
PREFIX_DAYS = 2  # observed days before init prepended for the persistence rule


def main() -> None:
    lats, lons = canonical_centres(0.5)
    mask = mcz_mask(lats, lons)
    print(f"MCZ cells: {int(mask.sum())}", flush=True)

    # --- observed MCZ index 2000-2019 -------------------------------------------
    parts = []
    for year in range(2000, 2020):
        grid = regrid_obs_to_grid(xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))
        parts.append(pd.Series(area_mean(grid.values, mask), index=pd.DatetimeIndex(grid["time"].values)))
    obs = pd.concat(parts).sort_index()
    all_files = sorted(f for y in range(2000, 2010) for f in glob.glob(str(MV_ROOT / f"{y}*.nc")))
    n_train = int(0.8 * len(all_files))  # same split as the rich cache / trainers
    val_start = pd.Timestamp(Path(all_files[n_train]).stem[:8])
    clim_mask = (obs.index < val_start) | (obs.index.year >= 2010)  # exclude val years
    doy = obs.index.dayofyear.values
    mu, sd = windowed_climatology(doy[clim_mask], obs.values[clim_mask], np.arange(1, 367))
    obs_anom = pd.Series((obs.values - mu[doy - 1]) / sd[doy - 1], index=obs.index)
    obs_state = pd.Series(regime_states(obs_anom.values), index=obs.index)
    print(f"obs index {obs.index[0].date()}..{obs.index[-1].date()} | val from {val_start.date()}", flush=True)

    # --- forecast MCZ index per member/lead for inits touching JJAS -------------
    rows, inits, fc = [], [], []
    for row, path in enumerate(all_files):
        init = pd.Timestamp(Path(path).stem[:8])
        if not (pd.Timestamp(init.year, 5, 24) <= init <= pd.Timestamp(init.year, 9, 30)):
            continue
        with xr.open_dataset(path) as ds:
            pr = ds["total_precipitation"].isel(run=0).transpose("member", "lead", "latitude", "longitude")
            fc.append(area_mean(pr.values, mask))  # [member, lead]
        rows.append(row); inits.append(init)
        if len(inits) % 200 == 0:
            print(f"  read {len(inits)} forecasts", flush=True)
    fc = np.stack(fc)  # [n_init, member, lead]
    n_init, n_mem, n_lead = fc.shape
    inits = pd.DatetimeIndex(inits); rows = np.array(rows)
    valid = np.array([[init + pd.Timedelta(days=d) for d in range(n_lead)] for init in inits])
    vdoy = np.vectorize(lambda t: t.dayofyear)(valid)
    is_train = (inits < val_start)[:, None].repeat(n_lead, 1)

    # model climatology per lead (train inits only), standardise every member
    fc_anom = np.full(fc.shape, np.nan)
    for L in range(n_lead):
        m_doy = np.repeat(vdoy[:, L][:, None], n_mem, 1)
        tr = is_train[:, L]
        m_mu, m_sd = windowed_climatology(m_doy[tr].ravel(), fc[tr, :, L].ravel(), np.arange(1, 367))
        fc_anom[:, :, L] = (fc[:, :, L] - m_mu[vdoy[:, L] - 1][:, None]) / m_sd[vdoy[:, L] - 1][:, None]

    prefix = np.array([[obs_anom.get(init - pd.Timedelta(days=k), np.nan) for k in range(PREFIX_DAYS, 0, -1)]
                       for init in inits])  # [n_init, PREFIX_DAYS]
    traj = np.concatenate([np.repeat(prefix[:, None, :], n_mem, 1), fc_anom], axis=2)
    member_state = regime_states(np.nan_to_num(traj, nan=0.0))[:, :, PREFIX_DAYS:]  # [n_init, member, lead]
    p_active = (member_state == 1).mean(1)
    p_break = (member_state == -1).mean(1)

    o_state = np.vectorize(lambda t: obs_state.get(t, 0))(valid).astype(np.int8)
    o_anom = np.vectorize(lambda t: obs_anom.get(t, np.nan))(valid)
    in_jjas = np.vectorize(lambda t: t.month in JJAS)(valid)
    in_ja = np.vectorize(lambda t: t.month in (7, 8))(valid)
    busts = detect_monsoon_phase_bust(p_active, p_break, o_state)

    keep = in_jjas & np.isfinite(o_anom)
    lead_idx = np.repeat(np.arange(1, n_lead + 1)[None, :], n_init, 0)
    out = {
        "cache_row": np.repeat(rows[:, None], n_lead, 1)[keep], "lead": lead_idx[keep],
        "init": np.repeat(np.asarray(inits.strftime("%Y%m%d"), dtype=str)[:, None], n_lead, 1)[keep],
        "p_active": p_active[keep], "p_break": p_break[keep],
        "ens_mean_anom": np.nanmean(fc_anom, 1)[keep], "ens_spread": np.nanstd(fc_anom, 1)[keep],
        "obs_state": o_state[keep], "obs_anom": o_anom[keep],
        "is_train": is_train[keep], "in_ja": in_ja[keep],
        **{k: v[keep] for k, v in busts.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)

    tr, va = out["is_train"], ~out["is_train"]
    print(f"\nsaved {OUT} | {keep.sum():,} (init,lead) rows | train {tr.sum():,} val {va.sum():,}")
    days = obs_state[obs_state.index.month.isin(JJAS)]
    print(f"observed JJAS days: active {100*(days==1).mean():.1f}% | break {100*(days==-1).mean():.1f}%")
    for k in ("miss_break", "fa_break", "miss_active", "fa_active", "bust"):
        print(f"  {k:<12} base {100*out[k].mean():5.2f}% | val positives {int(out[k][va].sum())}")

    # ensemble-only baseline: logistic regression on the ensemble's own regime signal
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score as ap
    feats = np.column_stack([out["p_active"], out["p_break"], out["ens_mean_anom"],
                             out["ens_spread"], out["lead"]])
    feats = np.nan_to_num(feats)
    print("\nEnsemble-only logistic baseline (val PR-AUC vs base rate):")
    for k in ("miss_break", "fa_break", "miss_active", "fa_active", "bust"):
        y = out[k].astype(int)
        if y[tr].sum() < 5 or y[va].sum() < 1:
            print(f"  {k:<12} too few positives"); continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(feats[tr], y[tr])
        s = clf.predict_proba(feats[va])[:, 1]
        print(f"  {k:<12} PR-AUC {ap(y[va], s):.3f}  (base {y[va].mean():.3f})")


if __name__ == "__main__":
    main()
