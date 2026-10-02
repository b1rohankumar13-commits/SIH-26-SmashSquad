"""Live forecast analysis: run the frozen bust models on one forecast from GEFS or NEPS.

    python -m scripts.live_forecast --source gefs --init 20220815            # files already on disk
    python -m scripts.live_forecast --source gefs --init 20260927 --fetch    # download first
    python -m scripts.live_forecast --source gefs --init 20220815 --realtime # force real-time inputs

Sources share one layout (per init YYYYMMDD00.nc, same variables as the operational GEFS store):
  gefs  data/interim/forecasts/gefs/operational/{rain,tmax,mv}  (src/acquisition/download_gefs_operational.py)
  neps  data/interim/forecasts/neps/{rain,tmax,mv}              (scripts/neps_tigge.py: TIGGE -> this layout)

Hazards: rain and storms all year, heatwave for Mar-Jun inits, western disturbances for Dec-Mar
inits (their training seasons). Active/break monsoon is not run live: its inputs come from the
29-variable reforecast archive, which operational downloads do not carry.

Some training inputs are observations that do not exist in real time. When they are missing
(or with --realtime) the run substitutes the forecast system's own previous run and records it:
  heatwave  observed Tmax departure on the 2 days before issue (ERA5) -> previous inits' day-1 Tmax
  wd        yesterday's IMD rain -> previous init's day-1 rain; upstream ERA5 trough -> left at its
            training mean (the forecast crop does not reach 40-65E)
  storms    MJO index (BoM RMM) missing -> left at its training mean
If truth for the valid dates is on disk (IMD rain, ERA5 Tmax, ERA5 storm and WD tracks), the run
is also scored: observed busts, and how many the coloured area caught.

Writes outputs/live/<source>/<init>/<hazard>/{p_miss,p_fa,ens_prob,domain[,obs_miss,obs_fa],
region_*}.npy + explain.json, and outputs/live/<source>/<init>/meta.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from scripts import export_explanations as ex
from scripts.export_dashboard_predictions import (DEV, LEAD01, LEADS, LAT, LON, LPS_CACHE, LPS_CKPTS, OUT as DASH,
                                                   WD_CACHE, WD_CKPTS, heatwave_checkpoints)
from scripts.export_regions import expected_cells, region_any, region_flags
from src.detection.heavy_rainfall import ensemble_exceedance_probability
from src.features.rainfall_labels import regrid_obs_to_grid
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE = PROJECT_ROOT / "outputs" / "live"
CACHES = Path(r"D:\sih-data\caches")
STORES = {"gefs": PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "operational",
          "neps": PROJECT_ROOT / "data" / "interim" / "forecasts" / "neps"}
SOURCE_LABEL = {"gefs": "NOAA GEFS (operational)", "neps": "NCMRWF NEPS-G (via TIGGE)"}
IMD = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
ERA5_TMAX = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "tmax"
ERA5_WD = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "wd"
LPS_TRUTH_ALL = PROJECT_ROOT / "data" / "processed" / "lps_truth_all.npz"
WD_TRUTH = PROJECT_ROOT / "data" / "processed" / "wd_truth.npz"
SEASON = {"heatwave": (3, 4, 5, 6), "wd": (12, 1, 2, 3)}
HAZARDS = ("rain", "heatwave", "lps", "wd")
LATS, LONS = canonical_centres(0.5)


def day(init: str, offset: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"{init[:4]}-{init[4:6]}-{init[6:8]}") + pd.Timedelta(days=offset)


def ymd(t: pd.Timestamp) -> str:
    return t.strftime("%Y%m%d")


def store_file(source: str, kind: str, init: str) -> Path:
    return STORES[source] / kind / f"{init}00.nc"


@lru_cache(maxsize=None)
def land() -> np.ndarray:
    return np.isfinite(regrid_obs_to_grid(xr.open_dataset(IMD / "2005" / "RF25_ind2005_rfp25.nc")).isel(time=0).values)


@lru_cache(maxsize=None)
def imd_year(year: int) -> xr.DataArray | None:
    f = IMD / str(year) / f"RF25_ind{year}_rfp25.nc"
    if not f.exists():
        return None
    o = regrid_obs_to_grid(xr.open_dataset(f)).load()
    o["time"] = pd.DatetimeIndex(o["time"].values).normalize()
    return o


def imd_day(t: pd.Timestamp) -> np.ndarray | None:
    o = imd_year(t.year)
    if o is None or t not in o["time"].values:
        return None
    return o.sel(time=t).values.astype(np.float32)


@lru_cache(maxsize=None)
def era5_tmax_month(year: int, month: int) -> xr.DataArray | None:
    f = ERA5_TMAX / f"era5_tmax_{year}{month:02d}.nc"
    if not f.exists():
        return None
    t = xr.open_dataset(f)["t2m"].rename({"valid_time": "time"}).sortby("latitude").sortby("longitude")
    t = (t.interp(latitude=LATS, longitude=LONS, method="linear") - 273.15).load()
    t["time"] = pd.DatetimeIndex(t["time"].values).normalize()
    return t


def era5_tmax(t: pd.Timestamp) -> np.ndarray | None:
    m = era5_tmax_month(t.year, t.month)
    return None if m is None or t not in m["time"].values else m.sel(time=t).values.astype(np.float32)


def members(ds: xr.Dataset, var: str, level: int | None = None) -> np.ndarray:
    a = ds[var] if level is None else ds[var].sel(level=level)
    return a.isel(run=0).transpose("member", "lead", "latitude", "longitude").values.astype(np.float32)


def predict(models: list, x: np.ndarray) -> np.ndarray:
    """Mean over checkpoints of [lead, lat, lon, 2] probabilities for one input [lead, lat, lon, C]."""
    acc = np.zeros((LEADS, LAT, LON, 2), np.float32)
    with torch.no_grad():
        for m, inp in models:
            acc += m(torch.from_numpy(inp(x)[None]).to(DEV))[0].cpu().numpy()
    return acc / len(models)


# ----------------------------------------------------------------------------- models
@lru_cache(maxsize=None)
def rain_model():
    from src.models.graphnet.grid_graph import build_grid_graph
    from src.models.graphnet.model import GridGraphNet
    ck = torch.load(PROJECT_ROOT / "outputs" / "graphnet_20yr_directional_nbhd24.pt", map_location=DEV)
    m = GridGraphNet(build_grid_graph(), in_channels=24, hidden=ck["hidden"], out_channels=2).to(DEV)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@lru_cache(maxsize=None)
def heat_models(C: int) -> list:
    from scripts.train_heatwave import HeatwaveBustNet
    out = []
    for st in heatwave_checkpoints():
        m = HeatwaveBustNet(st["arch"], C, st["wide_idx"]).to(DEV); m.load_state_dict(st["model_state"]); m.eval()
        out.append(m)
    return out


@lru_cache(maxsize=None)
def ckpt_models(paths: tuple[Path, ...], in_ch: int | None = None) -> list[tuple]:
    from scripts.train_heatwave import HeatwaveBustNet
    out = []
    for p in paths:
        st = torch.load(p, map_location=DEV)
        m = HeatwaveBustNet(st["arch"], in_ch or st["in_channels"], st["wide_idx"]).to(DEV)
        m.load_state_dict(st["model_state"]); m.eval()
        out.append((m, st))
    return out


# ----------------------------------------------------------------------------- hazards
# Each returns: x (model input), models [(module, input_fn)], ens_prob [L,H,W], obs [L,H,W,2] | None,
# substitutions [str], names (x channel names), mean/std (to report physical values), groups.

def day1_departure(path: Path, t: pd.Timestamp, bias: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Forecast day-1 Tmax departure from the run issued on t: the real-time stand-in for the
    observed Tmax departure on t."""
    if not path.exists():
        return np.full((LAT, LON), np.nan, np.float32)
    with xr.open_dataset(path) as d:
        T = d["tmax_2m"].isel(run=0, lead=0).transpose("member", "latitude", "longitude").values - 273.15
    return (T.mean(0) - bias[0, t.month] - normal[t.dayofyear - 1]).astype(np.float32)


def rain_inputs(source: str, init: str) -> dict:
    from src.features.build_gridded_sequences_nbhd import NBHD_CHANNEL_NAMES, build_nbhd_sequences, standardize_nbhd
    from src.features.rainfall_labels import build_directional_bust_labels
    std = np.load(CACHES / "rain_20yr_24ch_nbhd" / "standardizer.npz")
    with xr.open_dataset(store_file(source, "rain", init)) as ds:
        x = standardize_nbhd(build_nbhd_sequences(ds, sequence_length=LEADS), std["mean"], std["std"])[0].astype(np.float32)
        tp = ds["total_precipitation"].isel(run=0)
        ens = np.stack([ensemble_exceedance_probability(tp.sel(lead=L).values, 64.5) for L in tp["lead"].values])
        obs = None
        years = {day(init, L).year for L in range(LEADS)}
        if all(imd_year(y) is not None for y in years):
            truth = xr.concat([imd_year(y) for y in sorted(years)], dim="time")
            obs = build_directional_bust_labels(ds, truth)
            if not np.isfinite(obs).any():
                obs = None
    return {"x": x, "models": [(rain_model(), lambda v: v)], "ens": ens.astype(np.float32), "obs": obs, "subs": [],
            "names": list(NBHD_CHANNEL_NAMES), "mean": std["mean"], "std": std["std"], "groups": ex.RAIN_GROUPS}


def heatwave_inputs(source: str, init: str, realtime: bool) -> dict:
    from scripts.build_heatwave_cache import CHANNELS, MV_VARS
    from src.detection.heatwave import detect_heatwave_bust, ensemble_heatwave_probability, heatwave_day
    train = CACHES / "heatwave_10yr"
    normal = np.load(train / "normal.npy"); bias = np.load(train / "bias.npz")["bias"]
    std = np.load(train / "standardizer.npz")
    # Observed persistence (ERA5) when it covers both days; otherwise both days fall back to the
    # forecast's own day-1 Tmax from the runs issued on those days.
    prev = [day(init, -k) for k in (1, 2)]
    observed = not realtime and all(era5_tmax(t) is not None for t in prev)
    valid = [day(init, L) for L in range(LEADS)]
    vdoy = np.array([v.dayofyear for v in valid]); vmonth = np.array([v.month for v in valid])
    norm_v = normal[vdoy - 1]

    def tmax_c(i: str) -> np.ndarray | None:
        f = store_file(source, "tmax", i)
        if not f.exists():
            return None
        with xr.open_dataset(f) as d:
            T = d["tmax_2m"].isel(run=0).transpose("member", "lead", "latitude", "longitude").values - 273.15
        return T - bias[np.arange(LEADS), vmonth][None]  # training bias correction

    Tc = tmax_c(init)
    P = ensemble_heatwave_probability(Tc, norm_v)
    subs = []

    def dep(back: int) -> np.ndarray:
        t = day(init, -back)
        if observed:
            return era5_tmax(t) - normal[t.dayofyear - 1]
        f = store_file(source, "tmax", ymd(t))
        subs.append(f"observed Tmax {ymd(t)} -> {source.upper()} day-1 forecast from that day" if f.exists()
                    else f"no Tmax for {ymd(t)}: observed-heat input left at its training mean")
        return day1_departure(f, t, bias, normal)

    dep1, dep2 = dep(1), dep(2)
    with xr.open_dataset(store_file(source, "mv", init)) as d:
        mv = [d[v].isel(run=0).transpose("member", "lead", "latitude", "longitude").values for v in MV_VARS]
    ang = 2 * np.pi * vdoy / 365.25
    b = lambda a: np.broadcast_to(a, (LEADS, LAT, LON))
    mean_c = Tc.mean(0)
    chans = [mean_c, Tc.std(0), mean_c - norm_v, P, (Tc >= 40).mean(0), norm_v]
    for f in mv:
        chans += [f.mean(0), f.std(0)]
    chans += [b(dep1), b(dep2), b(np.sin(ang)[:, None, None]), b(np.cos(ang)[:, None, None]),
              b(np.linspace(0, 1, LEADS)[:, None, None]), b(land().astype(np.float32))]
    x = np.nan_to_num((np.stack(chans, -1) - std["mean"]) / std["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    obs = None
    truth = [era5_tmax(v) for v in valid]
    if all(t is not None for t in truth):
        o = np.stack(truth)
        obs = detect_heatwave_bust(P[None], heatwave_day(o, norm_v)[None],
                                   (np.broadcast_to(land(), o.shape) & np.isfinite(o))[None])[0]
    C = len(CHANNELS)
    return {"x": x, "models": [(m, lambda v: v) for m in heat_models(C)], "ens": P.astype(np.float32), "obs": obs,
            "subs": subs, "names": list(CHANNELS), "mean": std["mean"], "std": std["std"], "groups": ex.HEAT_GROUPS}


def lps_inputs(source: str, init: str, realtime: bool) -> dict:
    from scripts.build_lps_cache import GEFS_ZETA_BY_LEAD, process_init
    from scripts.build_lps_env import ENV, load_rmm
    from src.detection.lps_tracker import strike_bust_labels, strike_mask
    from src.features.build_gridded_sequences_nbhd import NBHD_CHANNEL_NAMES
    meta = json.loads((LPS_CACHE / "meta.json").read_text())
    truth = np.load(PROJECT_ROOT / "data" / "processed" / "lps_truth_20yr.npz")
    params = json.loads(str(truth["params"]))
    path = store_file(source, "rain", init)
    _, xl, p300, _ = process_init((0, str(path), truth["mask"], GEFS_ZETA_BY_LEAD, params["deficit_min"], "ops_rain"))
    lstd = np.load(LPS_CACHE / "standardizer.npz")
    xl = (xl - lstd["mean"]) / lstd["std"]
    subs = []
    rmm = load_rmm()
    prev = day(init, -1)
    r = rmm.reindex([prev]).iloc[0]
    if r.isna().any():
        subs.append(f"MJO index for {ymd(prev)} not on disk: left at its training mean")
        r = r.fillna(0.0)
    valid = [day(init, L + 1) for L in range(LEADS)]
    doy = np.array([v.dayofyear for v in valid]); ang = 2 * np.pi * doy / 365.25
    clim = np.load(LPS_CACHE / "clim_strike300.npy")
    xe = np.empty((LEADS, LAT, LON, len(ENV)), np.float32)
    xe[..., 0], xe[..., 1], xe[..., 2] = r["rmm1"], r["rmm2"], r["amp"]
    xe[..., 3] = np.sin(ang)[:, None, None]; xe[..., 4] = np.cos(ang)[:, None, None]; xe[..., 5] = clim[doy - 1]
    estd = np.load(LPS_CACHE / "env_standardizer.npz")
    xe = (xe - estd["mean"]) / estd["std"]
    if "MJO" in " ".join(subs):
        xe[..., :3] = 0.0
    x24 = rain_inputs(source, init)["x"]
    x = np.concatenate([x24, xl.astype(np.float32), xe, LEAD01.astype(np.float32)], -1)
    n24, nl, ne = x24.shape[-1], xl.shape[-1], len(ENV)
    plain = list(range(n24 + nl)) + [n24 + nl + ne]
    models = [(m, (lambda v: v) if st.get("env") else (lambda v, c=plain: v[..., c])) for m, st in ckpt_models(tuple(LPS_CKPTS))]
    obs = None
    z = np.load(LPS_TRUTH_ALL)
    tdates = set(z["dates"].tolist())
    if all(ymd(v) in tdates for v in valid):
        by = {}
        for t, la, lo in zip(z["t"], z["lat"], z["lon"]):
            by.setdefault(str(z["dates"][t]), []).append((float(la), float(lo)))
        o = np.stack([strike_mask(by.get(ymd(v), []), LATS, LONS, 300.0) for v in valid])
        obs = strike_bust_labels(p300, o)
    rstd = np.load(Path(meta["rich_cache"]) / "standardizer.npz")
    names = list(NBHD_CHANNEL_NAMES) + meta["channels"] + list(ENV) + ["lead01"]
    return {"x": x, "models": models, "ens": p300.astype(np.float32), "obs": obs, "subs": subs, "names": names,
            "mean": np.r_[rstd["mean"], lstd["mean"], estd["mean"]], "std": np.r_[rstd["std"], lstd["std"], estd["std"]],
            "groups": ex.LPS_GROUPS, "valid_shift": 0}


def wd_inputs(source: str, init: str, realtime: bool) -> dict:
    from scripts.build_wd_cache import CHANNELS, EVENT_MM, FA_AT, MISS_BELOW, features
    from src.detection import wd_tracker as wd
    from src.detection.lps_tracker import strike_mask
    la, lo = np.meshgrid(LATS, LONS, indexing="ij")
    region = land() & (la >= 24) & (lo <= 85)
    subs = []
    prev_t = day(init, -1)
    obs_prev = None if realtime else imd_day(prev_t)
    if obs_prev is None:
        f = store_file(source, "rain", ymd(prev_t))
        if f.exists():
            with xr.open_dataset(f) as d:
                obs_prev = np.clip(members(d, "total_precipitation"), 0, None).mean(0)[0]
            subs.append(f"IMD rain {ymd(prev_t)} -> {source.upper()} day-1 rain from that day")
        else:
            obs_prev = np.zeros((LAT, LON), np.float32)
            subs.append(f"no rain for {ymd(prev_t)}: yesterday's-rain input set to dry")
    up = None if realtime else upstream_era5(day(init))
    if up is None:
        subs.append("upstream trough (40-65E) not available in real time: left at its training mean")
        up = (np.nan, np.nan)
    _, raw, P = features((0, str(store_file(source, "rain", init)), obs_prev, up[0], up[1], day(init).dayofyear,
                          region.astype(np.float32)))
    std = np.load(WD_CACHE / "standardizer.npz")
    x = np.nan_to_num((raw - std["mean"]) / std["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    obs = None
    truth = [imd_day(day(init, L)) for L in range(LEADS)]
    z = np.load(WD_TRUTH)
    epoch = pd.Timestamp(str(z["epoch"]))
    days = set((z["day"]).tolist())
    need = [(day(init, L) - epoch).days + o for L in range(LEADS) for o in (-1, 0, 1)]
    if all(t is not None for t in truth) and max(need) <= max(days):
        cen: dict[int, list] = {}
        for d, a, b_ in zip(z["day"], z["lat"], z["lon"]):
            cen.setdefault(int(d), []).append((float(a), float(b_)))
        obs = np.full((LEADS, LAT, LON, 2), np.nan, np.float32)
        for L in range(LEADS):
            k = (day(init, L) - epoch).days
            gate = strike_mask(cen.get(k - 1, []) + cen.get(k, []) + cen.get(k + 1, []), LATS, LONS, wd.GATE_KM)
            o = truth[L]; ok = region & np.isfinite(o); heavy = o >= EVENT_MM; ev = heavy & gate
            obs[L, ..., 0] = np.where(ok, np.where(heavy & ~ev, np.nan, (P[L] < MISS_BELOW) & ev), np.nan)
            obs[L, ..., 1] = np.where(ok, (P[L] >= FA_AT) & ~heavy, np.nan)
    return {"x": x, "models": [(m, lambda v: v) for m, _ in ckpt_models(tuple(WD_CKPTS), len(CHANNELS))], "ens": P, "obs": obs,
            "subs": subs, "names": list(CHANNELS), "mean": std["mean"], "std": std["std"], "groups": ex.WD_GROUPS}


def upstream_era5(t: pd.Timestamp) -> tuple[float, float] | None:
    from scripts.build_wd_cache import UPSTREAM
    from src.detection import wd_tracker as wd
    f = ERA5_WD / f"era5_wd_{t.year}{t.month:02d}.nc"
    if not f.exists():
        return None
    ds = xr.open_dataset(f).sel(pressure_level=500.0).sortby("latitude")
    times = pd.DatetimeIndex(ds["valid_time"].values).normalize()
    if t not in times:
        return None
    k = times.get_loc(t)
    lats, lons = ds["latitude"].values, ds["longitude"].values
    a0, a1, o0, o1 = UPSTREAM
    box = ((lats >= a0) & (lats <= a1))[:, None] & ((lons >= o0) & (lons <= o1))[None, :]
    z = np.where(box, wd.smoothed_zeta(ds["u"].values[k], ds["v"].values[k], lats, lons), -np.inf)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    return float(z[i, j]), float(lons[j])


# ----------------------------------------------------------------------------- outputs
def region_outputs(h: str, entry: dict, p: np.ndarray, domain: np.ndarray, obs: np.ndarray | None) -> dict:
    reg = np.load(DASH / "regions.npy"); R = int(reg.max()) + 1
    out = {}
    for j, d in enumerate(("miss", "false_alarm")):
        t = entry["tiers"][d]; knots = entry["regions"]["calibration"][d]; cut = ex.display_cut(t)
        P = np.zeros((LEADS, R), np.float16); F = np.zeros((LEADS, R), np.int8); O = np.full((LEADS, R), -1, np.int8)
        E = np.zeros((LEADS, R), np.float16)
        for L in range(LEADS):
            pc = ex.calibrated(p[L, ..., j], t["calibration"], L + 1)
            s = expected_cells(pc[None], domain[None], reg, R); E[L] = s[0]
            P[L] = np.interp(s[0], knots[L]["x"], knots[L]["y"]) if len(knots[L]["x"]) > 1 else knots[L]["y"][0]
            F[L] = region_flags(((pc >= cut) & domain)[None], reg, R)[0]
            if obs is not None:
                o = np.where(np.isfinite(obs[L, ..., j]), obs[L, ..., j], -1).astype(np.int8)
                O[L] = region_any(o[None], (domain & (o >= 0))[None], reg, R)[0]
        key = "miss" if d == "miss" else "fa"
        out[f"region_p_{key}"], out[f"region_flag_{key}"], out[f"region_obs_{key}"], out[f"region_exp_{key}"] = P, F, O, E
    return out


def explain(h: str, entry: dict, inp: dict, p: np.ndarray, domain: np.ndarray, init: str,
            source: str = "gefs") -> dict:
    found = {}
    for j, d in enumerate(("miss", "false_alarm")):
        t = entry["tiers"][d]; cut = ex.display_cut(t)
        for L in range(LEADS):
            cal = ex.calibrated(p[L, ..., j], t["calibration"], L + 1)
            flag = (cal >= cut) & domain
            from scipy import ndimage
            lab, n = ndimage.label(flag, structure=np.ones((3, 3)))
            if n == 0:
                continue
            exp = ndimage.sum(cal, lab, index=np.arange(1, n + 1))
            size = ndimage.sum(np.ones_like(cal), lab, index=np.arange(1, n + 1))
            keep = [k for k in np.argsort(-exp) if size[k] >= ex.MIN_CELLS][:ex.TOP_AREAS]
            if keep:
                found[(0, L, d)] = [(lab == k + 1, float(exp[k])) for k in keep]
    if not found:
        return {}

    class Wrap(torch.nn.Module):
        def __init__(self, m, fn):
            super().__init__(); self.m, self.fn = m, fn

        def forward(self, x):
            return self.m(self.fn(x))

    x = inp["x"]
    fwd = [(Wrap(m, fn), None) for m, fn in inp["models"]]
    ex._INITS[f"live-{h}"] = [init]
    raw = ex.raw_reader(inp["names"], inp["mean"], inp["std"], lambda r: x)
    res = ex.attribute(fwd, lambda r, needs: x, [0], inp["names"], inp["groups"], raw, found, 1, f"live-{h}")
    out = res.get(init, {})
    if source != "gefs":  # driver names say GEFS; name the system actually analysed
        for per in out.values():
            for areas in per.values():
                for area in areas:
                    for d in area["drivers"]:
                        d["name"] = d["name"].replace("GEFS", source.upper())
    return out


def validation(entry: dict, p: np.ndarray, obs: np.ndarray, domain: np.ndarray) -> dict:
    """Observed busts and how many fell inside the coloured area (the map's cut-off)."""
    out = {}
    for j, d in enumerate(("miss", "false_alarm")):
        t = entry["tiers"][d]; cut = ex.display_cut(t)
        busts = caught = flagged = 0
        for L in range(LEADS):
            o = obs[L, ..., j]; ok = np.isfinite(o) & domain
            shown = (ex.calibrated(p[L, ..., j], t["calibration"], L + 1) >= cut) & ok
            busts += int((o[ok] == 1).sum()); flagged += int(shown.sum()); caught += int((shown & (o == 1)).sum())
        out[d] = {"busts": busts, "flagged": flagged, "caught": caught}
    return out


def run(source: str, init: str, hazards: list[str], realtime: bool = False) -> dict:
    index = json.loads((DASH / "index.json").read_text())
    root = LIVE / source / init
    root.mkdir(parents=True, exist_ok=True)
    meta = {"source": source, "source_label": SOURCE_LABEL[source], "init": init, "realtime": realtime,
            "created": pd.Timestamp.now().isoformat(timespec="seconds"), "hazards": {}}
    builders = {"rain": lambda: rain_inputs(source, init), "heatwave": lambda: heatwave_inputs(source, init, realtime),
                "lps": lambda: lps_inputs(source, init, realtime), "wd": lambda: wd_inputs(source, init, realtime)}
    for h in hazards:
        t0 = time.time()
        if h in SEASON and day(init).month not in SEASON[h]:
            meta["hazards"][h] = {"status": "out_of_season"}
            print(f"{h}: out of season", flush=True)
            continue
        need = {"heatwave": ("tmax", "mv")}.get(h, ("rain",))
        missing = [k for k in need if not store_file(source, k, init).exists()]
        if missing:
            meta["hazards"][h] = {"status": "missing_input", "missing": missing}
            print(f"{h}: missing {missing}", flush=True)
            continue
        inp = builders[h]()
        p = predict(inp["models"], inp["x"])
        entry = index[h]
        domain = np.load(DASH / h / "domain.npy")
        d = root / h; d.mkdir(exist_ok=True)
        arrays = {"p_miss": p[..., 0].astype(np.float16), "p_fa": p[..., 1].astype(np.float16),
                  "ens_prob": inp["ens"].astype(np.float16), "domain": domain}
        obs = inp["obs"]
        if obs is not None:
            arrays["obs_miss"] = np.where(np.isfinite(obs[..., 0]), obs[..., 0], -1).astype(np.int8)
            arrays["obs_fa"] = np.where(np.isfinite(obs[..., 1]), obs[..., 1], -1).astype(np.int8)
        arrays.update(region_outputs(h, entry, p, domain, obs))
        for k, v in arrays.items():
            np.save(d / f"{k}.npy", v)
        (d / "explain.json").write_text(json.dumps(explain(h, entry, inp, p, domain, init, source), separators=(",", ":"),
                                                   ensure_ascii=False), encoding="utf-8")
        info = {"status": "ok", "substitutions": inp["subs"], "validated": obs is not None,
                "valid_shift": inp.get("valid_shift", -1), "seconds": round(time.time() - t0, 1)}
        if obs is not None:
            info["validation"] = validation(entry, p, obs, domain)
        meta["hazards"][h] = info
        print(f"{h}: ok in {info['seconds']}s | substitutions {len(inp['subs'])} | "
              f"validated {obs is not None}" + (f" {info['validation']}" if obs is not None else ""), flush=True)
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def fetch(source: str, init: str, realtime: bool) -> None:
    """Download one init into the source's store (plus the two days before when the real-time
    stand-ins need that system's own previous runs)."""
    py = sys.executable
    days = [ymd(day(init, -2)), ymd(day(init, -1)), init] if realtime else [init]
    for i in days:
        if store_file(source, "rain", i).exists():  # already downloaded: never fetch twice
            print(f"fetch: {source} {i} already on disk, skipped", flush=True)
            continue
        if source == "gefs":
            cmd = [py, "-m", "src.acquisition.download_gefs_operational", "--date", i]
        else:
            cmd = [py, "-m", "scripts.neps_tigge", "--date", i]
        print("fetch: " + " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def newest_available(source: str) -> str:
    """Newest 00 UTC run that should be downloadable now: GEFS posts a run ~5 h after 00 UTC;
    TIGGE releases NEPS after a 48-hour embargo."""
    now = pd.Timestamp.now(tz="UTC")
    lag = pd.Timedelta(hours=6) if source == "gefs" else pd.Timedelta(hours=50)
    return ymd((now - lag).tz_localize(None).normalize())


def latest(source: str) -> str:
    files = sorted(glob.glob(str(STORES[source] / "rain" / "*.nc")))
    if not files:
        raise SystemExit(f"No {source} forecasts on disk; pass --init and --fetch.")
    return Path(files[-1]).stem[:8]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(STORES), default="gefs")
    ap.add_argument("--init", help="YYYYMMDD (default: latest on disk)")
    ap.add_argument("--fetch", action="store_true", help="download the forecast first")
    ap.add_argument("--realtime", action="store_true", help="use the real-time stand-ins even where observations exist")
    ap.add_argument("--hazards", default=",".join(HAZARDS))
    args = ap.parse_args()
    init = args.init or (newest_available(args.source) if args.fetch else latest(args.source))
    if args.fetch:
        fetch(args.source, init, args.realtime)
    t0 = time.time()
    run(args.source, init, args.hazards.split(","), args.realtime)
    print(f"done in {time.time() - t0:.0f}s -> {LIVE / args.source / init}", flush=True)


if __name__ == "__main__":
    main()
