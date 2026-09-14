"""Pretrain the GNN encoder on a dense forecast target, then fine-tune on the rare bust labels.

Two pretexts, both dense (every land cell/day) so they sidestep the bust-label rarity:
  error       regress log1p(obs) - log1p(forecast rain); learns average error magnitude.
  exceedance  regress 1[obs >= t] - P(ensemble >= t) at the bust thresholds; tail-focused,
              so the encoder learns the extremes where busts live.
Trains on train dates only; the encoder is loaded by run_graphnet_pilot (--init-encoder).
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.features.build_gridded_sequences import N_CHANNELS
from src.features.build_tabular_dataset import FEATURE_NAMES, THRESHOLDS_MM
from src.features.rainfall_labels import _obs_for_init, regrid_obs_to_grid
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.model import GridGraphNet

ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
OBS_ROOT = ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
CACHE = Path(os.environ.get("SIH_CACHE_DIR") or ROOT / "data" / "interim" / "_graphnet_cache")

EXCEEDANCE_CHANNELS = ("exc_heavy", "exc_very_heavy", "exc_extreme")


def parse_years(spec: str) -> range:
    start, end = (spec.split("-") + [spec])[:2]
    return range(int(start), int(end) + 1)


def sorted_files(years):
    files = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    return sorted(files)


def _load_obs(year, cache):
    if year not in cache:
        cache[year] = regrid_obs_to_grid(
            xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))
    return cache[year]


def build_error_target(files):
    """log1p(observed) - log1p(forecast rain), forecast recovered by un-standardizing X ch0."""
    path = CACHE / "R.dat"
    n = len(files)
    if path.exists() and path.stat().st_size == n * 9 * 66 * 70 * 4:
        print("reusing error-target cache", flush=True)
        return np.memmap(path, np.float32, "r", shape=(n, 9, 66, 70, 1))

    z = np.load(CACHE / "standardizer.npz")
    mean0, std0 = float(z["mean"][0]), float(z["std"][0])
    grids = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n, 9, 66, 70, N_CHANNELS))
    target = np.memmap(path, np.float32, "w+", shape=(n, 9, 66, 70, 1))
    obs_cache = {}
    print(f"building error target ({n} dates) ...", flush=True)
    for i, f in enumerate(files):
        init = Path(f).stem
        obs = _obs_for_init(_load_obs(int(init[:4]), obs_cache), init, np.arange(1, 10)).values
        forecast = grids[i, ..., 0] * std0 + mean0
        target[i, ..., 0] = np.log1p(np.clip(obs, 0, None)) - np.log1p(np.clip(forecast, 0, None))
        if (i + 1) % 200 == 0:
            target.flush()
            print(f"error target {i + 1}/{n}", flush=True)
    target.flush()
    return target


def build_exceedance_target(files):
    """1[obs >= t] - P(ensemble >= t) at the bust thresholds; a dense, tail-focused pretext.

    P(ensemble >= t) is reused from the tabular cache exceedance channels, so no forecast
    netCDF is re-read. Ocean cells (no observation) are NaN so training masks them out.
    """
    path = CACHE / "E.dat"
    n, k = len(files), len(THRESHOLDS_MM)
    if path.exists() and path.stat().st_size == n * 9 * 66 * 70 * k * 4:
        print("reusing exceedance-target cache", flush=True)
        return np.memmap(path, np.float32, "r", shape=(n, 9, 66, 70, k))

    table = np.memmap(CACHE / "T.dat", np.float32, "r", shape=(n, 9, 66, 70, len(FEATURE_NAMES)))
    exc_idx = [FEATURE_NAMES.index(c) for c in EXCEEDANCE_CHANNELS]
    target = np.memmap(path, np.float32, "w+", shape=(n, 9, 66, 70, k))
    obs_cache = {}
    print(f"building exceedance target ({n} dates, {k} thresholds) ...", flush=True)
    for i, f in enumerate(files):
        init = Path(f).stem
        obs = _obs_for_init(_load_obs(int(init[:4]), obs_cache), init, np.arange(1, 10)).values
        land = np.isfinite(obs)
        for j, t in enumerate(THRESHOLDS_MM):
            hit = np.where(land, (obs >= t).astype(np.float32), np.nan)
            target[i, ..., j] = hit - table[i, ..., exc_idx[j]]
        if (i + 1) % 200 == 0:
            target.flush()
            print(f"exceedance target {i + 1}/{n}", flush=True)
    target.flush()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretext", choices=["error", "exceedance"], default="error")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tail-weight", type=float, default=0.0,
                        help="Upweight cells by 1 + w*|target| so rare tail-error cells dominate.")
    parser.add_argument("--years", type=str, default="2000-2004", help="Year range, e.g. 2000-2009.")
    parser.add_argument("--out", type=str, default="outputs/encoder_pretrained.pt")
    args = parser.parse_args()

    files = sorted_files(parse_years(args.years))
    n = len(files)
    n_train = max(1, int(0.8 * n))

    if args.pretext == "exceedance":
        target = build_exceedance_target(files)
        out_channels = len(THRESHOLDS_MM)
        mean = np.zeros(out_channels, np.float32)   # target is already bounded in [-1, 1]
        std = np.ones(out_channels, np.float32)
        print(f"train dates {n_train} | pretext exceedance ({out_channels} thresholds)", flush=True)
    else:
        target = build_error_target(files)
        out_channels = 1
        train_err = np.asarray(target[:n_train, ..., 0])
        finite = np.isfinite(train_err)
        mean = np.array([train_err[finite].mean()], np.float32)
        std = np.array([train_err[finite].std()], np.float32)
        print(f"train dates {n_train} | error mean {mean[0]:.3f} std {std[0]:.3f}", flush=True)

    grids = np.memmap(CACHE / "X.dat", np.float32, "r", shape=(n, 9, 66, 70, N_CHANNELS))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = GridGraphNet(build_grid_graph(), in_channels=N_CHANNELS, hidden=args.hidden).to(device)
    reg_head = torch.nn.Linear(args.hidden, out_channels).to(device)
    enc_params = [p for name, p in net.named_parameters() if not name.startswith("head")]
    optimizer = torch.optim.AdamW(enc_params + list(reg_head.parameters()), lr=args.lr, weight_decay=1e-5)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    idx_all = np.arange(n_train)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        net.train()
        order = idx_all[np.random.permutation(n_train)]
        total, seen = 0.0, 0.0
        for start in range(0, n_train, args.batch):
            idx = np.sort(order[start:start + args.batch])
            x = torch.from_numpy(np.asarray(grids[idx])).to(device)
            r = np.asarray(target[idx])
            weight = np.isfinite(r).astype(np.float32) * (1.0 + args.tail_weight * np.abs(np.nan_to_num(r)))
            y = torch.from_numpy(np.nan_to_num((r - mean) / std)).to(device)
            w = torch.from_numpy(weight).to(device)
            optimizer.zero_grad()
            pred = reg_head(net.features(x))
            loss = ((pred - y) ** 2 * w).sum() / w.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        torch.save(net.state_dict(), out)
        print(f"pretrain epoch {epoch:2d} | {time.time() - started:5.1f}s | MSE {total / seen:.4f} "
              f"| saved -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
