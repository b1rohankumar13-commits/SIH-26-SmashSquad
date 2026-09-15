"""Train the graph model on reforecast dates; stream features to a disk cache and save the best model."""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.metrics import average_precision_score

from src.features.build_gridded_sequences import ChannelStandardizer, N_CHANNELS, build_gridded_sequences
from src.features.rainfall_labels import build_bust_labels, regrid_obs_to_grid
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.mesh import build_multi_mesh
from src.models.graphnet.model import GridGraphNet
from src.models.graphnet.multimesh_model import MultiMeshGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
OBS_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"
CACHE_DIR = Path(os.environ.get("SIH_CACHE_DIR") or PROJECT_ROOT / "data" / "interim" / "_graphnet_cache")


def _load_obs(year: int) -> xr.DataArray:
    return regrid_obs_to_grid(xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))


def _parse_years(args) -> list[int]:
    if args.years:
        if "-" in args.years:
            start, end = args.years.split("-")
            return list(range(int(start), int(end) + 1))
        return [int(y) for y in args.years.split(",")]
    return [args.year]


def _run_id(path: str) -> str:
    return Path(path).stem


def _fit_standardizer(train_files: list[str], sample_size: int = 300) -> ChannelStandardizer:
    stride = max(1, len(train_files) // sample_size)
    seqs = []
    for path in train_files[::stride][:sample_size]:
        with xr.open_dataset(path) as ds:
            seqs.append(build_gridded_sequences(ds, sequence_length=int(ds.sizes["lead"])).load())
    sample = xr.concat(seqs, dim="run")
    return ChannelStandardizer.fit(sample, list(sample["run"].values))


def build_cache(files: list[str], n_train: int, *, sample_size: int = 300):
    """Fit the standardizer on a training sample, then stream all dates to memmaps on disk."""
    standardizer = _fit_standardizer(files[:n_train], sample_size)
    print(f"standardizer fitted on {min(sample_size, n_train)} dates; streaming cache ...", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n, leads = len(files), 9
    features = np.memmap(CACHE_DIR / "X.dat", np.float32, "w+", shape=(n, leads, 66, 70, N_CHANNELS))
    labels = np.memmap(CACHE_DIR / "Y.dat", np.float32, "w+", shape=(n, leads, 66, 70, 1))
    obs_cache: dict[int, xr.DataArray] = {}
    for i, path in enumerate(files):
        year = int(_run_id(path)[:4])
        if year not in obs_cache:
            obs_cache[year] = _load_obs(year)
        with xr.open_dataset(path) as ds:
            seq = build_gridded_sequences(ds, sequence_length=int(ds.sizes["lead"]))
            features[i] = standardizer.transform(seq)[0]
            labels[i] = build_bust_labels(ds, obs_cache[year])[..., None]
        if (i + 1) % 200 == 0:
            features.flush()
            print(f"cached {i + 1}/{n}", flush=True)
    features.flush()
    labels.flush()
    np.savez(CACHE_DIR / "standardizer.npz", mean=standardizer.mean, std=standardizer.std)
    return features, labels, standardizer


def load_or_build_cache(files: list[str], n_train: int, reuse: bool):
    x_path, y_path = CACHE_DIR / "X.dat", CACHE_DIR / "Y.dat"
    expected = len(files) * 9 * 66 * 70 * N_CHANNELS * 4
    if reuse and x_path.exists() and y_path.exists() and x_path.stat().st_size == expected:
        features = np.memmap(x_path, np.float32, "r", shape=(len(files), 9, 66, 70, N_CHANNELS))
        labels = np.memmap(y_path, np.float32, "r", shape=(len(files), 9, 66, 70, 1))
        npz = CACHE_DIR / "standardizer.npz"
        if npz.exists():
            data = np.load(npz)
            standardizer = ChannelStandardizer(mean=data["mean"], std=data["std"])
        else:
            standardizer = _fit_standardizer(files[:n_train])
            np.savez(npz, mean=standardizer.mean, std=standardizer.std)
        print(f"reusing cache ({len(files)} dates)", flush=True)
        return features, labels, standardizer
    return build_cache(files, n_train)


def masked_weighted_bce(pred, target, mask, pos_weight):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return bce * weight * mask


def _to_batch(features, labels, idx, device):
    x = torch.from_numpy(np.asarray(features[idx])).to(device)
    y_raw = torch.from_numpy(np.asarray(labels[idx]))
    mask = torch.isfinite(y_raw).float().to(device)
    y = torch.nan_to_num(y_raw, nan=0.0).to(device)
    return x, y, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--years", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8, help="Early-stop after N epochs without PR-AUC gain.")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse the on-disk feature cache if present.")
    parser.add_argument("--model", choices=["flat", "multimesh"], default="flat")
    parser.add_argument("--mesh-levels", type=int, default=4)
    parser.add_argument("--processor-layers", type=int, default=6)
    parser.add_argument("--init-encoder", type=str, default=None, help="Pretrained encoder state_dict to load.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the head (freeze encoder).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    years = _parse_years(args)
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "outputs" / f"graphnet_pilot_{years[0]}_{years[-1]}.pt")

    files: list[str] = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    files = sorted(files)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No forecast files for {years} in {FORECAST_ROOT}")

    run_ids = [_run_id(f) for f in files]
    n_train = max(1, int(0.8 * len(files)))
    if n_train >= len(files):
        n_train = len(files) - 1
    print(f"{len(files)} init dates from {years} | train {n_train} val {len(files) - n_train}", flush=True)

    features, labels, standardizer = load_or_build_cache(files, n_train, args.reuse_cache)

    train_labels = np.asarray(labels[:n_train])
    pos = float((train_labels == 1).sum())
    neg = float((train_labels == 0).sum())
    print(f"train bust cells {int(pos)} / {int(pos + neg)} valid ({100 * pos / max(pos + neg, 1):.2f}%)",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "multimesh":
        model = MultiMeshGraphNet(build_multi_mesh(n_levels=args.mesh_levels),
                                  in_channels=N_CHANNELS, hidden=args.hidden,
                                  processor_layers=args.processor_layers)
    else:
        model = GridGraphNet(build_grid_graph(), in_channels=N_CHANNELS, hidden=args.hidden)
    model = model.to(device)
    if args.init_encoder:
        state = torch.load(PROJECT_ROOT / args.init_encoder, map_location=device)
        model.load_state_dict(state.get("model_state", state), strict=False)
        print(f"loaded pretrained encoder from {args.init_encoder}", flush=True)
    if args.freeze_encoder:
        for name, p in model.named_parameters():
            if not name.startswith("head"):
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"model {args.model} | device {device} | params {sum(p.numel() for p in model.parameters())} "
          f"| pos_weight {pos_weight.item():.1f} | weight_decay {args.weight_decay} | patience {args.patience}",
          flush=True)

    train_idx, val_idx = np.arange(n_train), np.arange(n_train, len(files))
    best_ap, best_epoch, since_improve = -1.0, 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = train_idx[np.random.permutation(n_train)]
        running, seen = 0.0, 0.0
        for start in range(0, n_train, args.batch):
            idx = np.sort(order[start:start + args.batch])
            x, y, mask = _to_batch(features, labels, idx, device)
            optimizer.zero_grad()
            elementwise = masked_weighted_bce(model(x), y, mask, pos_weight)
            loss = elementwise.sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()
            running += elementwise.sum().item()
            seen += mask.sum().item()
        train_loss = running / max(seen, 1.0)

        model.eval()
        preds, targets, val_num, val_den = [], [], 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch):
                idx = val_idx[start:start + args.batch]
                x, y, mask = _to_batch(features, labels, idx, device)
                out = model(x)
                elementwise = masked_weighted_bce(out, y, mask, pos_weight)
                val_num += elementwise.sum().item()
                val_den += mask.sum().item()
                keep = mask.bool().cpu().numpy().reshape(-1)
                preds.append(out.cpu().numpy().reshape(-1)[keep])
                targets.append(y.cpu().numpy().reshape(-1)[keep])
        val_loss = val_num / max(val_den, 1.0)
        p, t = np.concatenate(preds), np.concatenate(targets)
        ap = average_precision_score(t, p) if t.min() != t.max() else float("nan")
        print(f"epoch {epoch:2d} | {time.time() - started:5.1f}s | train_loss {train_loss:.4f} "
              f"| val_loss {val_loss:.4f} | val PR-AUC {ap:.4f} (base {100 * t.mean():.2f}%)", flush=True)

        if np.isfinite(ap) and ap > best_ap:
            best_ap, best_epoch, since_improve = ap, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "model": args.model,
                "hidden": args.hidden, "in_channels": N_CHANNELS,
                "mesh_levels": args.mesh_levels, "processor_layers": args.processor_layers,
                "standardizer_mean": standardizer.mean, "standardizer_std": standardizer.std,
                "years": years, "val_pr_auc": best_ap, "epoch": epoch,
                "run_ids": run_ids, "n_train": n_train,
            }, out_path)
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"early stop: no PR-AUC gain in {args.patience} epochs", flush=True)
                break

    print(f"\nBest val PR-AUC {best_ap:.4f} at epoch {best_epoch}. Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
