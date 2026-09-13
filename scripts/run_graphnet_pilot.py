"""Train the graph model on reforecast dates for one or more years; save the best model."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from sklearn.metrics import average_precision_score

from src.features.build_gridded_sequences import ChannelStandardizer, build_gridded_sequences
from src.features.rainfall_labels import build_bust_labels, regrid_obs_to_grid
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.mesh import build_multi_mesh
from src.models.graphnet.model import GridGraphNet
from src.models.graphnet.multimesh_model import MultiMeshGraphNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_ROOT = PROJECT_ROOT / "data" / "interim" / "forecasts" / "gefs" / "reforecast_2000_2019"
OBS_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "imd" / "rainfall"


def _load_obs(year: int) -> xr.DataArray:
    return regrid_obs_to_grid(xr.open_dataset(OBS_ROOT / str(year) / f"RF25_ind{year}_rfp25.nc"))


def _parse_years(args) -> list[int]:
    if args.years:
        if "-" in args.years:
            start, end = args.years.split("-")
            return list(range(int(start), int(end) + 1))
        return [int(y) for y in args.years.split(",")]
    return [args.year]


def assemble(years: list[int], limit: int | None):
    files: list[str] = []
    for year in years:
        files += glob.glob(str(FORECAST_ROOT / f"{year}*.nc"))
    files = sorted(files)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No forecast files for {years} yet in {FORECAST_ROOT}")
    obs_cache: dict[int, xr.DataArray] = {}
    sequences, labels = [], []
    for path in files:
        ds = xr.open_dataset(path)
        year = int(str(ds["run"].values[0])[:4])
        if year not in obs_cache:
            obs_cache[year] = _load_obs(year)
        sequences.append(build_gridded_sequences(ds, sequence_length=int(ds.sizes["lead"])))
        labels.append(build_bust_labels(ds, obs_cache[year]))
    features = xr.concat(sequences, dim="run")
    label_array = np.stack(labels)[..., None].astype(np.float32)
    return features, label_array, list(features["run"].values)


def masked_weighted_bce(pred, target, mask, pos_weight):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    return (bce * weight * mask).sum() / mask.sum().clamp(min=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--years", type=str, default=None, help="Range 'YYYY-YYYY' or list 'Y,Y'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["flat", "multimesh"], default="flat")
    parser.add_argument("--mesh-levels", type=int, default=4)
    parser.add_argument("--processor-layers", type=int, default=6)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    years = _parse_years(args)
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "outputs" / f"graphnet_pilot_{years[0]}_{years[-1]}.pt")
    features, labels, run_ids = assemble(years, args.limit)
    print(f"Assembled {len(run_ids)} init dates from {years}, tensor {tuple(features.shape)}")

    split = max(1, int(0.8 * len(run_ids)))
    train_runs, val_runs = run_ids[:split], run_ids[split:]
    if not val_runs:
        train_runs, val_runs = run_ids[:-1], run_ids[-1:]
    standardizer = ChannelStandardizer.fit(features, train_runs)
    X = torch.from_numpy(standardizer.transform(features))
    Y = torch.from_numpy(labels)
    mask = torch.isfinite(Y).float()
    Y = torch.nan_to_num(Y, nan=0.0)

    n_train = len(train_runs)
    Xtr, Ytr, Mtr = X[:n_train], Y[:n_train], mask[:n_train]
    Xva, Yva, Mva = X[n_train:], Y[n_train:], mask[n_train:]

    pos = (Ytr * Mtr).sum().item()
    neg = (Mtr.sum() - (Ytr * Mtr).sum()).item()
    print(f"train dates {n_train}, val dates {len(val_runs)} | "
          f"train bust cells {int(pos)} / {int(pos + neg)} valid ({100 * pos / max(pos + neg, 1):.2f}%)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "multimesh":
        model = MultiMeshGraphNet(build_multi_mesh(n_levels=args.mesh_levels),
                                  in_channels=X.shape[-1], hidden=args.hidden,
                                  processor_layers=args.processor_layers)
    else:
        model = GridGraphNet(build_grid_graph(), in_channels=X.shape[-1], hidden=args.hidden)
    model = model.to(device)
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"model {args.model} | device {device} | "
          f"params {sum(p.numel() for p in model.parameters())} | pos_weight {pos_weight.item():.1f}")

    best_ap = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_train)
        total = 0.0
        for start in range(0, n_train, args.batch):
            idx = order[start:start + args.batch]
            xb, yb, mb = Xtr[idx].to(device), Ytr[idx].to(device), Mtr[idx].to(device)
            optimizer.zero_grad()
            loss = masked_weighted_bce(model(xb), yb, mb, pos_weight)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        train_loss = total / n_train

        model.eval()
        with torch.no_grad():
            preds = model(Xva.to(device)).cpu()
        valid = Mva.bool().numpy().reshape(-1)
        p = preds.numpy().reshape(-1)[valid]
        t = Yva.numpy().reshape(-1)[valid]
        ap = average_precision_score(t, p) if (t.min() != t.max()) else float("nan")
        print(f"epoch {epoch:2d} | train_loss {train_loss:.4f} | "
              f"val PR-AUC {ap:.4f} (base {100 * t.mean():.2f}%)", flush=True)

        if np.isfinite(ap) and ap > best_ap:
            best_ap = ap
            torch.save({
                "model_state": model.state_dict(), "model": args.model,
                "hidden": args.hidden, "in_channels": int(X.shape[-1]),
                "mesh_levels": args.mesh_levels, "processor_layers": args.processor_layers,
                "standardizer_mean": standardizer.mean, "standardizer_std": standardizer.std,
                "years": years, "val_pr_auc": best_ap, "train_runs": train_runs,
            }, out_path)

    print(f"\nBest val PR-AUC {best_ap:.4f}. Saved best model -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
