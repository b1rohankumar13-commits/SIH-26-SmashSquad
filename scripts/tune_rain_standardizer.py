"""Tune the rebuilt rain standardizer so the rain model reproduces outputs saved before the loss.

scripts/rebuild_live_stats.py re-fits the 24-channel rain standardizer on recent operational runs;
that flags ~1.7x the original cells. Live runs saved earlier (outputs/live/gefs/<init>/rain/p_*.npy)
were made with the original standardizer, so with their forecasts re-downloaded the original
mean/std can be recovered: optimise the 48 numbers so the frozen model's logits match the saved
ones (model weights untouched). Dates are split into fit / held-out to check it generalises.

    python -m scripts.tune_rain_standardizer
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from scripts import export_explanations as ex
from scripts.live_forecast import LIVE, rain_model, store_file
from src.features.build_gridded_sequences_nbhd import build_nbhd_sequences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STD = Path(r"D:\sih-data\caches\rain_20yr_24ch_nbhd\standardizer.npz")
HOLDOUT = ("20220706", "20260930")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(init: str):
    with xr.open_dataset(store_file("gefs", "rain", init)) as ds:
        raw = build_nbhd_sequences(ds, sequence_length=9).transpose("run", "lead", "latitude", "longitude",
                                                                   "channel").values[0].astype(np.float32)
    d = LIVE / "gefs" / init / "rain"
    target = np.stack([np.load(d / "p_miss.npy"), np.load(d / "p_fa.npy")], -1).astype(np.float32)  # saved as float16
    return raw, target, np.load(d / "domain.npy")


def flags(p: np.ndarray, dom: np.ndarray) -> np.ndarray:
    idx = json.loads((PROJECT_ROOT / "outputs" / "dashboard" / "index.json").read_text())["rain"]["tiers"]
    out = []
    for j, d in enumerate(("miss", "false_alarm")):
        t = idx[d]
        out.append(np.stack([(ex.calibrated(p[L, ..., j], t["calibration"], L + 1) >= ex.display_cut(t)) & dom
                             for L in range(p.shape[0])]))
    return np.stack(out, -1)


def main() -> None:
    inits = sorted(p.parent.parent.name for p in (LIVE / "gefs").glob("*/rain/p_miss.npy")
                   if p.parent.parent.name.startswith(("2022", "2026093")) and store_file("gefs", "rain", p.parent.parent.name).exists())
    data = {i: load(i) for i in inits}
    fit = [i for i in inits if i not in HOLDOUT]
    print(f"fit on {fit}, held out {[i for i in inits if i in HOLDOUT]}", flush=True)

    m = rain_model()
    for q in m.parameters():
        q.requires_grad_(False)
    s = np.load(STD)
    mean = torch.tensor(s["mean"], dtype=torch.float32, device=DEV, requires_grad=True)
    logstd = torch.tensor(np.log(s["std"]), dtype=torch.float32, device=DEV, requires_grad=True)
    # NaN inputs become 0 after standardising (as in standardize_nbhd); keep them out of the graph
    # so they cannot poison gradients, and zero NaN targets outside the domain (weight 0 there).
    tens = {}
    for i, (r, t, dom) in data.items():
        ok = np.isfinite(r)
        tp = np.nan_to_num(t, nan=0.0)
        tens[i] = (torch.from_numpy(np.where(ok, r, 0)).to(DEV), torch.from_numpy(logit(tp)).to(DEV),
                   torch.from_numpy(dom & np.isfinite(t).all((0, -1))).to(DEV), torch.from_numpy(tp).to(DEV),
                   torch.from_numpy(ok).to(DEV))

    def predict_logits(i):
        raw, ok = tens[i][0], tens[i][4]
        x = torch.where(ok, (raw - mean) / logstd.exp(), torch.zeros_like(raw))
        return m.head(m.features(x[None]))[0]

    def loss_on(keys):
        tot = 0.0
        for i in keys:
            _, tl, dom, tp, _ = tens[i]
            z = predict_logits(i).clamp(-7, 7)  # compare only where it matters (p ~ 0.001-0.999)
            w = (1.0 + 20.0 * tp) * dom[None, :, :, None]  # weight cells the model considers risky
            tot = tot + ((z - tl.clamp(-7, 7)) ** 2 * w).sum() / w.sum()
        return tot / len(keys)

    def agreement(keys) -> float:
        """Jaccard overlap of flagged cells (miss + FA) between saved and current outputs."""
        both = union = 0
        with torch.no_grad():
            for i in keys:
                _, target, dom = data[i]
                fo, fn = flags(target, dom), flags(torch.sigmoid(predict_logits(i)).cpu().numpy(), dom)
                both += (fo & fn).sum(); union += (fo | fn).sum()
        return both / max(union, 1)

    def report(tag):
        with torch.no_grad():
            for i, (raw, target, dom) in data.items():
                p = torch.sigmoid(predict_logits(i)).cpu().numpy()
                fo, fn = flags(target, dom), flags(p, dom)
                r = np.corrcoef(target[:, dom, 0].ravel(), p[:, dom, 0].ravel())[0, 1]
                print(f"  {tag} {i}{' (held out)' if i in HOLDOUT else ''}: miss corr {r:.4f} | flagged miss "
                      f"orig {fo[..., 0].sum()} new {fn[..., 0].sum()} both {(fo & fn)[..., 0].sum()} | "
                      f"fa orig {fo[..., 1].sum()} new {fn[..., 1].sum()} both {(fo & fn)[..., 1].sum()}", flush=True)

    held = [i for i in inits if i in HOLDOUT]
    report("before")
    start = agreement(held)
    best = (start, mean.detach().clone(), logstd.detach().clone(), -1)
    print(f"flag agreement before: held-out {start:.3f}, fit {agreement(fit):.3f}", flush=True)
    opt = torch.optim.Adam([mean, logstd], lr=0.003)
    for step in range(600):
        opt.zero_grad()
        loss = loss_on(fit)
        loss.backward()
        opt.step()
        if step % 25 == 24:
            a = agreement(held)
            if a > best[0]:
                best = (a, mean.detach().clone(), logstd.detach().clone(), step)
            if step % 100 == 99:
                print(f"step {step} fit loss {loss.item():.4f} | agreement held-out {a:.3f} fit {agreement(fit):.3f}", flush=True)
    with torch.no_grad():
        mean.copy_(best[1]); logstd.copy_(best[2])
    report(f"best (step {best[3]})")
    if best[3] < 0:
        raise SystemExit(f"held-out flag agreement never beat {start:.3f}; standardizer left unchanged")
    print(f"flag agreement: held-out {start:.3f} -> {best[0]:.3f}, fit {agreement(fit):.3f}", flush=True)
    shutil.copy(STD, STD.with_name("standardizer_refit.npz"))
    np.savez(STD, mean=mean.detach().cpu().numpy().astype(np.float64), std=logstd.exp().detach().cpu().numpy().astype(np.float64))
    with open(STD.parent / "REBUILT.txt", "a") as f:
        f.write(f"standardizer.npz then tuned to reproduce saved live outputs on {fit} "
                f"(scripts/tune_rain_standardizer.py); pre-tune fit kept as standardizer_refit.npz\n")
    print(f"saved {STD}")


if __name__ == "__main__":
    main()
