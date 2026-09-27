"""Lagged-ensemble LPS channels: the GEFS runs issued 1 and 2 days earlier, at the same valid time.

For init t and lead L (valid t + L days, 00 UTC), the run from t-1 verifies at the same
moment at lead L+1, and the run from t-2 at lead L+2. Their tracker channels are already in
X_lps.dat (row of that init), so this is a re-indexing, not a re-run: 5 members become up
to 15, and a storm that persists across runs is more believable than one that flickers.
The store keeps 9 leads, so lag 1 is unavailable at lead 9 and lag 2 at leads 8-9 (and on
the first days / after gaps in the archive); the availability flags tell the model.

Writes X_lag.dat [n, lead, lat, lon, 11] next to X_lps.dat and adds "lag_channels" to meta.json:
  per lag k in (1, 2): p_strike300, p_strike600, zeta850_mean, msl_dip_mean  (standardized as X_lps)
                       avail  (1 if that run exists at that lead)
  p_strike300_lagged   pooled strike probability over all available runs' members (raw 0-1)

    python -m scripts.build_lps_lag --cache-dir D:\\sih-data\\caches\\lps_20yr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAG_SRC = ["p_strike300", "p_strike600", "zeta850_mean", "msl_dip_mean"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    n, L, H, W, names = meta["n"], meta["leads"], meta["lat"], meta["lon"], meta["channels"]
    Xl = np.memmap(cache / "X_lps.dat", np.float32, "r", shape=(n, L, H, W, len(names)))
    P = np.memmap(cache / "P.dat", np.float32, "r", shape=(n, L, H, W))
    src = [names.index(c) for c in LAG_SRC]
    lag_names = [f"{c}_lag{k}" for k in (1, 2) for c in LAG_SRC + ["avail"]] + ["p_strike300_lagged"]
    row_of = {d: k for k, d in enumerate(meta["inits"])}
    dates = pd.to_datetime(meta["inits"], format="%Y%m%d")
    prev = {k: [row_of.get((dates[i] - pd.Timedelta(days=k)).strftime("%Y%m%d")) for i in range(n)] for k in (1, 2)}

    out = np.memmap(cache / "X_lag.dat", np.float32, "w+", shape=(n, L, H, W, len(lag_names)))
    avail_count = np.zeros((2, L), np.int64)
    for i in range(n):
        x = np.zeros((L, H, W, len(lag_names)), np.float32)
        pooled, members = P[i].astype(np.float32) * 5, np.full(L, 5.0, np.float32)
        for j, k in enumerate((1, 2)):
            r = prev[k][i]
            if r is None:
                continue
            base = j * (len(LAG_SRC) + 1)
            for lead in range(L - k):  # run t-k at lead+k verifies with run t at lead
                x[lead, ..., base:base + len(LAG_SRC)] = Xl[r, lead + k][..., src]
                x[lead, ..., base + len(LAG_SRC)] = 1.0
                pooled[lead] += P[r, lead + k] * 5
                members[lead] += 5
                avail_count[j, lead] += 1
        x[..., -1] = pooled / members[:, None, None]
        out[i] = x
        if i % 1000 == 0:
            print(f"  {i}/{n}", flush=True)
    out.flush()
    meta["lag_channels"] = lag_names
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    share = (avail_count / n).round(3).tolist()
    print(f"X_lag.dat [{n}, {L}, {H}, {W}, {len(lag_names)}] | share of inits with lag1/lag2 by lead: {share}", flush=True)


if __name__ == "__main__":
    main()
