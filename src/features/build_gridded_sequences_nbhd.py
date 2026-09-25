"""19-channel set + 5 neighbourhood-ensemble channels (issuance-time, no observations).

Motivation: a rainfall *miss* is a cell where no member forecast heavy rain locally,
often because heavy rain was forecast nearby but displaced, or just below threshold.
These channels expose that, using the neighbourhood approach from the Fractions Skill
Score family (Roberts & Lean 2008) applied *within the ensemble* rather than against
observations (which would leak the label):

- nep645_r1 / nep645_r2: neighbourhood maximum ensemble probability (Schwartz & Sobash
  2017) - fraction of members with >= 64.5 mm anywhere within +-1 / +-2 degrees.
- p20 / p35: local ensemble probability of >= 20 / 35 mm (sub-heavy signal).
- dfss35_r1: local member-vs-member FSS at 35 mm over a +-1 degree window (spatial
  agreement between members; Dey, Roberts et al. 2014). 1 = members agree on where rain is.

The base 19-channel builder is reused unchanged (it feeds the 20yr production model).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter, uniform_filter

from src.features.build_gridded_sequences import CANONICAL_DIMS, CHANNEL_NAMES, build_gridded_sequences

NBHD_NAMES: tuple[str, ...] = ("nep645_r1", "nep645_r2", "p20", "p35", "dfss35_r1")
NBHD_CHANNEL_NAMES: tuple[str, ...] = tuple(CHANNEL_NAMES) + NBHD_NAMES
NBHD_N_CHANNELS: int = len(NBHD_CHANNEL_NAMES)
GRID_STEP = 0.5


def _window(half_width_deg: float) -> int:
    return 2 * int(round(half_width_deg / GRID_STEP)) + 1  # +-1 deg -> 5 cells, +-2 deg -> 9


def neighbourhood_channels(member_rain: np.ndarray) -> np.ndarray:
    """member_rain [run, lead, member, lat, lon] -> [run, lead, lat, lon, 5]."""
    rain = np.nan_to_num(np.asarray(member_rain, dtype=np.float64), nan=0.0)
    spatial = lambda w: (1, 1, 1, w, w)
    w1, w2 = _window(1.0), _window(2.0)

    nep1 = (maximum_filter(rain, size=spatial(w1), mode="nearest") >= 64.5).mean(2)
    nep2 = (maximum_filter(rain, size=spatial(w2), mode="nearest") >= 64.5).mean(2)
    p20 = (rain >= 20.0).mean(2)
    p35 = (rain >= 35.0).mean(2)

    frac = uniform_filter((rain >= 35.0).astype(np.float64), size=spatial(w1), mode="constant")
    num = np.zeros(frac[:, :, 0].shape); den = np.zeros_like(num)
    for i, j in combinations(range(frac.shape[2]), 2):
        fi, fj = frac[:, :, i], frac[:, :, j]
        num += uniform_filter((fi - fj) ** 2, size=(1, 1, w1, w1), mode="constant")
        den += uniform_filter(fi ** 2 + fj ** 2, size=(1, 1, w1, w1), mode="constant")
    dfss = np.where(den > 1e-12, 1.0 - num / np.maximum(den, 1e-12), 1.0)
    return np.stack([nep1, nep2, p20, p35, dfss], axis=-1).astype(np.float32)


def build_nbhd_sequences(dataset: xr.Dataset, *, member_dim: str = "member",
                         sequence_length: int | None = None) -> xr.DataArray:
    base = build_gridded_sequences(dataset, member_dim=member_dim, sequence_length=sequence_length)
    rain = dataset["total_precipitation"].transpose("run", "lead", member_dim, "latitude", "longitude")
    extra = neighbourhood_channels(rain.values)
    extra_da = xr.DataArray(extra, dims=(*CANONICAL_DIMS, "channel"),
                            coords={**{d: base.coords[d] for d in CANONICAL_DIMS}, "channel": list(NBHD_NAMES)})
    stacked = xr.concat([base, extra_da], dim="channel").transpose(*CANONICAL_DIMS, "channel")
    stacked.attrs["channels"] = list(NBHD_CHANNEL_NAMES)
    return stacked


def standardize_nbhd(sequences: xr.DataArray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = sequences.transpose(*CANONICAL_DIMS, "channel").values.astype(np.float64)
    if array.shape[-1] != mean.size:
        raise ValueError("Channel count does not match the fitted standardizer")
    return np.nan_to_num((array - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
