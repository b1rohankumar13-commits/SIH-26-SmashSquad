"""Monsoon low-pressure-system (LPS) tracker and strike-probability labels.

The same detector runs on ERA5 (truth) and on every GEFS member (forecast), so a
bust reflects GEFS error rather than a mismatch between track definitions.

Detection (one 00 UTC snapshot at a time), following the vorticity-based approach of
published LPS catalogues (e.g. Hunt et al. 2016), simplified to daily snapshots:
  1. 850 hPa relative vorticity from u/v on the sphere, Gaussian-smoothed to the
     synoptic scale (sigma ~1.5 deg) so a system gives one broad peak.
  2. Candidates = local maxima of smoothed vorticity >= ZETA_MIN within a ~3 deg
     neighbourhood, inside the LPS belt (lat 5-30N).
  3. Each candidate needs a surface pressure dip: the lowest MSLP within 2.5 deg must
     sit >= DEFICIT_MIN below the mean of a 5-8 deg annulus.
  4. Centres are only accepted inside centre_mask(): the LPS belt, >= EDGE_DEG from the
     grid edge (smoothing is unreliable there), below MAX_OROG_M (850 hPa is underground
     or orographically distorted over the Himalaya/Tibet), and outside the
     Thar/Baluchistan heat low (lat > 22N, lon < 72E), a shallow quasi-stationary
     thermal low rather than a travelling LPS.
Truth and forecast are run on the identical 0.5 deg model grid (ERA5 is regridded to it)
with the identical mask, so the two sides differ only in the weather.
Tracking links daily detections within MAX_STEP_DEG; tracks shorter than min_days
can be dropped. Strike: a cell is struck when a system centre lies within
STRIKE_KM at that valid time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

R_EARTH = 6.371e6
ZETA_MIN = 1.0e-5          # s^-1, smoothed 850 hPa vorticity
DEFICIT_MIN = 1.0          # hPa, local MSLP minimum below the 5-8 deg annulus mean
SMOOTH_SIGMA_DEG = 1.5
PEAK_RADIUS_DEG = 3.0
LAT_BAND = (5.0, 30.0)
MAX_STEP_DEG = 6.0         # max centre displacement per day when linking
STRIKE_KM = 300.0
EDGE_DEG = 2.0
MAX_OROG_M = 1000.0
MISS_BELOW, FALSE_ALARM_AT = 0.20, 0.50


@dataclass
class Detection:
    t: int              # time index
    lat: float
    lon: float
    zeta: float         # smoothed vorticity at the centre, s^-1
    deficit: float      # MSLP dip, hPa (positive = deeper)


def relative_vorticity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """zeta = dv/dx - du/dy + u tan(phi)/R on a regular lat/lon grid (lat ascending)."""
    phi = np.deg2rad(lats)[:, None]
    dlam = np.deg2rad(np.gradient(lons))[None, :]
    dphi = np.deg2rad(np.gradient(lats))[:, None]
    dvdx = np.gradient(v, axis=-1) / (R_EARTH * np.cos(phi) * dlam)
    dudy = np.gradient(u, axis=-2) / (R_EARTH * dphi)
    return dvdx - dudy + u * np.tan(phi) / R_EARTH


def _disk(radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_cells ** 2


def _annulus(inner: float, outer: float) -> np.ndarray:
    r = int(np.ceil(outer))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = x * x + y * y
    return (d2 >= inner ** 2) & (d2 <= outer ** 2)


def centre_mask(lats: np.ndarray, lons: np.ndarray, orog_m: np.ndarray | None = None) -> np.ndarray:
    """Where an LPS centre may be accepted ([lat, lon] bool)."""
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    ok = (lat2d >= LAT_BAND[0]) & (lat2d <= LAT_BAND[1])
    ok &= (lat2d >= lats.min() + EDGE_DEG) & (lat2d <= lats.max() - EDGE_DEG)
    ok &= (lon2d >= lons.min() + EDGE_DEG) & (lon2d <= lons.max() - EDGE_DEG)
    ok &= ~((lat2d > 22.0) & (lon2d < 72.0))
    if orog_m is not None:
        ok &= orog_m < MAX_OROG_M
    return ok


def detect(u: np.ndarray, v: np.ndarray, msl_hpa: np.ndarray, lats: np.ndarray, lons: np.ndarray,
           t: int = 0, *, mask: np.ndarray | None = None, zeta_min: float = ZETA_MIN,
           deficit_min: float = DEFICIT_MIN) -> list[Detection]:
    """Detect LPS centres in one snapshot. Arrays [lat, lon], lat ascending, regular grid."""
    step = float(abs(lats[1] - lats[0]))
    zeta = ndimage.gaussian_filter(relative_vorticity(u, v, lats, lons), SMOOTH_SIGMA_DEG / step, mode="nearest")
    peaks = (zeta == ndimage.maximum_filter(zeta, footprint=_disk(PEAK_RADIUS_DEG / step), mode="nearest"))
    peaks &= zeta >= zeta_min
    peaks &= centre_mask(lats, lons) if mask is None else mask
    if not peaks.any():
        return []
    local_min = ndimage.minimum_filter(msl_hpa, footprint=_disk(2.5 / step), mode="nearest")
    ring = _annulus(5.0 / step, 8.0 / step)
    ring_mean = ndimage.convolve(msl_hpa, ring / ring.sum(), mode="nearest")
    deficit = ring_mean - local_min
    out = []
    for i, j in zip(*np.nonzero(peaks)):
        if deficit[i, j] >= deficit_min:
            out.append(Detection(t, float(lats[i]), float(lons[j]), float(zeta[i, j]), float(deficit[i, j])))
    return out


def _gc_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return float(np.rad2deg(np.arccos(np.clip(c, -1, 1))))


def link_tracks(dets: list[Detection], *, max_step_deg: float = MAX_STEP_DEG) -> list[list[Detection]]:
    """Greedy nearest-neighbour linking of consecutive-day detections into tracks."""
    by_t: dict[int, list[Detection]] = {}
    for d in dets:
        by_t.setdefault(d.t, []).append(d)
    tracks: list[list[Detection]] = []
    open_tracks: list[list[Detection]] = []
    for t in sorted(by_t):
        cands = list(by_t[t])
        pairs = sorted(((_gc_deg(tr[-1].lat, tr[-1].lon, c.lat, c.lon), k, m)
                        for k, tr in enumerate(open_tracks) if tr[-1].t == t - 1
                        for m, c in enumerate(cands)), key=lambda p: p[0])
        used_tr, used_c = set(), set()
        for dist, k, m in pairs:
            if dist > max_step_deg or k in used_tr or m in used_c:
                continue
            open_tracks[k].append(cands[m]); used_tr.add(k); used_c.add(m)
        for m, c in enumerate(cands):
            if m not in used_c:
                new = [c]; open_tracks.append(new); tracks.append(new)
        open_tracks = [tr for tr in open_tracks if tr[-1].t == t]
    return tracks


def strike_mask(centres: list[tuple[float, float]], lats: np.ndarray, lons: np.ndarray,
                radius_km: float = STRIKE_KM) -> np.ndarray:
    """Boolean [lat, lon]: cell within radius_km of any centre."""
    out = np.zeros((lats.size, lons.size), bool)
    if not centres:
        return out
    phi = np.deg2rad(lats)[:, None]; lam = np.deg2rad(lons)[None, :]
    for clat, clon in centres:
        p0, l0 = np.deg2rad(clat), np.deg2rad(clon)
        c = np.sin(phi) * np.sin(p0) + np.cos(phi) * np.cos(p0) * np.cos(lam - l0)
        out |= R_EARTH / 1e3 * np.arccos(np.clip(c, -1, 1)) <= radius_km
    return out


def strike_bust_labels(ens_prob: np.ndarray, observed: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Directional labels [..., 2] (miss, false_alarm) from ensemble strike probability."""
    miss = (ens_prob < MISS_BELOW) & observed
    fa = (ens_prob >= FALSE_ALARM_AT) & ~observed
    out = np.stack([miss, fa], -1).astype(np.float32)
    if valid is not None:
        out[~valid] = np.nan
    return out
