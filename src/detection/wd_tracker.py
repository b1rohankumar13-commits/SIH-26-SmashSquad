"""Western-disturbance (WD) detection and tracking on 500 hPa relative vorticity.

A WD is a mid-latitude trough carried east on the subtropical westerly jet. Centres are
local maxima of smoothed 500 hPa vorticity embedded in westerly flow (500 hPa zonal wind
around the centre > 0), in the 20-42N band. Daily 00 UTC centres are linked into tracks
with the LPS linker, using a larger step because WDs move ~8-12 deg/day.

Used for ERA5 truth (which heavy winter precipitation is WD-driven, observed tracks) and on
GEFS members (does the forecast carry a WD near a cell).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.detection.lps_tracker import Detection, _disk, relative_vorticity

ZETA_MIN = 4.0e-5          # s^-1, smoothed 500 hPa vorticity
SMOOTH_SIGMA_DEG = 2.0
PEAK_RADIUS_DEG = 4.0
LAT_BAND = (20.0, 42.0)
WESTERLY_MIN = 5.0         # m/s, mean 500 hPa u within 3 deg of the centre
MAX_STEP_DEG = 14.0        # max centre displacement per day when linking
# A track counts as a WD for India if it passes through this box.
INDIA_BOX = (25.0, 40.0, 60.0, 80.0)   # lat0, lat1, lon0, lon1
GATE_KM = 1200.0           # heavy precipitation within this distance of a WD centre is WD-driven
GEO_ZETA_MIN = 5.0e-5      # s^-1, same WD frequency when vorticity comes from geostrophic wind


def smoothed_zeta(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    step = float(abs(lats[1] - lats[0]))
    return ndimage.gaussian_filter(relative_vorticity(u, v, lats, lons), SMOOTH_SIGMA_DEG / step, mode="nearest")


def detect(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray, t: int = 0, *,
           zeta_min: float = ZETA_MIN) -> list[Detection]:
    """WD centres in one 500 hPa snapshot. Arrays [lat, lon], lat ascending, regular grid.
    Detection.deficit holds the local westerly wind (m/s) instead of an MSLP dip."""
    step = float(abs(lats[1] - lats[0]))
    zeta = smoothed_zeta(u, v, lats, lons)
    peaks = zeta == ndimage.maximum_filter(zeta, footprint=_disk(PEAK_RADIUS_DEG / step), mode="nearest")
    peaks &= zeta >= zeta_min
    lat2d = np.broadcast_to(lats[:, None], zeta.shape)
    peaks &= (lat2d >= LAT_BAND[0]) & (lat2d <= LAT_BAND[1])
    if not peaks.any():
        return []
    ubar = ndimage.convolve(u, _disk(3.0 / step) / _disk(3.0 / step).sum(), mode="nearest")
    return [Detection(t, float(lats[i]), float(lons[j]), float(zeta[i, j]), float(ubar[i, j]))
            for i, j in zip(*np.nonzero(peaks & (ubar >= WESTERLY_MIN)))]


def affects_india(track: list[Detection]) -> bool:
    a0, a1, o0, o1 = INDIA_BOX
    return any(a0 <= d.lat <= a1 and o0 <= d.lon <= o1 for d in track)


def geostrophic_wind(gh_m: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u_g, v_g) from geopotential height in metres, [..., lat, lon] with lat ascending.
    For stores that keep 500 hPa heights but not winds (the GEFS rain store)."""
    from src.detection.lps_tracker import R_EARTH
    phi = np.deg2rad(lats)[:, None]
    f = 2 * 7.2921e-5 * np.sin(phi)
    g_f = 9.80665 / f
    dzdy = np.gradient(gh_m, axis=-2) / (R_EARTH * np.deg2rad(np.gradient(lats))[:, None])
    dzdx = np.gradient(gh_m, axis=-1) / (R_EARTH * np.cos(phi) * np.deg2rad(np.gradient(lons))[None, :])
    return -g_f * dzdy, g_f * dzdx
