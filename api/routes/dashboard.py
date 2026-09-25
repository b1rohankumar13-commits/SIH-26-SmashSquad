"""Dashboard data API over the prediction exports in outputs/dashboard/.

Serves what scripts/export_dashboard_predictions.py wrote - no model is run here.
Grid payloads are columnar (one array per field) over observed cells only, which keeps
a (hazard, init, lead) map to ~1,150 cells / ~60 KB of JSON.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "outputs" / "dashboard"
CASES_FILE = PROJECT_ROOT / "dashboard" / "case_studies.py"

router = APIRouter(prefix="/api", tags=["dashboard"])


@lru_cache(maxsize=1)
def _index() -> dict:
    path = EXPORT_DIR / "index.json"
    if not path.exists():
        raise HTTPException(503, "No exported predictions. Run scripts/export_dashboard_predictions.py.")
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def _arrays(hazard: str) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode="r") for p in (EXPORT_DIR / hazard).glob("*.npy")}


@lru_cache(maxsize=1)
def _grid() -> tuple[np.ndarray, np.ndarray]:
    return np.load(EXPORT_DIR / "lats.npy"), np.load(EXPORT_DIR / "lons.npy")


def _hazard(h: str) -> dict:
    idx = _index()
    if h not in idx:
        raise HTTPException(404, f"Unknown hazard '{h}'. Available: {sorted(idx)}")
    return idx[h]


def _init_pos(h: str, init: str) -> int:
    inits = _hazard(h)["inits"]
    try:
        return inits.index(init)
    except ValueError:
        raise HTTPException(404, f"No forecast issued {init} for {h}.") from None


def _round(a: np.ndarray, nd: int = 4) -> list:
    return np.round(np.asarray(a, np.float64), nd).tolist()


@router.get("/hazards")
def hazards() -> list[dict]:
    """Hazard metadata (without the long init lists)."""
    out = []
    for key, h in _index().items():
        meta = {k: v for k, v in h.items() if k != "inits"}
        out.append({"id": key, "n_inits": len(h["inits"]), "first_init": h["inits"][0],
                    "last_init": h["inits"][-1], **meta})
    order = {"rain": 0, "heatwave": 1, "monsoon": 2}
    return sorted(out, key=lambda m: order.get(m["id"], 9))


@router.get("/hazards/{hazard}/inits")
def inits(hazard: str) -> dict:
    h = _hazard(hazard)
    return {"inits": h["inits"], "notable": h.get("notable_inits", [])}


@router.get("/grid/{hazard}")
def grid(hazard: str, init: str = Query(..., pattern=r"^\d{8}$"), lead: int = Query(..., ge=1, le=9)) -> dict:
    h = _hazard(hazard)
    if h["kind"] != "grid":
        raise HTTPException(400, f"{hazard} is regional; use /api/monsoon.")
    k, L = _init_pos(hazard, init), lead - 1
    a = _arrays(hazard)
    lats, lons = _grid()
    obs_m = np.asarray(a["obs_miss"][k, L])
    keep = obs_m >= 0
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    return {"init": init, "lead": lead,
            "lat": _round(lat2[keep], 2), "lon": _round(lon2[keep], 2),
            "p_miss": _round(np.asarray(a["p_miss"][k, L])[keep]),
            "p_fa": _round(np.asarray(a["p_fa"][k, L])[keep]),
            "obs_miss": obs_m[keep].astype(int).tolist(),
            "obs_fa": np.asarray(a["obs_fa"][k, L])[keep].astype(int).tolist(),
            "ens_prob": _round(np.asarray(a["ens_prob"][k, L])[keep], 3)}


@router.get("/grid/{hazard}/leads")
def grid_leads(hazard: str, init: str = Query(..., pattern=r"^\d{8}$")) -> list[dict]:
    """Per lead: cells at high alert per direction, and cells that actually busted."""
    h = _hazard(hazard)
    k = _init_pos(hazard, init)
    a, t = _arrays(hazard), h["tiers"]
    rows = []
    for L in range(a["p_miss"].shape[1]):
        valid = np.asarray(a["obs_miss"][k, L]) >= 0
        row = {"lead": L + 1}
        for d, pk, ok in (("miss", "p_miss", "obs_miss"), ("false_alarm", "p_fa", "obs_fa")):
            p = np.asarray(a[pk][k, L], np.float32)[valid]
            row[f"{d}_flagged"] = int((p >= t[d]["high"]["threshold"]).sum())
            row[f"{d}_observed"] = int((np.asarray(a[ok][k, L])[valid] == 1).sum())
        rows.append(row)
    return rows


@router.get("/monsoon")
def monsoon(init: str = Query(..., pattern=r"^\d{8}$")) -> dict:
    k = _init_pos("monsoon", init)
    a = _arrays("monsoon")
    return {"init": init, "targets": _hazard("monsoon")["targets"],
            "p": _round(a["p"][k]), "obs": np.asarray(a["obs"][k]).astype(int).tolist(),
            "p_active": _round(a["p_active"][k]), "p_break": _round(a["p_break"][k])}


@router.get("/cases")
def cases() -> list[dict]:
    spec = importlib.util.spec_from_file_location("case_studies", CASES_FILE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    idx = _index()
    return [{"id": k, **c} for k, c in mod.CASES.items()
            if c["hazard"] in idx and c["init"] in idx[c["hazard"]]["inits"]]
