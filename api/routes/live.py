"""Live forecast analysis API over outputs/live/<source>/<init>/ (scripts/live_forecast.py).

Runs are read from disk; POST /api/live/run starts the pipeline (optionally downloading the
forecast first) in the ML environment as a background process and returns a job id to poll.
Grid payloads match /api/grid so the web map code is shared.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE = PROJECT_ROOT / "outputs" / "live"
JOBS = LIVE / "jobs"
# The API runs in the dashboard venv; models run in the ML venv (torch, xarray, cfgrib).
ML_PYTHON = os.environ.get("SIH_ML_PYTHON", r"D:\sih-venv\Scripts\python.exe")
SOURCES = ("gefs", "neps")

router = APIRouter(prefix="/api/live", tags=["live"])
_procs: dict[str, subprocess.Popen] = {}


def _run_dir(source: str, init: str) -> Path:
    if source not in SOURCES:
        raise HTTPException(404, f"Unknown source '{source}'.")
    d = LIVE / source / init
    if not (d / "meta.json").exists():
        raise HTTPException(404, f"No {source} run for {init}.")
    return d


def _load(d: Path, name: str):
    p = d / f"{name}.npy"
    return np.load(p) if p.exists() else None


def _round(a, nd: int = 4) -> list:
    return np.round(np.asarray(a, np.float64), nd).tolist()


@router.get("/runs")
def runs() -> list[dict]:
    """Every live run on disk, newest forecast first."""
    out = []
    for m in LIVE.glob("*/*/meta.json"):
        meta = json.loads(m.read_text())
        out.append({k: meta[k] for k in ("source", "source_label", "init", "created", "realtime")} |
                   {"hazards": meta["hazards"]})
    return sorted(out, key=lambda r: (r["init"], r["source"]), reverse=True)


@router.get("/{source}/{init}/grid/{hazard}")
def grid(source: str, init: str, hazard: str, lead: int = Query(..., ge=1, le=9)) -> dict:
    d = _run_dir(source, init) / hazard
    if not d.exists():
        raise HTTPException(404, f"{hazard} was not run for this forecast.")
    L = lead - 1
    dom = np.load(d / "domain.npy")
    lat, lon = np.meshgrid(np.load(PROJECT_ROOT / "outputs" / "dashboard" / "lats.npy"),
                           np.load(PROJECT_ROOT / "outputs" / "dashboard" / "lons.npy"), indexing="ij")
    om, of = _load(d, "obs_miss"), _load(d, "obs_fa")
    unknown = [-1] * int(dom.sum())
    return {"init": init, "lead": lead, "lat": _round(lat[dom], 2), "lon": _round(lon[dom], 2),
            "p_miss": _round(np.load(d / "p_miss.npy")[L][dom]), "p_fa": _round(np.load(d / "p_fa.npy")[L][dom]),
            "obs_miss": om[L][dom].astype(int).tolist() if om is not None else unknown,
            "obs_fa": of[L][dom].astype(int).tolist() if of is not None else unknown,
            "ens_prob": _round(np.load(d / "ens_prob.npy")[L][dom], 3)}


@router.get("/{source}/{init}/regions/{hazard}")
def regions(source: str, init: str, hazard: str) -> dict:
    d = _run_dir(source, init) / hazard
    names = [f["properties"]["name"] for f in
             json.loads((PROJECT_ROOT / "outputs" / "dashboard" / "subdivisions.geojson").read_text())["features"]]
    idx = json.loads((PROJECT_ROOT / "outputs" / "dashboard" / "index.json").read_text())
    arr = {k: np.load(d / f"region_{k}.npy") for k in ("p_miss", "p_fa", "obs_miss", "obs_fa", "flag_miss", "flag_fa")}
    return {"init": init, "names": names, "covered": idx[hazard]["regions"]["covered"],
            **{k: (_round(v, 3) if k.startswith("p_") else v.astype(int).tolist()) for k, v in arr.items()}}


@router.get("/{source}/{init}/explain/{hazard}")
def explain(source: str, init: str, hazard: str, lead: int = Query(..., ge=1, le=9)) -> dict:
    p = _run_dir(source, init) / hazard / "explain.json"
    per = json.loads(p.read_text(encoding="utf-8")).get(str(lead), {}) if p.exists() else {}
    return {"miss": per.get("miss", []), "false_alarm": per.get("false_alarm", [])}


class RunRequest(BaseModel):
    source: str = Field(pattern="^(gefs|neps)$")
    init: str | None = Field(default=None, pattern=r"^\d{8}$")
    fetch: bool = True
    realtime: bool = True


@router.post("/run")
def start(req: RunRequest) -> dict:
    """Start scripts/live_forecast.py in the background; poll /api/live/jobs/{id}."""
    if any(p.poll() is None for p in _procs.values()):
        raise HTTPException(409, "A live run is already in progress.")
    JOBS.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex[:10]
    cmd = [ML_PYTHON, "-u", "-m", "scripts.live_forecast", "--source", req.source]
    cmd += ["--init", req.init] if req.init else []
    cmd += (["--fetch"] if req.fetch else []) + (["--realtime"] if req.realtime else [])
    log = open(JOBS / f"{job}.log", "w", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    _procs[job] = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    (JOBS / f"{job}.json").write_text(json.dumps(req.model_dump()))
    return {"job": job}


@router.get("/jobs/{job}")
def job(job: str) -> dict:
    log = JOBS / f"{job}.log"
    if not log.exists():
        raise HTTPException(404, "Unknown job.")
    p = _procs.get(job)
    code = p.poll() if p else None
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    status = "running" if p and code is None else ("done" if code == 0 else "failed")
    req = json.loads((JOBS / f"{job}.json").read_text())
    done = next((l for l in reversed(lines) if l.startswith("done in")), None)
    init = done.split("\\")[-1].split("/")[-1] if done else None
    return {"job": job, "status": status, "source": req["source"], "init": init, "log": lines[-12:]}
