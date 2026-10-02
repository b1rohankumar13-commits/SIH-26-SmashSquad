"""Forecast bulletin: one CSV per forecast with every covered IMD subdivision x day.

Columns: day, valid date, subdivision, forecast confidence (%), grade, chance of a missed event,
chance of a false alarm, expected busting cells, whether the grid map flags cells there, and the top reasons (drivers of
the flagged areas overlapping the subdivision). The grade matches the dashboard's Confidence view
(web/src/lib/risk.ts: validation cut-offs with 25% / 10% floors). Opens directly in Excel.

  /api/bulletin/{hazard}.csv?init=YYYYMMDD                 validation-period forecast
  /api/bulletin/{hazard}.csv?init=YYYYMMDD&source=gefs      live run (gefs | neps)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from api.routes.dashboard import _arrays, _explain, _hazard, _init_pos, _subdivisions

LIVE = Path(__file__).resolve().parents[2] / "outputs" / "live"
LOW_FLOOR, MEDIUM_FLOOR = 0.25, 0.10
LEADS = 9

router = APIRouter(prefix="/api", tags=["bulletin"])


def grade(p_any: float, cuts: list[float]) -> str:
    if p_any >= max(cuts[1], LOW_FLOOR):
        return "Low"
    return "Medium" if p_any >= max(cuts[0], MEDIUM_FLOOR) else "High"


def reasons(areas: list[dict], region: int, top: int = 3) -> str:
    """Drivers of the explained areas overlapping the subdivision, weighted by expected busts inside it."""
    w: dict[str, float] = {}
    total = 0.0
    for a in areas:
        inside = a.get("regions", {}).get(str(region), 0)
        if not inside:
            continue
        weight = a["expected"] * inside / a["cells"]
        total += weight
        for d in a["drivers"]:
            w[d["name"]] = w.get(d["name"], 0.0) + weight * d["share"]
    best = sorted(w.items(), key=lambda kv: -kv[1])[:top]
    return "; ".join(f"{n} ({round(100 * v / (total or 1))}%)" for n, v in best)


@router.get("/bulletin/{hazard}.csv")
def bulletin(hazard: str, init: str = Query(..., pattern=r"^\d{8}$"),
             source: str | None = Query(None, pattern="^(gefs|neps)$")) -> Response:
    h = _hazard(hazard)
    if "regions" not in h or not h["regions"].get("confidence_cuts"):
        raise HTTPException(404, f"No region-wise output for {hazard}.")
    cuts = h["regions"]["confidence_cuts"]
    if source:
        d = LIVE / source / init / hazard
        if not (d / "region_p_miss.npy").exists():
            raise HTTPException(404, f"No {source} run of {hazard} for {init}.")
        a = {k: np.load(d / f"region_{k}.npy") for k in ("p_miss", "p_fa", "flag_miss", "flag_fa", "exp_miss", "exp_fa")}
        ex = json.loads((d / "explain.json").read_text(encoding="utf-8")) if (d / "explain.json").exists() else {}
        shift = json.loads((d.parent / "meta.json").read_text())["hazards"][hazard].get("valid_shift", -1)
    else:
        k = _init_pos(hazard, init)
        arr = _arrays(hazard)
        a = {n: np.asarray(arr[f"region_{n}"][k]) for n in ("p_miss", "p_fa", "flag_miss", "flag_fa", "exp_miss", "exp_fa")}
        ex = _explain(hazard).get(init, {})
        shift = h.get("valid_shift", -1)
    names = [f["properties"]["name"] for f in _subdivisions()["features"]]
    t0 = pd.Timestamp(f"{init[:4]}-{init[4:6]}-{init[6:]}")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["hazard", "issued", "source", "day", "valid", "subdivision", "confidence_pct", "grade",
                "chance_missed_event_pct", "chance_false_alarm_pct", "expected_busting_cells", "flagged_on_map",
                "main_reasons"])
    for L in range(LEADS):
        per = ex.get(str(L + 1), {})
        areas = per.get("miss", []) + per.get("false_alarm", [])
        valid = (t0 + pd.Timedelta(days=L + 1 + shift)).strftime("%Y-%m-%d")
        rows = []
        for r in h["regions"]["covered"]:
            pm, pf = float(a["p_miss"][L][r]), float(a["p_fa"][L][r])
            p_any = 1 - (1 - pm) * (1 - pf)
            flagged = bool(a["flag_miss"][L][r] or a["flag_fa"][L][r])
            rows.append(((p_any, float(a["exp_miss"][L][r]) + float(a["exp_fa"][L][r])), [h["label"], init, source or "GEFS reforecast (validation)", L + 1, valid, names[r],
                                 round(100 * (1 - p_any)), grade(p_any, cuts), round(100 * pm, 1), round(100 * pf, 1),
                                 round(float(a["exp_miss"][L][r]) + float(a["exp_fa"][L][r]), 1),
                                 "yes" if flagged else "", reasons(areas, r) if flagged else ""]))
        for _, row in sorted(rows, key=lambda x: (-x[0][0], -x[0][1])):
            w.writerow(row)
    name = f"bulletin_{hazard}_{init}{'_' + source if source else ''}.csv"
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
