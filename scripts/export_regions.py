"""Region-wise bust risk over IMD meteorological subdivisions, for the dashboard.

Boundaries: IMD's subdivision file (mausam.imd.gov.in .../district_shapefiles/sd_boundary.json,
UTM 44N), saved as data/raw/boundaries/imd_subdivisions.json, is heavily generalised, so coasts and
state borders come from the Survey of India state file and IMD's lines are used only to split states
that hold several subdivisions. Written (lightly simplified) to outputs/dashboard/subdivisions.geojson.

Each 0.5 deg cell belongs to the subdivision containing its centre (island subdivisions with no
centre inside take the cells their polygons touch). For every grid hazard, forecast and lead:
  score  = 1 - prod(1 - p_i) over the subdivision's observed cells (p_i = calibrated cell risk)
  target = at least one of those cells busted
and an isotonic fit per lead day on the validation forecasts turns the score into "chance GEFS
busts somewhere in this subdivision" (the same in-sample calibration the cell map uses). The
GEFS-only baseline goes through the same steps, so region skill is model vs GEFS alone; it is
scored only on subdivisions where the bust is possible (a cell with GEFS < 20% / >= 50%).

Writes outputs/dashboard/<hazard>/region_{p_miss,p_fa,obs_miss,obs_fa}.npy [n_init, lead, R]
(obs -1 = no observed cell), region_flag_{miss,fa}.npy (1 = the grid map colours MIN_FLAGGED+ cells
there; the region map only shades those) and index.json[hazard]["regions"] (skill).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.export_explanations import calibrated, display_cut
from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "data" / "raw" / "boundaries" / "imd_subdivisions.json"
# DataMeet copy of the Survey of India state boundaries (github.com/datameet/maps,
# Survey-of-India-Index-Maps/Boundaries/India-States.*)
STATES = PROJECT_ROOT / "data" / "raw" / "boundaries" / "India-States"
OUT = PROJECT_ROOT / "outputs" / "dashboard"
LEADS = 9
# A subdivision is shaded when the grid map colours at least this many of its cells (or 10% of a
# small one): one stray cell should not paint a whole subdivision.
MIN_FLAGGED = 3

# IMD's short labels -> readable names
NAMES = {
    "A & N ISLAND": "Andaman & Nicobar Islands", "HAR. CHD & DELHI": "Haryana, Chandigarh & Delhi",
    "N M M T": "Nagaland, Manipur, Mizoram & Tripura", "SHWB & SIKKIM": "Sub-Himalayan West Bengal & Sikkim",
    "S. I. KARNATAKA": "South Interior Karnataka", "N. I. KARNATAKA": "North Interior Karnataka",
    "TAMILNADU & PONDICHERY": "Tamil Nadu & Puducherry", "ORISSA": "Odisha",
    "KONKAN & GOA": "Konkan & Goa", "ASSAM & MEGHALAYA": "Assam & Meghalaya",
    "SAURASHTRA & KUTCH": "Saurashtra & Kutch", "JAMMU & KASHMIR": "Jammu & Kashmir",
}


def pretty(name: str) -> str:
    return NAMES.get(name, name.title())


# Survey of India state -> IMD subdivision(s). Where a state holds several subdivisions it is cut
# along IMD's (coarse) internal lines; coasts and state borders come from the detailed state file.
STATE_SUBDIVISIONS = {
    "Andaman & Nicobar Island": ["A & N ISLAND"], "Lakshadweep": ["LAKSHADWEEP"],
    "Andhra Pradesh": ["COASTAL ANDHRA PRADESH", "RAYALASEEMA"], "Telangana": ["TELANGANA"],
    "Arunanchal Pradesh": ["ARUNACHAL PRADESH"], "Assam": ["ASSAM & MEGHALAYA"], "Meghalaya": ["ASSAM & MEGHALAYA"],
    "Manipur": ["N M M T"], "Mizoram": ["N M M T"], "Nagaland": ["N M M T"], "Tripura": ["N M M T"],
    "Sikkim": ["SHWB & SIKKIM"], "West Bengal": ["GANGETIC WEST BENGAL", "SHWB & SIKKIM"],
    "Bihar": ["BIHAR"], "Jharkhand": ["JHARKHAND"], "Odisha": ["ORISSA"], "Chhattisgarh": ["CHHATTISGARH"],
    "Uttar Pradesh": ["WEST UTTAR PRADESH", "EAST UTTAR PRADESH"], "Uttarakhand": ["UTTARAKHAND"],
    "Haryana": ["HAR. CHD & DELHI"], "Chandigarh": ["HAR. CHD & DELHI"], "NCT of Delhi": ["HAR. CHD & DELHI"],
    "Punjab": ["PUNJAB"], "Himachal Pradesh": ["HIMACHAL PRADESH"], "Jammu & Kashmir": ["JAMMU & KASHMIR"],
    "Rajasthan": ["WEST RAJASTHAN", "EAST RAJASTHAN"],
    "Madhya Pradesh": ["WEST MADHYA PRADESH", "EAST MADHYA PRADESH"],
    "Gujarat": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
    "Dadara & Nagar Havelli": ["GUJARAT REGION"], "Daman & Diu": ["GUJARAT REGION", "SAURASHTRA & KUTCH"],
    "Goa": ["KONKAN & GOA"],
    "Maharashtra": ["KONKAN & GOA", "MADHYA MAHARASHTRA", "MARATHWADA", "VIDARBHA"],
    "Karnataka": ["COASTAL KARNATAKA", "N. I. KARNATAKA", "S. I. KARNATAKA"],
    "Kerala": ["KERALA"], "Tamil Nadu": ["TAMILNADU & PONDICHERY"],
    # Puducherry's Mahe and Yanam enclaves sit in Kerala and coastal Andhra
    "Puducherry": ["TAMILNADU & PONDICHERY", "KERALA", "COASTAL ANDHRA PRADESH"],
}


def load_subdivisions() -> list[dict]:
    """IMD subdivisions with Survey of India coasts and state borders (IMD's own file is too
    generalised - ~100 vertices a state - and sits 10-20 km off the basemap along coasts)."""
    import shapefile
    from shapely.ops import unary_union
    utm = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True).transform
    feats = sorted(json.loads(SRC.read_text())["features"], key=lambda f: f["properties"]["OBJECTID_1"])
    coarse = {f["properties"]["subdivisio"]: transform(utm, shape(f["geometry"])).buffer(0) for f in feats}
    pieces: dict[str, list] = {k: [] for k in coarse}
    for rec in shapefile.Reader(str(STATES)).iterShapeRecords():
        state, geom = rec.record[0], shape(rec.shape.__geo_interface__).buffer(0)
        subs = STATE_SUBDIVISIONS[state]
        if len(subs) == 1:
            pieces[subs[0]].append(geom)
            continue
        parts = {k: geom.intersection(coarse[k]) for k in subs}
        rest = geom.difference(unary_union(list(parts.values())))
        for frag in getattr(rest, "geoms", [rest]):  # slivers outside IMD's coarse outline: nearest candidate
            if not frag.is_empty and frag.area > 0:
                k = min(subs, key=lambda k: coarse[k].distance(frag))
                parts[k] = parts[k].union(frag)
        for k, g in parts.items():
            if not g.is_empty:
                pieces[k].append(g)
    return [{"name": pretty(f["properties"]["subdivisio"]), "zone": f["properties"]["region_cod"].title(),
             "geom": unary_union(pieces[f["properties"]["subdivisio"]]).buffer(0)} for f in feats]


def cell_regions(subs: list[dict]) -> np.ndarray:
    import shapely
    lats, lons = canonical_centres(0.5)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    reg = np.full(la.shape, -1, np.int16)
    for r, s in enumerate(subs):
        reg[(reg < 0) & shapely.contains_xy(s["geom"], lo, la)] = r
    for r, s in enumerate(subs):  # small island subdivisions: cells their polygons touch
        if (reg == r).sum() == 0:
            cells = shapely.box(lo - 0.25, la - 0.25, lo + 0.25, la + 0.25)
            reg[(reg < 0) & shapely.intersects(cells, s["geom"])] = r
    return reg


def write_geojson(subs: list[dict]) -> None:
    feats = [{"type": "Feature", "id": r,
              "properties": {"id": r, "name": s["name"], "zone": s["zone"]},
              "geometry": mapping(s["geom"].simplify(0.02, preserve_topology=True))}
             for r, s in enumerate(subs)]
    geo = json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"))
    # 4 decimals (~10 m) is plenty for a 0.5 deg product
    import re
    geo = re.sub(r"(\d+\.\d{4})\d+", r"\1", geo)
    (OUT / "subdivisions.geojson").write_text(geo)


def noisy_or(p: np.ndarray, valid: np.ndarray, reg: np.ndarray, R: int) -> tuple[np.ndarray, np.ndarray]:
    """p, valid: [n, H, W] -> score [n, R] and count of valid cells [n, R]."""
    lg = np.log1p(-np.clip(p, 0, 0.999999)) * valid
    score = np.zeros((p.shape[0], R)); count = np.zeros((p.shape[0], R), int)
    for r in range(R):
        m = reg == r
        score[:, r] = 1 - np.exp(lg[:, m].sum(1))
        count[:, r] = valid[:, m].sum(1)
    return score, count


def region_any(obs: np.ndarray, valid: np.ndarray, reg: np.ndarray, R: int) -> np.ndarray:
    out = np.full((obs.shape[0], R), -1, np.int8)
    for r in range(R):
        m = reg == r
        has = valid[:, m].any(1)
        out[has, r] = (obs[:, m] == 1).any(1)[has]
    return out


def hazard_regions(name: str, entry: dict, reg: np.ndarray, R: int) -> dict:
    a = {k: np.load(OUT / name / f"{k}.npy") for k in ("p_miss", "p_fa", "obs_miss", "obs_fa", "b_miss", "b_fa", "ens_prob")}
    n = a["p_miss"].shape[0]
    valid = a["obs_miss"] >= 0
    out, meta = {}, {"skill": {}}
    for d, pk, bk, ok, possible in (("miss", "p_miss", "b_miss", "obs_miss", lambda e: e < 0.2),
                                    ("false_alarm", "p_fa", "b_fa", "obs_fa", lambda e: e >= 0.5)):
        cal = entry["tiers"][d]["calibration"]
        P = np.zeros((n, LEADS, R), np.float16); O = np.full((n, LEADS, R), -1, np.int8)
        F = np.zeros((n, LEADS, R), np.int8)  # subdivision has a cell the grid map colours
        cut = display_cut(entry["tiers"][d])
        ys, ms, bs = [], [], []
        for L in range(LEADS):
            pc = calibrated(a[pk][:, L].astype(np.float32), cal, L + 1)
            shown = (pc >= cut) & valid[:, L]
            for r in range(R):
                m = reg == r
                F[:, L, r] = shown[:, m].sum(1) >= min(MIN_FLAGGED, max(1, round(0.1 * m.sum())))
            # GEFS-only goes through its own isotonic so the two scores are on the same footing
            bl = a[bk][:, L].astype(np.float32)
            y = region_any(a[ok][:, L], valid[:, L], reg, R)
            s, _ = noisy_or(pc, valid[:, L], reg, R)
            fin = y >= 0
            # skill only where this bust can happen: some observed cell with GEFS < 20% (miss) / >= 50% (FA)
            elig = region_any(possible(a["ens_prob"][:, L].astype(np.float32)).astype(np.int8), valid[:, L], reg, R) == 1
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s[fin], y[fin])
            P[:, L] = iso.predict(s.ravel()).reshape(s.shape); O[:, L] = y
            cell = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                bl[valid[:, L]], (a[ok][:, L][valid[:, L]] == 1))
            sb, _ = noisy_or(cell.predict(bl.ravel()).reshape(bl.shape), valid[:, L], reg, R)
            ys.append(y[fin & elig]); ms.append(P[:, L][fin & elig].astype(np.float64)); bs.append(sb[fin & elig])
        y, m, b = np.concatenate(ys), np.concatenate(ms), np.concatenate(bs)
        meta["skill"][d] = {"model": {"auroc": float(roc_auc_score(y, m)), "pr_auc": float(average_precision_score(y, m))},
                            "baseline": {"auroc": float(roc_auc_score(y, b)), "pr_auc": float(average_precision_score(y, b))},
                            "base_rate": float(y.mean()), "n": int(len(y))}
        out[d] = (P, O, F)
    np.save(OUT / name / "region_p_miss.npy", out["miss"][0]); np.save(OUT / name / "region_obs_miss.npy", out["miss"][1])
    np.save(OUT / name / "region_p_fa.npy", out["false_alarm"][0]); np.save(OUT / name / "region_obs_fa.npy", out["false_alarm"][1])
    np.save(OUT / name / "region_flag_miss.npy", out["miss"][2]); np.save(OUT / name / "region_flag_fa.npy", out["false_alarm"][2])
    meta["covered"] = sorted({int(r) for r in np.unique(reg[valid.any((0, 1))]) if r >= 0})
    return meta


def main() -> None:
    subs = load_subdivisions()
    R = len(subs)
    write_geojson(subs)
    reg = cell_regions(subs)
    np.save(OUT / "regions.npy", reg)
    print(f"{R} subdivisions | cells assigned {int((reg >= 0).sum())} | "
          f"smallest {min((reg == r).sum() for r in range(R))} cells", flush=True)
    index = json.loads((OUT / "index.json").read_text())
    for name, entry in index.items():
        if entry["kind"] != "grid":
            continue
        entry["regions"] = hazard_regions(name, entry, reg, R)
        for d, s in entry["regions"]["skill"].items():
            print(f"  {name}/{d}: {len(entry['regions']['covered'])} subdivisions | any-bust rate {100 * s['base_rate']:.1f}% | "
                  f"AUROC {s['model']['auroc']:.3f} vs {s['baseline']['auroc']:.3f} | "
                  f"PR-AUC {s['model']['pr_auc']:.3f} vs {s['baseline']['pr_auc']:.3f} (model vs GEFS-only)", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
