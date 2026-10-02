"""NCMRWF NEPS-G from TIGGE -> the operational store layout the live pipeline reads.

    python -m scripts.neps_tigge --date 20260925 --dry-run   # print the requests only
    python -m scripts.neps_tigge --date 20260925             # download + convert
    python -m scripts.neps_tigge --date 20260925 --convert-only

TIGGE (ECMWF Data Store, dataset "tigge-forecasts", origin "ncmrwf") carries NEPS-G with a
control and 11 perturbed members at 12-hourly steps, released after a 48-hour embargo. Access
needs an ECMWF account with the TIGGE licence accepted; credentials come from ~/.cdsapirc or
ECDS_URL / ECDS_KEY (the key is never read or printed here).

The models were trained on 5 GEFS members, so the control plus perturbed members 1-4 are kept
(ensemble spreads and probabilities then mean the same thing). Fields are reduced exactly as the
GEFS stores are (src/acquisition/download_gefs_reforecast.py):
  rain  total_precipitation  day-d total = tp(24d) - tp(24(d-1))        [kg m-2]
        mean_sea_level_pressure, u/v/q 850 hPa, gh 500 hPa  at 24d h (00 UTC)
  tmax  tmax_2m  max of the 6-hourly maxima ending 24(d-1)+12 and 24d h [K]; with 12-hourly
        output this covers 06-12 UTC (the Indian afternoon) and 18-24 UTC
  mv    pres_msl, tmp_2m, tmp_850, hgt_500, hgt_300 at 24d h; spfh_2m from 2 m dewpoint and
        surface pressure
and interpolated to the 0.5 deg cell-centred model grid (5.25-37.75N, 65.25-99.75E).
Writes data/interim/forecasts/neps/{rain,tmax,mv}/<init>00.nc.

Variable names match the ECDS request schema (api/retrieve/v1/processes/tigge-forecasts); TIGGE has
no regridding option, so fields arrive on the native NEPS grid and are interpolated here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

from src.models.graphnet.grid_graph import canonical_centres

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "forecasts" / "tigge" / "ncmrwf"
STORE = PROJECT_ROOT / "data" / "interim" / "forecasts" / "neps"
DATASET = "tigge-forecasts"
AREA = [39, 64, 4, 101]  # N, W, S, E: the model box plus a 1 deg margin for interpolation
LEADS = 9
MEMBERS = [0, 1, 2, 3, 4]  # control + perturbed 1-4
MEMBER_NAMES = ["c00", "p01", "p02", "p03", "p04"]
DAY_STEPS = [str(24 * d) for d in range(0, LEADS + 1)]
HALF_DAY_STEPS = [str(12 * k) for k in range(1, 2 * LEADS + 1)]
FORECAST_TYPES = {"cf": "control_forecast", "pf": "perturbed_forecast"}
PRODUCTS = {
    "surface": {"level_type": "single_level", "leadtime_hour": DAY_STEPS,
                "variable": ["total_precipitation", "mean_sea_level_pressure", "2_m_temperature",
                             "2_m_dewpoint_temperature", "surface_pressure"]},
    "tmax": {"level_type": "single_level", "leadtime_hour": HALF_DAY_STEPS,
             "variable": ["maximum_2_m_temperature_in_the_last_6_hours"]},
    # One request per level, holding only the variables used there: a combined request returns
    # every variable at every level (~2.5x the data for the 11 perturbed members).
    "p850": {"level_type": "pressure", "level_value": ["850_hpa"], "leadtime_hour": DAY_STEPS[1:],
             "variable": ["specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]},
    "p500": {"level_type": "pressure", "level_value": ["500_hpa"], "leadtime_hour": DAY_STEPS[1:],
             "variable": ["geopotential_height"]},
    "p300": {"level_type": "pressure", "level_value": ["300_hpa"], "leadtime_hour": DAY_STEPS[1:],
             "variable": ["geopotential_height"]},
}


def requests(init: str) -> list[tuple[Path, dict]]:
    out = []
    for product, spec in PRODUCTS.items():
        for ft, name in FORECAST_TYPES.items():
            req = {"origin": "ncmrwf", "year": init[:4], "month": init[4:6], "day": init[6:8], "time": "00:00",
                   "forecast_type": name, "data_format": "grib", "area": AREA,
                   **{k: v for k, v in spec.items()}}
            out.append((RAW / init / f"neps_{init}_00_{product}_{ft}.grib2", req))
    return out


def download(init: str) -> None:
    """Submit all requests at once: each waits in ECMWF's tape queue for minutes, so in
    parallel the total is about the slowest one rather than the sum."""
    from concurrent.futures import ThreadPoolExecutor

    import cdsapi
    url = os.environ.get("ECDS_URL", "https://ecds.ecmwf.int/api")

    def one(item: tuple[Path, dict]) -> None:
        path, req = item
        product, ft = path.stem.split("_00_")[1].rsplit("_", 1)
        have = raw_file(path.stem.split("_")[1], product, ft)
        if have.exists() and have.stat().st_size > 0:
            print(f"{path.name}: exists", flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"{path.name}: requesting ...", flush=True)
        cdsapi.Client(url=url, key=os.environ.get("ECDS_KEY"), quiet=True).retrieve(DATASET, req, str(tmp))
        tmp.replace(path)
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB", flush=True)

    todo = requests(init)
    with ThreadPoolExecutor(len(todo)) as pool:
        list(pool.map(one, todo))  # re-raises the first failure


def raw_file(init: str, product: str, ft: str) -> Path:
    """Per-level pressure files, or the older all-level "pressure" file when that is what is on disk."""
    f = RAW / init / f"neps_{init}_00_{product}_{ft}.grib2"
    combined = RAW / init / f"neps_{init}_00_pressure_{ft}.grib2"
    return combined if product.startswith("p") and product[1:].isdigit() and not f.exists() and combined.exists() else f


def read(init: str, product: str, short: str, level: int | None = None) -> xr.DataArray:
    """[member, step, lat, lon] for the kept members, control first."""
    parts = []
    for ft in FORECAST_TYPES:
        keys = {"shortName": short, **({"level": level} if level else {})}
        ds = xr.open_dataset(raw_file(init, product, ft), engine="cfgrib",
                             backend_kwargs={"filter_by_keys": keys, "indexpath": ""})
        a = ds[list(ds.data_vars)[0]]
        if "number" not in a.dims:
            a = a.expand_dims(number=[0])
        parts.append(a)
    a = xr.concat(parts, dim="number").sel(number=MEMBERS)
    a = a.assign_coords(step=(a["step"].values / np.timedelta64(1, "h")).astype(int))
    a = a.sortby("latitude").sortby("longitude")
    lats, lons = canonical_centres(0.5)
    return a.interp(latitude=lats, longitude=lons, method="linear").transpose("number", "step", "latitude", "longitude")


def at_days(a: xr.DataArray) -> np.ndarray:
    return a.sel(step=[24 * d for d in range(1, LEADS + 1)]).values.astype(np.float32)


def to_store(init: str, fields: dict[str, np.ndarray], kind: str, levels: dict[str, dict[int, np.ndarray]] | None = None) -> None:
    lats, lons = canonical_centres(0.5)
    coords = {"run": [f"{init}00"], "member": MEMBER_NAMES, "lead": np.arange(1, LEADS + 1),
              "latitude": lats, "longitude": lons}
    dims = ("run", "member", "lead", "latitude", "longitude")
    data = {k: (dims, v[None]) for k, v in fields.items()}
    if levels:
        coords["level"] = [500, 850]
        for k, per in levels.items():
            arr = np.full((2, 1, len(MEMBERS), LEADS, lats.size, lons.size), np.nan, np.float32)
            for i, lev in enumerate((500, 850)):
                if lev in per:
                    arr[i, 0] = per[lev]
            data[k] = (("level",) + dims, arr)
    out = STORE / kind / f"{init}00.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(data, coords=coords, attrs={"source": "NCMRWF NEPS-G via TIGGE (ECMWF)",
                                           "members": "control + perturbed 1-4"}).to_netcdf(out)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}", flush=True)


def convert(init: str) -> None:
    tp = read(init, "surface", "tp")
    rain = np.diff(tp.sel(step=[24 * d for d in range(0, LEADS + 1)]).values, axis=1).clip(0).astype(np.float32)
    msl = at_days(read(init, "surface", "msl"))
    to_store(init, {"total_precipitation": rain, "mean_sea_level_pressure": msl}, "rain", levels={
        "u_wind": {850: at_days(read(init, "p850", "u", 850))},
        "v_wind": {850: at_days(read(init, "p850", "v", 850))},
        "specific_humidity": {850: at_days(read(init, "p850", "q", 850))},
        "geopotential_height": {500: at_days(read(init, "p500", "gh", 500))},
    })
    mx = read(init, "tmax", "mx2t6")
    tmax = np.stack([mx.sel(step=[24 * (d - 1) + 12, 24 * d]).max("step").values for d in range(1, LEADS + 1)], 1)
    to_store(init, {"tmax_2m": tmax.astype(np.float32)}, "tmax")
    td, sp = at_days(read(init, "surface", "2d")), at_days(read(init, "surface", "sp"))
    e = 611.2 * np.exp(17.67 * (td - 273.15) / (td - 29.65))  # vapour pressure (Pa), Bolton 1980
    to_store(init, {"pres_msl": msl, "tmp_2m": at_days(read(init, "surface", "2t")),
                    "spfh_2m": (0.622 * e / (sp - 0.378 * e)).astype(np.float32),
                    "tmp_850": at_days(read(init, "p850", "t", 850)),
                    "hgt_500": at_days(read(init, "p500", "gh", 500)),
                    "hgt_300": at_days(read(init, "p300", "gh", 300))}, "mv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="init YYYYMMDD (00 UTC)")
    ap.add_argument("--dry-run", action="store_true", help="print the TIGGE requests and exit")
    ap.add_argument("--convert-only", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        for path, req in requests(args.date):
            print(path.name, req)
        return
    if not args.convert_only:
        download(args.date)
    convert(args.date)


if __name__ == "__main__":
    main()
