"""Download ERA5 00 UTC upper-air fields for western-disturbance (WD) truth.

Geopotential and u/v wind at 500 and 200 hPa (reanalysis-era5-pressure-levels), one
request per month. WDs are winter systems, so only October-April is fetched. The box
reaches west to 40E so troughs are seen over Iran/Afghanistan before they enter the
model domain (5-38N 65-100E). Credentials come from CDSAPI_KEY/CDSAPI_URL or
~/.cdsapirc; the key value is never read or logged here.
"""

from __future__ import annotations

import argparse
import os
from calendar import monthrange
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "wd"
AREA = [45.0, 40.0, 20.0, 100.0]  # [N, W, S, E]
MONTHS = (1, 2, 3, 4, 10, 11, 12)
DATASET = "reanalysis-era5-pressure-levels"
EXTRA = {"variable": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
         "pressure_level": ["200", "500"]}


def download_month(client, year: int, month: int, out: Path) -> None:
    request = {
        "product_type": ["reanalysis"],
        "year": [str(year)], "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, monthrange(year, month)[1] + 1)],
        "time": ["00:00"], "area": AREA, "grid": [0.5, 0.5],
        "data_format": "netcdf", "download_format": "unarchived",
        **EXTRA,
    }
    client.retrieve(DATASET, request, str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2000-2019,2022", help="ranges and single years, comma separated")
    args = ap.parse_args()

    import cdsapi
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY")
    client = cdsapi.Client(url=url, key=key) if key else cdsapi.Client()

    years = []
    for part in args.years.split(","):
        a, _, b = part.partition("-")
        years += range(int(a), int(b or a) + 1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for year in years:
        for month in MONTHS:
            out = OUT_ROOT / f"era5_wd_{year}{month:02d}.nc"
            if out.exists() and out.stat().st_size > 0:
                print(f"{year}-{month:02d}: exists, skipping", flush=True)
                continue
            tmp = out.with_suffix(".part")
            print(f"{year}-{month:02d}: requesting ...", flush=True)
            download_month(client, year, month, tmp)
            tmp.replace(out)
            print(f"{year}-{month:02d}: wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
