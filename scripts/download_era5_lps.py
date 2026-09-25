"""Download ERA5 00 UTC fields for low-pressure-system (LPS) tracking truth, 2000-2009.

Matches the GEFS reforecast 'instant' fields (lead L = init + L days, 00 UTC):
  --kind pl : u/v wind at 850 hPa   (reanalysis-era5-pressure-levels)
  --kind sl : mean sea-level pressure (reanalysis-era5-single-levels)
0.5 deg grid over the India box plus a 3 deg margin so tracks are not clipped at the
model-domain edge. One NetCDF per month (monthly CDS requests clear fastest and keep
the run resumable). Credentials come from CDSAPI_KEY/CDSAPI_URL or ~/.cdsapirc; the
key value is never read or logged here.
"""

from __future__ import annotations

import argparse
import os
from calendar import monthrange
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "lps"
AREA = [41.0, 62.0, 2.0, 103.0]  # [N, W, S, E]: model box 5-38N 65-100E + 3 deg margin
KINDS = {
    "pl": ("reanalysis-era5-pressure-levels",
           {"variable": ["u_component_of_wind", "v_component_of_wind"], "pressure_level": ["850"]}),
    "sl": ("reanalysis-era5-single-levels", {"variable": ["mean_sea_level_pressure"]}),
}


def download_month(client, kind: str, year: int, month: int, out: Path) -> None:
    dataset, extra = KINDS[kind]
    request = {
        "product_type": ["reanalysis"],
        "year": [str(year)], "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, monthrange(year, month)[1] + 1)],
        "time": ["00:00"], "area": AREA, "grid": [0.5, 0.5],
        "data_format": "netcdf", "download_format": "unarchived",
        **extra,
    }
    client.retrieve(dataset, request, str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=list(KINDS), required=True)
    ap.add_argument("--years", default="2000-2009")
    args = ap.parse_args()

    import cdsapi
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY")
    client = cdsapi.Client(url=url, key=key) if key else cdsapi.Client()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    a, b = (int(y) for y in args.years.split("-"))
    for year in range(a, b + 1):
        for month in range(1, 13):
            out = OUT_ROOT / f"era5_{args.kind}_{year}{month:02d}.nc"
            if out.exists() and out.stat().st_size > 0:
                print(f"{args.kind} {year}-{month:02d}: exists, skipping", flush=True)
                continue
            tmp = out.with_suffix(".part")
            print(f"{args.kind} {year}-{month:02d}: requesting ...", flush=True)
            download_month(client, args.kind, year, month, tmp)
            tmp.replace(out)
            print(f"{args.kind} {year}-{month:02d}: wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
