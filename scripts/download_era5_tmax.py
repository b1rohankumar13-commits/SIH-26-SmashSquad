"""Download ERA5 daily-maximum 2 m temperature over India as heatwave ground truth.

Credentials come from the environment (CDSAPI_KEY, and CDSAPI_URL which we default
to the public CDS endpoint) or ~/.cdsapirc - this script never handles the key
value itself. One NetCDF per year of daily-max 2 m temperature on the India domain;
regrid to the 0.5 deg model grid happens at cache-build time.
"""

from __future__ import annotations

import argparse
import os
from calendar import monthrange
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "observations" / "era5" / "tmax"
# India domain, matching the reforecast config [North, West, South, East].
AREA = [38.0, 65.0, 5.0, 100.0]
DATASET = "derived-era5-single-levels-daily-statistics"


def download_month(client, year: int, month: int, out: Path) -> None:
    # One month per request: CDS stalls/queues on whole-year requests, but small
    # monthly jobs clear quickly and the run stays resumable per month.
    ndays = monthrange(year, month)[1]
    request = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+05:30",   # IST, so the daily max aligns to the Indian calendar day
        "frequency": "1_hourly",
        "area": AREA,
    }
    client.retrieve(DATASET, request, str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=str, default="2000-2009")
    parser.add_argument("--year", type=int, default=None, help="Single year (overrides --years).")
    args = parser.parse_args()

    import cdsapi
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY")
    if key:  # credentials via env (key value never logged); else fall back to ~/.cdsapirc
        client = cdsapi.Client(url=url, key=key)
    else:
        client = cdsapi.Client()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.year:
        years = [args.year]
    else:
        a, b = (int(y) for y in args.years.split("-"))
        years = list(range(a, b + 1))

    for year in years:
        for month in range(1, 13):
            out = OUT_ROOT / f"era5_tmax_{year}{month:02d}.nc"
            if out.exists() and out.stat().st_size > 0:
                print(f"{year}-{month:02d}: exists, skipping", flush=True)
                continue
            print(f"{year}-{month:02d}: requesting daily-max 2m T over India ...", flush=True)
            download_month(client, year, month, out)
            print(f"{year}-{month:02d}: wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
