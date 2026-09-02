"""Download one NCEP TIGGE rainfall-ensemble case from the ECMWF Data Store."""

from pathlib import Path

import cdsapi


DATASET = "tigge-forecasts"
REQUEST = {
    "origin": "ncep",
    "year": "2024",
    "month": "07",
    "day": "01",
    "time": "00:00",
    "level_type": "single_level",
    "variable": ["total_precipitation"],
    "forecast_type": "perturbed_forecast",
    "leadtime_hour": [
        "0",
        "24",
        "48",
        "72",
        "96",
        "120",
        "144",
        "168",
        "192",
        "216",
        "240",
    ],
    "data_format": "grib",
    # ECDS area order: north, west, south, east.
    "area": [38, 66, 6, 100],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "forecasts"
    / "tigge"
    / "ncep"
    / "2024"
    / "07"
    / "01"
    / "tigge_ncep_20240701_00_tp_ensemble.grib2"
)


def download_tigge() -> Path:
    """Submit the pilot request and return the downloaded GRIB2 path."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(DATASET, REQUEST, str(OUTPUT_FILE))
    return OUTPUT_FILE


if __name__ == "__main__":
    downloaded_file = download_tigge()
    print(f"Downloaded TIGGE pilot file to: {downloaded_file}")
