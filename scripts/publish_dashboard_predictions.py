"""Publish one Xarray probability grid for the BustSentinel dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import xarray as xr

from src.inference.dashboard_output import (
    assign_regions,
    publish_current_predictions,
    to_prediction_geodataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="NetCDF file containing model probabilities")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variable", default="overall_bust_probability")
    parser.add_argument("--init-time")
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--region-column", default="region_id")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "current_predictions",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with xr.open_dataset(args.input) as dataset:
        predictions = to_prediction_geodataframe(
            dataset,
            variable=args.variable,
            run_id=args.run_id,
            init_time=args.init_time,
        )
    if args.boundaries:
        boundaries = gpd.read_file(args.boundaries)
        predictions = assign_regions(
            predictions,
            boundaries,
            region_column=args.region_column,
        )
    output = publish_current_predictions(
        predictions,
        args.output_directory,
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
