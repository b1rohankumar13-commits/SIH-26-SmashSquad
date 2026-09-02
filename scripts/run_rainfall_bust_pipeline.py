"""Run rainfall metrics, threshold fitting, and honest bust labelling in order."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_rainfall_metrics import build_metrics
from scripts.fit_rainfall_bust_thresholds import fit_thresholds
from scripts.label_rainfall_bust_events import label_events


def main():
    print("Stage 1/3: calculating grid and event metrics")
    build_metrics()
    print("Stage 2/3: fitting historical normalization and Q90/Q95 thresholds")
    fit_thresholds()
    print("Stage 3/3: creating candidate or strict labels")
    label_events()
    print("Rainfall bust pipeline completed.")


if __name__ == "__main__":
    main()
