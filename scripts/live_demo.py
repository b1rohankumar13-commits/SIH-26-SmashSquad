"""Demo-mode live run: the fetch step is simulated (~5 s) and the forecast is read from disk.

For presentations without network or ECMWF queue risk. The analysis itself is the real pipeline
(scripts/live_forecast.py) on a forecast already in the source's store; with no date given it
uses the newest one there. Started by POST /api/live/run with demo=true.

The heavy imports (torch, models) load in a background thread while the fetch messages print, so
the simulated download also hides the ~30-40 s start-up.

    python -m scripts.live_demo --source neps --init 20220315
"""

from __future__ import annotations

import argparse
import threading
import time

LABEL = {"gefs": "NOAA GEFS (noaa-gefs-pds)", "neps": "NCMRWF NEPS-G (ECMWF TIGGE)"}
STEPS = ["rain, MSLP, 850 hPa winds and moisture, 500 hPa heights", "maximum temperature",
         "2 m and upper-air temperature, heights"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["gefs", "neps"], default="gefs")
    ap.add_argument("--init", help="YYYYMMDD on disk (default: newest in the store)")
    ap.add_argument("--delay", type=float, default=5.0)
    args = ap.parse_args()
    t0 = time.time()
    print(f"fetch: {LABEL[args.source]} {args.init or 'newest run'} 00 UTC", flush=True)

    loaded: dict = {}
    loader = threading.Thread(target=lambda: loaded.update(lf=__import__("scripts.live_forecast", fromlist=["run"])))
    loader.start()
    for s in STEPS:
        time.sleep(args.delay / (len(STEPS) + 1))
        print(f"  {s}: done", flush=True)
    loader.join()
    lf = loaded["lf"]
    init = args.init or lf.latest(args.source)
    if not lf.store_file(args.source, "rain", init).exists():
        raise SystemExit(f"No {args.source} forecast for {init} on disk (demo mode reads saved files only).")
    time.sleep(max(0.0, args.delay - (time.time() - t0)))
    print("fetch complete, analysing ...", flush=True)
    t1 = time.time()
    lf.run(args.source, init, list(lf.HAZARDS))
    print(f"done in {time.time() - t1:.0f}s -> {lf.LIVE / args.source / init}", flush=True)


if __name__ == "__main__":
    main()
