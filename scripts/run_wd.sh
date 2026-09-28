#!/usr/bin/env bash
# Western-disturbance pipeline: waits for the ERA5 download (scripts/download_era5_wd.py),
# then truth -> 20-year cache -> GNN and CNN, 2 seeds each (checkpoints picked on miss PR-AUC,
# since false alarms are ranked by the GEFS probability itself) -> 2022 test cache.
# Run from the project root:  nohup bash scripts/run_wd.sh > logs/wd_chain.log 2>&1 &
set -euo pipefail
PY=D:/sih-venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
stamp() { echo "=== $(date '+%F %T') $*"; }

stamp "waiting for the ERA5 WD download"
until grep -qx done logs/era5_wd.log; do
  if grep -q Traceback logs/era5_wd.log; then stamp "ERA5 download failed - see logs/era5_wd.log"; exit 1; fi
  sleep 60
done
stamp "download done: $(ls data/raw/observations/era5/wd/*.nc | wc -l) files"

stamp "truth"
$PY -m scripts.build_wd_truth
stamp "cache 2000-2019"
$PY -m scripts.build_wd_cache --years 2000-2019 --cache-dir D:/sih-data/caches/wd_20yr
for arch in gnn cnn; do
  for seed in 0 1; do
    stamp "train $arch seed $seed"
    $PY -m scripts.train_heatwave --arch $arch --seed $seed --tag _s$seed --select miss --name wd \
      --cache-dir D:/sih-data/caches/wd_20yr
  done
done
stamp "test cache 2022"
$PY -m scripts.build_wd_cache --years 2022 --store ops --cache-dir D:/sih-data/caches/wd_test_2022 \
  --standardizer D:/sih-data/caches/wd_20yr/standardizer.npz
stamp "all done"
