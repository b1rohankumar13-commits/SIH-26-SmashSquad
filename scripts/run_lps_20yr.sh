#!/usr/bin/env bash
# Overnight 20-year LPS pipeline: waits for the ERA5 2010-2019 downloads, then
# truth -> GEFS tracker cache -> lagged channels -> 20yr+lag model, 20yr no-lag, 8yr no-lag
# (isolating the lag and the extra-years effects) -> all scored on 2016-2019.
# Run from the project root:  nohup bash scripts/run_lps_20yr.sh > logs/lps_20yr_chain.log 2>&1 &
set -euo pipefail
PY=D:/sih-venv/Scripts/python.exe
export PYTHONUNBUFFERED=1
ERA5=data/raw/observations/era5/lps
stamp() { echo "=== $(date '+%F %T') $*"; }

stamp "waiting for ERA5 2010-2019 (pl + sl, 120 months each)"
until [ "$(ls $ERA5/era5_pl_201[0-9]*.nc 2>/dev/null | wc -l)" -ge 120 ] && \
      [ "$(ls $ERA5/era5_sl_201[0-9]*.nc 2>/dev/null | wc -l)" -ge 120 ]; do
  if grep -q Traceback logs/era5_lps_pl_2010_2019.log logs/era5_lps_sl_2010_2019.log; then
    stamp "ERA5 download failed - see logs/era5_lps_*_2010_2019.log"; exit 1
  fi
  sleep 120
done

stamp "ERA5 truth 2000-2019 (same thresholds as the 10yr truth)"
$PY -m scripts.build_lps_truth --zeta-min 4e-5 --deficit-min 2.0 --out data/processed/lps_truth_20yr.npz

stamp "GEFS tracker cache 2000-2019 from the rain store"
$PY -m scripts.build_lps_cache --store rain --years 2000-2019 --train-last 2015 \
    --truth data/processed/lps_truth_20yr.npz --cache-dir 'D:\sih-data\caches\lps_20yr' --workers 4

stamp "lagged-ensemble channels (runs from 1 and 2 days earlier)"
$PY -m scripts.build_lps_lag --cache-dir 'D:\sih-data\caches\lps_20yr'

stamp "train 20yr GNN + lagged ensemble (2000-2015, val 2016-2019)"
$PY -m scripts.train_lps --arch gnn --cache-dir 'D:\sih-data\caches\lps_20yr' --tag 20lag --lag

stamp "train 20yr GNN, no lag"
$PY -m scripts.train_lps --arch gnn --cache-dir 'D:\sih-data\caches\lps_20yr' --tag 20

stamp "train equal-features ablation (2000-2007, val 2008-2009)"
$PY -m scripts.train_lps --arch gnn --cache-dir 'D:\sih-data\caches\lps_20yr' --tag 20_8yr \
    --train-last 2007 --val-years 2008-2009

stamp "score all three on 2016-2019"
$PY -m scripts.eval_lps_window outputs/lps_gnn20lag.pt outputs/lps_gnn20.pt outputs/lps_gnn20_8yr.pt --years 2016-2019
stamp "done"
