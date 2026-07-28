#!/bin/bash
set -e
ROOT=/Users/yuanhangli/Documents/code/quant-strategy
PYTHON=$ROOT/.venv/bin/python3
EXPORT=$ROOT/scripts/export_execution_excel.py
FILES=(
  wf_daily_v25_w100_ts2024-01-02_te2024-02-07_cap10000.json
  wf_daily_v25_w100_ts2023-01-01_te2023-12-29_cap10000.json
  wf_daily_v25_w100_ts2023-01-01_te2024-09-23_cap10000.json
  wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json
)
for file in "${FILES[@]}"; do
  while [ ! -f "$ROOT/data/processed/$file" ]; do sleep 30; done
  $PYTHON $EXPORT "$ROOT/data/processed/$file"
done
