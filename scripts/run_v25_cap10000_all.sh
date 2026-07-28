#!/bin/bash
set -e
export PYTHONPATH=/Users/yuanhangli/Documents/code/quant-strategy
export QUANT_DATA_DIR=/Users/yuanhangli/Documents/code/quant-strategy/data
PYTHON=/Users/yuanhangli/Documents/code/quant-strategy/.venv/bin/python3
SCRIPT=/Users/yuanhangli/Documents/code/quant-strategy/scripts/wf_daily_expanding.py
DATA=training_data_v25.parquet
COMMON="--train-data ${DATA} --window 100 --initial-capital 10000"

$PYTHON $SCRIPT $COMMON --test-start 2024-01-02 --test-end 2024-02-07
$PYTHON $SCRIPT $COMMON --test-start 2023-01-01 --test-end 2023-12-29
$PYTHON $SCRIPT $COMMON --test-start 2023-01-01 --test-end 2024-09-23
$PYTHON $SCRIPT $COMMON --test-start 2023-01-01 --test-end 2026-07-16
