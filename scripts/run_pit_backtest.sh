#!/usr/bin/env bash
# PIT 无偏股票池上的 regime 空仓对比: off / breadth / any
# 训练集 training_data_pit_v24.parquet (519只), 每日只在当期生效的300只成分股中选股
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/pit_backtest
mkdir -p "$LOG"
PY=.venv/bin/python
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --hold-days 5 --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-24"

run() {
  $PY scripts/wf_v35_breadth_alpha.py $COMMON \
      --regime-filter "$1" --regime-ma 20 --regime-breadth 0.40 --regime-confirm 2 \
      --tag "pit_$1" > "$LOG/$1.log" 2>&1
  echo "done $1 rc=$?"
}

run off     & P1=$!
run breadth & P2=$!
wait $P1 $P2
run any
echo "ALL DONE"
