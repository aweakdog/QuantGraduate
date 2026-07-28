#!/usr/bin/env bash
# regime 空仓过滤器对比: off / ma / breadth / both / any
# 其余参数固定: 10万本金, periodic 每5天换仓持3只, T+1开盘成交
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/regime_sweep
mkdir -p "$LOG"
PY=.venv/bin/python
COMMON="--initial-capital 100000 --portfolio-mode periodic --hold-days 5 --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-24"

run() {  # $1=filter
  $PY scripts/wf_v35_breadth_alpha.py $COMMON \
      --regime-filter "$1" --regime-ma 20 --regime-breadth 0.40 --regime-confirm 2 \
      --tag "regime_$1" > "$LOG/$1.log" 2>&1
  echo "done $1 rc=$?"
}

run off  & P1=$!
run both & P2=$!
wait $P1 $P2

run ma  & P3=$!
run any & P4=$!
wait $P3 $P4

run breadth
echo "ALL DONE"
