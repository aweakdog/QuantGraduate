#!/usr/bin/env bash
# 对症实验: 针对"急涨行情追高被打脸"的三种修法, 基线 = pit_breadth
#   g10  : 反转护栏, 剔除近5日涨幅前10%
#   g20  : 反转护栏, 剔除近5日涨幅前20%
#   h10  : 换仓周期 5天 -> 10天 (降频, 不加护栏)
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/pit_backtest
mkdir -p "$LOG"
PY=".venv/bin/python -u"   # -u: 关闭stdout缓冲, 进度实时写进日志
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-24 \
        --regime-filter breadth --regime-ma 20 --regime-breadth 0.40 --regime-confirm 2"

$PY scripts/wf_v35_breadth_alpha.py $COMMON --hold-days 5  --reversal-guard 0.10 \
    --tag pit_g10 > "$LOG/g10.log" 2>&1 &
P1=$!
$PY scripts/wf_v35_breadth_alpha.py $COMMON --hold-days 5  --reversal-guard 0.20 \
    --tag pit_g20 > "$LOG/g20.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "g10/g20 done"

$PY scripts/wf_v35_breadth_alpha.py $COMMON --hold-days 10 --reversal-guard 0.0 \
    --tag pit_h10 > "$LOG/h10.log" 2>&1
echo "ALL DONE"
