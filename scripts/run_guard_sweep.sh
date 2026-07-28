#!/usr/bin/env bash
# 反转护栏敏感性扫描: 0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.30
# 复用 preds_pit_5d.pkl 缓存, 每组只需 1-2 分钟
# 目的: 确认 g10 不是运气, 而是处在平滑高原上
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/guard_sweep
mkdir -p "$LOG"
PY=".venv/bin/python -u"   # -u: 关闭stdout缓冲, 进度实时写进日志
CACHE=preds_pit_5d.pkl
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --hold-days 5 --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-24 \
        --regime-filter breadth --regime-breadth 0.40 --regime-confirm 2 \
        --load-preds $CACHE"

PIDS=()
for G in 0.00 0.05 0.10 0.15 0.20 0.30; do
    TAG="pit_guard${G/0./}pct"
    $PY scripts/wf_v35_breadth_alpha.py $COMMON --reversal-guard "$G" \
        --tag "$TAG" > "$LOG/guard${G/0./}pct.log" 2>&1 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
echo "ALL DONE"
