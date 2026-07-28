#!/usr/bin/env bash
# 用【无泄漏】预测缓存重扫空仓择时参数网格 (广度阈值 x 确认天数)
# 之前的网格结论建立在泄漏特征集上, 需要重新验证。
# 固定 h10 + 无护栏 (修复后该组合最优)。
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/clean_grid
mkdir -p "$LOG"
PY=".venv/bin/python -u"
CACHE=preds_clean_5d.pkl
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --hold-days 10 --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-27 \
        --reversal-guard 0.0 --load-preds $CACHE"

for B in 0.30 0.35 0.40 0.45 0.50; do
  PIDS=()
  for C in 1 2 3 4; do
    TAG="cgrid_B${B/0./}C${C}"
    $PY scripts/wf_v35_breadth_alpha.py $COMMON \
        --regime-filter breadth --regime-ma 20 \
        --regime-breadth "$B" --regime-confirm "$C" \
        --tag "$TAG" > "$LOG/B${B/0./}C${C}.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  echo "  阈值 $B 完成"
done

# 对照: 完全关闭空仓择时
$PY scripts/wf_v35_breadth_alpha.py $COMMON --regime-filter off \
    --tag cgrid_off > "$LOG/off.log" 2>&1
echo "ALL DONE"
