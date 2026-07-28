#!/usr/bin/env bash
# 空仓择时参数二维扫描: 广度阈值 x 确认天数
#
# 关键点: 阈值/确认天数只影响【执行层】, 不影响模型训练。
# 所以先训练一次并缓存逐日预测, 之后 20 个组合直接复用缓存, 每个只需 1-2 分钟。
#
# 目的: 判断 (0.40, 确认2天) 是"平滑高原"里的一点, 还是孤立的过拟合尖峰。
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/regime_grid
mkdir -p "$LOG"
PY=".venv/bin/python -u"   # -u: 关闭stdout缓冲, 进度实时写进日志
CACHE=preds_pit_5d.pkl
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --hold-days 5 --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-24"

# ── 阶段1: 生成预测缓存 (只训练一次) ──
if [ ! -f "data/processed/$CACHE" ]; then
  echo "[1/2] 训练并缓存预测 ..."
  $PY scripts/wf_v35_breadth_alpha.py $COMMON \
      --regime-filter breadth --regime-breadth 0.40 --regime-confirm 2 \
      --save-preds "$CACHE" --tag pit_cache_base > "$LOG/_cache.log" 2>&1
  echo "  缓存完成 rc=$?"
else
  echo "[1/2] 复用已有缓存 data/processed/$CACHE"
fi

# ── 阶段2: 网格replay ──
echo "[2/2] 网格扫描 (5 阈值 x 4 确认天数) ..."
PIDS=()
for B in 0.30 0.35 0.40 0.45 0.50; do
  for C in 1 2 3 4; do
    TAG="pit_gridB${B/0./}C${C}"
    $PY scripts/wf_v35_breadth_alpha.py $COMMON \
        --regime-filter breadth --regime-breadth "$B" --regime-confirm "$C" \
        --load-preds "$CACHE" --tag "$TAG" > "$LOG/B${B/0./}C${C}.log" 2>&1 &
    PIDS+=($!)
    # 每 5 个一批, 避免内存/IO 打满
    if [ ${#PIDS[@]} -ge 5 ]; then
      wait "${PIDS[@]}"
      PIDS=()
      echo "  一批完成"
    fi
  done
done
[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
echo "ALL DONE"
