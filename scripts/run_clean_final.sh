#!/usr/bin/env bash
# 修掉特征筛选未来泄漏后的最终对照实验
#
# 泄漏根因: select_features 原先在 s=df[date<TEST_START] 为空时静默退化成
#           s=df, 即用含测试期标签的全样本筛特征。PIT 数据恰好从 TEST_START
#           开始, 所以一直在泄漏。现改为用【首个可出信号日】之前的数据。
#
# 阶段1: 训练一次, 缓存逐日预测 (特征集已无泄漏)
# 阶段2: 复用缓存扫执行层参数 (换仓周期 x 反转护栏)
set -u
cd "$(dirname "$0")/.."
LOG=data/processed/clean_final
mkdir -p "$LOG"
PY=".venv/bin/python -u"   # -u: 关闭stdout缓冲, 进度实时写进日志
CACHE=preds_clean_5d.pkl
COMMON="--train-file training_data_pit_v24.parquet --pit-universe universe_pit.parquet \
        --initial-capital 100000 --portfolio-mode periodic --tranche-n 3 \
        --exec-mode t1open --test-start 2022-09-01 --test-end 2026-07-27 \
        --regime-filter breadth --regime-ma 20 --regime-breadth 0.40 --regime-confirm 2"

# ── 阶段1: 生成干净预测缓存 ──
if [ ! -f "data/processed/$CACHE" ]; then
  echo "[1/2] 训练并缓存预测 (无泄漏特征集) ..."
  $PY scripts/wf_v35_breadth_alpha.py $COMMON \
      --hold-days 5 --reversal-guard 0.0 \
      --save-preds "$CACHE" --tag clean_h5g0 > "$LOG/train.log" 2>&1
  rc=$?
  echo "  缓存完成 rc=$rc"
  [ $rc -ne 0 ] && { tail -20 "$LOG/train.log"; exit 1; }
else
  echo "[1/2] 缓存已存在, 跳过训练"
fi

# ── 阶段2: 执行层扫描 ──
echo "[2/2] 执行层扫描 (换仓周期 x 反转护栏) ..."
PIDS=()
for H in 5 10; do
  for G in 0.00 0.10 0.15; do
    TAG="clean_h${H}g${G/0./}"
    $PY scripts/wf_v35_breadth_alpha.py $COMMON \
        --hold-days "$H" --reversal-guard "$G" \
        --load-preds "$CACHE" --tag "$TAG" > "$LOG/h${H}g${G/0./}.log" 2>&1 &
    PIDS+=($!)
  done
done
wait "${PIDS[@]}"
echo "ALL DONE"
