#!/bin/bash
# 2019 扩窗·生产血统矩阵 (TODO#4, 见 docs/findings_2026-09-05_expand_2019_prodlineage.md)
#   与生产同一条链 (pipeline.feature_engine -> build_tick_augmented --lag 1), 起点 2019-01-01, 池 PIT 2019 (630 只)
#   产物: data/processed/training_data_pit_v24_2019.parquet (464 列) -> training_data_pit_v24_2019_tick1.parquet (522 列)
#   不碰生产任何文件: 独立 --features-dir features_2019, 独立逐笔面板目录 tick_micro_2019r
#   2022-09-01 起的每行每列与生产矩阵逐值相等 (09-05 对账 72/72 列 100%), 即生产矩阵的严格超集
#   在 041 跑 (逐笔面板洞由 040 的 tick_micro_2015u 补); 全程 ~10 min
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 POLARS_MAX_THREADS=4
PY=./.venv/bin/python
PROC=data/processed
TICK_SRC=${TICK_SRC:-eez040.ece.ust.hk:quant-strategy/data/processed/tick_micro_2015u}

echo "== A. 逐笔面板 tick_micro_2019r: 生产面板硬链接 + 合并面板补洞 $(date +%H:%M)"
mkdir -p $PROC/tick_micro_2019r
for f in $PROC/tick_micro/*.parquet; do ln -f "$f" "$PROC/tick_micro_2019r/$(basename "$f")" 2>/dev/null || cp "$f" "$PROC/tick_micro_2019r/"; done
rsync -q --ignore-existing "$TICK_SRC"/2019*.parquet "$TICK_SRC"/2020*.parquet $PROC/tick_micro_2019r/ || echo "  (补洞 rsync 失败, 沿用生产面板)"
echo "   面板 $(ls $PROC/tick_micro_2019r | wc -l) 天: $(ls $PROC/tick_micro_2019r | head -1) ~ $(ls $PROC/tick_micro_2019r | tail -1)"

echo "== B. feature_engine 全量重建 cutoff 2019-01-01 $(date +%H:%M)"
nice -n 10 $PY -u -m pipeline.feature_engine --no-incremental --procs 8 \
  --watchlist watchlist_pit_2019.json --out training_data_pit_v24_2019.parquet \
  --cutoff 2019-01-01 --features-dir features_2019 2>&1 | grep -v "Warning\|result\[\|result.loc" | tail -3

echo "== C. 逐笔增广 lag1 $(date +%H:%M)"
TICK_MICRO_DIR=$PWD/$PROC/tick_micro_2019r nice -n 10 $PY -u scripts/build_tick_augmented.py \
  --source training_data_pit_v24_2019.parquet --lag 1 2>&1 | tail -6
echo "DONE $(date +%H:%M)  (T1A/T1B 未增广: t1a 面板只覆盖 519 只, 630 池末日覆盖率 84% 过不了 90% 闸; 需要时 build_t1_augmented --only t1b)"
