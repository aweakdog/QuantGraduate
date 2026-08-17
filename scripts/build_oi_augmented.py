# -*- coding: utf-8 -*-
"""隔夜/日内收益分解特征增广 (实验层, 不动主链矩阵)

动机 (findings 2026-08-16 §12.8): tick 级执行研究发现买入日"高开阴跌"、
卖出日"日内阳升"的稳定结构 —— 隔夜与日内收益是两个独立信息源。
作为执行择时奖金太小 (~2pp/年), 但作为截面特征作用于全池排序, 值得一测。

特征 (全部只用日线 open/close, 信号日收盘即可得, 无 PIT 滞后问题):
  oi_on_5/20    : 近5/20日 隔夜收益均值 (open_t/close_{t-1}-1)
  oi_id_5/20    : 近5/20日 日内收益均值 (close_t/open_t-1)
  oi_gap_20     : 日内-隔夜 之差 (20日)
  oi_fade_20    : 高开回吐频率 = mean(1[隔夜>0.5% 且 日内<0])
全部按日横截面 z (截尾±3), 只保留相对位置, 与 tick 增广同一惯例。

⚠ K线为未复权价: 相邻日比值在除权日会跳变, 已按 ±15% 截断日收益兜底,
   对 5/20 日均值的污染 ≤0.75%, 且横截面 z 后进一步稀释 —— 可接受。

用法:
    python scripts/build_oi_augmented.py \
        --source training_data_pit_v24_tick1.parquet \
        --output training_data_pit_v24_tick1_oi.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
KL = ROOT / "data" / "raw" / "kline"
WIN = 3.0
CLIP = 0.15


def stock_oi(code6: str) -> pd.DataFrame | None:
    p = KL / f"{code6}.parquet"
    if not p.exists():
        return None
    k = pd.read_parquet(p, columns=["时间", "开盘价", "收盘价"]).rename(
        columns={"时间": "date", "开盘价": "open", "收盘价": "close"})
    k["date"] = pd.to_datetime(k["date"]).dt.strftime("%Y-%m-%d")
    k = k.sort_values("date")
    o = pd.to_numeric(k["open"], errors="coerce")
    c = pd.to_numeric(k["close"], errors="coerce")
    on = (o / c.shift(1) - 1).clip(-CLIP, CLIP)
    id_ = (c / o - 1).clip(-CLIP, CLIP)
    out = pd.DataFrame({"code": code6, "date": k["date"].values})
    out["oi_on_5"] = on.rolling(5, min_periods=5).mean().values
    out["oi_on_20"] = on.rolling(20, min_periods=20).mean().values
    out["oi_id_5"] = id_.rolling(5, min_periods=5).mean().values
    out["oi_id_20"] = id_.rolling(20, min_periods=20).mean().values
    out["oi_gap_20"] = out["oi_id_20"] - out["oi_on_20"]
    fade = ((on > 0.005) & (id_ < 0)).astype(float)
    out["oi_fade_20"] = fade.rolling(20, min_periods=20).mean().values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24_tick1.parquet")
    ap.add_argument("--output", default="training_data_pit_v24_tick1_oi.parquet")
    a = ap.parse_args()

    mx = pd.read_parquet(PROC / a.source)
    date_col = "date" if "date" in mx.columns else "时间"
    mx["_c6"] = mx["code"].astype(str).str.extract(r"(\d{6})")[0]
    mx["_d"] = pd.to_datetime(mx[date_col]).dt.strftime("%Y-%m-%d")
    codes = sorted(mx["_c6"].dropna().unique())
    print(f"矩阵 {len(mx)} 行, {len(codes)} 只股票")

    parts = [df for c in codes if (df := stock_oi(c)) is not None]
    oi = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in oi.columns if c.startswith("oi_")]

    # 按日横截面 z + 截尾 (只在矩阵覆盖的日期上算, 用全池分布)
    oi = oi.merge(mx[["_c6", "_d"]].drop_duplicates().rename(
        columns={"_c6": "code", "_d": "date"}), on=["code", "date"], how="inner")
    for c in feat_cols:
        g = oi.groupby("date")[c]
        z = (oi[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        oi[c + "_xz"] = z.clip(-WIN, WIN)
    keep = ["code", "date"] + [c + "_xz" for c in feat_cols]

    merged = mx.merge(oi[keep].rename(columns={"code": "_c6", "date": "_d"}),
                      on=["_c6", "_d"], how="left").drop(columns=["_c6", "_d"])
    new_cols = [c for c in merged.columns if c.startswith("oi_")]
    cov = merged[new_cols[0]].notna().mean()
    print(f"新列 {len(new_cols)} 个, 覆盖率 {cov:.1%}")
    merged.to_parquet(PROC / a.output, index=False)
    print(f"已保存 {PROC / a.output}  ({len(merged)} 行 x {len(merged.columns)} 列)")


if __name__ == "__main__":
    main()
