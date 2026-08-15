# -*- coding: utf-8 -*-
"""把逐笔微观特征并进训练矩阵, 产出增广矩阵 (实验用, 不动主链)

主链 build_v23 -> build_training_v24_exec 保持原样, 本脚本只在其产物之上加列,
输出独立文件, 这样实验失败可以直接删文件, 不污染线上矩阵。

三个必须做对的处理
──────────────────
1. **按交易所分组做横截面标准化**。沪深撮合粒度不同, 按成交金额分档的 trd_net_* 两所
   中位数差一个量级(实测)。不分组标准化, 模型会把这几列当"交易所哑变量"用 ——
   不是泄漏(交易所归属事前已知且恒定), 但是纯噪声因子。分组 z 化后两所可比。
2. **只保留横截面相对位置, 不保留原始水平**。选股模型要的是"这只票在今天全池里排第几",
   绝对水平反而携带交易所与时间趋势的干扰。
3. **PIT 滞后可配**。逐笔 d 日数据在 d 日收盘后才完整, 而信号在 d 日收盘后生成、
   d+1 开盘买入。同日(lag=0)在原理上可用, 但要求供应商当晚就交付;
   lag=1 是不依赖交付时效的保守口径。两版都建, 用差值量化"当日信息值多少",
   若同日显著更好, 说明结论依赖线上未必拿得到的数据。

用法
────
    python scripts/build_tick_augmented.py --lag 1
    python scripts/build_tick_augmented.py --lag 0 --source training_data_pit_v24.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MICRO = PROC / "tick_micro"

# 逐笔抽取器产出的列里, 这些是纯价格水平, 矩阵已有等价物, 不再重复引入
DROP = {"code", "date", "close", "vwap", "prev_close", "day_amt"}
WIN = 3.0        # 横截面 z 的截尾
MA = 5           # 平滑窗口, 与 feature_engine 的 min_periods=window 惯例一致


def load_micro():
    fs = sorted(MICRO.glob("*.parquet"))
    if not fs:
        raise SystemExit(f"{MICRO} 是空的, 先跑 scripts/tick_micro_features.py")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["c6"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values(["c6", "date"]).drop_duplicates(["c6", "date"], keep="last")
    return df, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24.parquet")
    ap.add_argument("--lag", type=int, default=1,
                    help="1=用 d-1 的逐笔(不依赖交付时效); 0=用 d 当日")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    out = PROC / (a.output or a.source.replace(".parquet", f"_tick{a.lag}.parquet"))

    micro, nday = load_micro()
    feats = [c for c in micro.columns if c not in DROP and c != "c6"]
    print(f"逐笔面板: {len(micro):,} 股票日  {nday} 个交易日  "
          f"{micro.date.min().date()}~{micro.date.max().date()}  "
          f"股票 {micro.c6.nunique()} 只  原始特征 {len(feats)} 个")

    # ---------- 1. 按交易所分组做横截面 z ----------
    micro["exch"] = np.where(micro["c6"].str.startswith("6"), "SH", "SZ")
    zs = {}
    for c in feats:
        x = pd.to_numeric(micro[c], errors="coerce")
        mu = x.groupby([micro["date"], micro["exch"]]).transform("mean")
        sd = x.groupby([micro["date"], micro["exch"]]).transform("std")
        zs[f"tk_{c}_xz"] = ((x - mu) / sd.replace(0, np.nan)).clip(-WIN, WIN)
    z = pd.DataFrame(zs, index=micro.index)
    z["c6"], z["date"] = micro["c6"], micro["date"]

    # ---------- 2. 平滑 + 滞后 ----------
    z = z.sort_values(["c6", "date"])
    gz = z.groupby("c6", sort=False)
    for c in [f"tk_{f}_xz" for f in feats]:
        z[c + f"_ma{MA}"] = gz[c].transform(
            lambda s: s.rolling(MA, min_periods=MA).mean())
    tk_cols = [c for c in z.columns if c.startswith("tk_")]
    if a.lag > 0:
        # 在逐笔面板自己的交易日序列上后移, 等价于"取最近一个已知的逐笔观测"
        z[tk_cols] = z.groupby("c6", sort=False)[tk_cols].shift(a.lag)
    print(f"逐笔派生列: {len(tk_cols)} 个 (xz {len(feats)} + xz_ma{MA} {len(feats)}), "
          f"滞后 {a.lag} 天")

    # ---------- 3. 并进矩阵 ----------
    base = pd.read_parquet(PROC / a.source)
    base["date"] = pd.to_datetime(base["date"])
    base["c6"] = base["code"].astype(str).str.extract(r"(\d{6})")[0]
    dup = [c for c in tk_cols if c in base.columns]
    if dup:
        base = base.drop(columns=dup)
    n0 = len(base)
    m = base.merge(z[["c6", "date"] + tk_cols], on=["c6", "date"], how="left",
                   validate="one_to_one")
    assert len(m) == n0, f"合并后行数变了 {n0} -> {len(m)}"
    m = m.drop(columns=["c6"])

    cov = m[tk_cols].notna().mean()
    inwin = m["date"].between(micro.date.min(), micro.date.max())
    cov_in = m.loc[inwin, tk_cols].notna().mean()
    print(f"\n矩阵 {a.source}: {n0:,} 行 x {base.shape[1] - 1} 列 -> "
          f"{len(m):,} 行 x {m.shape[1]} 列")
    print(f"逐笔列非空率: 全表 中位 {cov.median():.1%}  "
          f"逐笔覆盖期内 中位 {cov_in.median():.1%} "
          f"(最低 {cov_in.min():.1%} 最高 {cov_in.max():.1%})")
    miss_day = m.loc[inwin].groupby("date")[tk_cols[0]].apply(
        lambda s: s.isna().all())
    if miss_day.any():
        bad = miss_day[miss_day].index.strftime("%Y%m%d").tolist()
        print(f"⚠ 覆盖期内有 {len(bad)} 个交易日整日无逐笔: {bad[:8]}"
              f"{' ...' if len(bad) > 8 else ''}")
    else:
        print("覆盖期内无整日缺失 ✓")

    tmp = out.with_suffix(".tmp.parquet")
    m.to_parquet(tmp, index=False)
    tmp.replace(out)
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
