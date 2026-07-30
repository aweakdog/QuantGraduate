"""特征信息含量审计: 421 个候选特征里, 真正能预测截面收益的有几个?

动机: 标签是按日期 demean 的截面收益 (y = fwd_5d_ret - 当日均值)。
对这样的标签:

  【任何在同一天对所有股票取值相同的特征, 预测能力恒为 0】

而 feature_engine 里大量特征是市场级的(商品价格/汇率/国债收益率/PMI/
全球指数/全市场事件聚合) —— 它们每天只有一个值, 对截面排序毫无信息。
此外 feature_engine.py:1210-1221 给几乎每个数值列都自动生成 _ma5/_ma20,
造成三倍冗余。

本脚本量化三件事:
  1. 有多少特征是"市场级常数"(截面标准差≈0) -> 对截面标签是纯噪声
  2. 有多少特征是 _ma5/_ma20 派生 -> 与母特征高度共线
  3. 去掉这两类后, 真正独立的截面特征还剩几个
并给出每类特征的截面 IC, 用数据验证上述判断。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--train-file", default="training_data_pit_v24.parquet")
ap.add_argument("--pit-universe", default="universe_pit.parquet")
ap.add_argument("--cutoff", default="2023-09-19", help="只用此日期前的数据, 防泄漏")
args = ap.parse_args()

from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
LABEL_RAW, LABEL = "fwd_5d_ret", "y_target"
LEAKAGE = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP = {"date", "code", "group", LABEL, "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
        "fwd_21d_ret", "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret",
        "fwd_1d_exec_ret", "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret",
        "fwd_1d_exec_excess"}
EXCL = {"mf_pct_1d", "mf_pct_1d_ma5", "mf_pct_1d_ma20",
        "macd_signal", "macd_signal_ma5", "macd_signal_ma20"}

print("加载数据 ...")
df = pd.read_parquet(DATA_DIR / "processed" / args.train_file)
df["date"] = pd.to_datetime(df["date"])

u = pd.read_parquet(DATA_DIR / "universe" / args.pit_universe)
u["effective_date"] = pd.to_datetime(u["effective_date"])
u["code6"] = u["code"].astype(str).str.zfill(6)
eff = pd.DatetimeIndex(sorted(pd.to_datetime(u["effective_date"].unique())))
members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
code6 = df["code"].astype(str).str[:6]
period = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
keep = np.zeros(len(df), dtype=bool)
for i, d in enumerate(eff):
    m = period == i
    if m.any():
        keep[m] = code6[m].isin(members[pd.Timestamp(d)]).values
df = df[keep].reset_index(drop=True)

df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())
feats = [c for c in df.columns
         if c not in SKIP and c not in LEAKAGE and c not in EXCL
         and pd.api.types.is_numeric_dtype(df[c])]

s = df[(df["date"] < pd.Timestamp(args.cutoff)) & df[LABEL].notna()].copy()
print(f"  {len(df):,} 行, {len(feats)} 个候选特征")
print(f"  审计样本: {len(s):,} 行, {s['date'].nunique()} 个交易日 (< {args.cutoff})\n")

# ── 1. 截面变异性: 每个特征在同一天内跨股票的标准差 ──
print("=" * 84)
print("一、截面变异性 —— 哪些特征在同一天对所有股票是同一个值?")
print("=" * 84)

# 每天算截面std, 再除以该特征的总体std做归一化, 对全部日期取均值
overall_std = s[feats].std().replace(0, np.nan)
cs_std = s.groupby("date")[feats].std().mean()
ratio = (cs_std / overall_std).fillna(0)

MARKET_WIDE = ratio[ratio < 0.01].index.tolist()   # 截面几乎无变异
CROSS_SEC = ratio[ratio >= 0.01].index.tolist()

print(f"  市场级常数特征 (截面std/总体std < 1%): {len(MARKET_WIDE)} 个")
print(f"  真正有截面变异的特征              : {len(CROSS_SEC)} 个")
print(f"\n  市场级特征示例(前25个):")
for c in MARKET_WIDE[:25]:
    print(f"    {c:<40} 截面变异比 {ratio[c]:.5f}")
if len(MARKET_WIDE) > 25:
    print(f"    ... 另外 {len(MARKET_WIDE)-25} 个")

# ── 2. MA 派生冗余 ──
print()
print("=" * 84)
print("二、_ma5/_ma20 自动派生造成的冗余")
print("=" * 84)
ma_derived = [c for c in feats if c.endswith("_ma5") or c.endswith("_ma20")]
base_feats = [c for c in feats if not (c.endswith("_ma5") or c.endswith("_ma20"))]
print(f"  母特征           : {len(base_feats)} 个")
print(f"  _ma5/_ma20 派生  : {len(ma_derived)} 个  ({len(ma_derived)/len(feats):.0%} 的候选池)")

# 派生特征与母特征的相关性
corrs = []
for c in ma_derived:
    base = c.rsplit("_ma", 1)[0]
    if base in s.columns:
        v = s[[base, c]].dropna()
        if len(v) > 1000:
            r = abs(v[base].corr(v[c]))
            if not np.isnan(r):
                corrs.append(r)
if corrs:
    corrs = np.array(corrs)
    print(f"  派生 vs 母特征 |相关系数| : 中位数 {np.median(corrs):.3f}, "
          f"均值 {corrs.mean():.3f}")
    print(f"    |r|>0.9 的占比 {(corrs>0.9).mean():.0%} | "
          f"|r|>0.95 的占比 {(corrs>0.95).mean():.0%}")

# ── 3. 分类做截面 IC ──
print()
print("=" * 84)
print("三、各类特征的截面 IC (用数据验证: 市场级特征是否真的没用)")
print("=" * 84)


def cross_sectional_ic(cols, label=LABEL, min_n=20):
    """每天算 spearman(feature, label) 的截面相关, 再对日期平均"""
    out = {}
    g = s.groupby("date")
    for c in cols:
        ics = []
        for _, grp in g:
            v = grp[[c, label]].dropna()
            if len(v) < min_n or v[c].nunique() < 3:
                continue
            r = v[c].rank().corr(v[label].rank())
            if not np.isnan(r):
                ics.append(r)
        if len(ics) >= 30:
            a = np.array(ics)
            out[c] = (a.mean(), a.mean() / a.std() * np.sqrt(len(a)) if a.std() else 0,
                      len(a))
    return out


def summarize(name, cols):
    if not cols:
        print(f"  {name:<26} (无)")
        return None
    res = cross_sectional_ic(cols)
    if not res:
        print(f"  {name:<26} 有效样本不足")
        return None
    tab = pd.DataFrame(res, index=["ic", "t", "n"]).T
    strong = (tab["t"].abs() > 2).sum()
    print(f"  {name:<26} {len(tab):>4} 个 | |IC|中位数 {tab['ic'].abs().median():.4f} "
          f"| |t|中位数 {tab['t'].abs().median():>5.2f} | |t|>2 的有 {strong:>3} 个 "
          f"({strong/len(tab):.0%})")
    return tab


print(f"{'类别':<26} {'数量':>6} {'|IC|中位数':>12} {'|t|中位数':>10} {'|t|>2 占比':>14}")
print("-" * 84)
tab_mkt = summarize("市场级常数特征", MARKET_WIDE[:80])
tab_cs = summarize("有截面变异的特征", CROSS_SEC[:120])
tab_base = summarize("母特征(非MA派生)", [c for c in base_feats if c in CROSS_SEC][:120])
tab_ma = summarize("MA派生特征", [c for c in ma_derived if c in CROSS_SEC][:120])

# ── 4. 有效独立信息量估计 ──
print()
print("=" * 84)
print("四、有效独立信息量 (对截面标签真正可用的特征数)")
print("=" * 84)
useful = [c for c in CROSS_SEC if not (c.endswith("_ma5") or c.endswith("_ma20"))]
print(f"  原始候选池                     : {len(feats)} 个")
print(f"  剔除市场级常数 (-{len(MARKET_WIDE)})           : {len(CROSS_SEC)} 个")
print(f"  再剔除 MA 派生 (-{len(CROSS_SEC)-len(useful)})           : {len(useful)} 个")

if len(useful) > 2:
    sub = s[useful].dropna(axis=1, how="all")
    cm = sub.corr().abs()
    np.fill_diagonal(cm.values, 0)
    # 贪心去相关, 阈值 0.7
    order = sub.std().sort_values(ascending=False).index.tolist()
    sel = []
    for c in order:
        if c not in cm.index:
            continue
        if all(cm.at[c, g] <= 0.7 for g in sel):
            sel.append(c)
    print(f"  再按 |r|>0.7 去相关            : {len(sel)} 个  <- 真正独立的信息维度")

    # 有效样本量对比
    n_days = s["date"].nunique()
    n_blocks = n_days // 5
    print(f"\n  独立样本区块 (交易日/5日重叠标签): 约 {n_blocks} 个")
    print(f"  独立信息维度 / 独立样本区块      : {len(sel)}/{n_blocks} = "
          f"{len(sel)/n_blocks:.2f}")
    if len(sel) / n_blocks > 0.3:
        print("  !! 比值过高: 参数维度接近样本量, 过拟合几乎必然发生")

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("  若'市场级常数特征'这一行的 |t|>2 占比与'有截面变异的特征'相当,")
print("  说明它们是通过与其他特征交互间接起作用(或纯属多重检验假阳性);")
print("  若明显更低, 则证实它们对截面标签是纯噪声, 应当从候选池剔除 ——")
print("  这将直接减少特征筛选阶段的多重检验负担。")
