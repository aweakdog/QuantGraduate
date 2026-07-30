"""分层单调性检验: 模型的排序能力是真的吗?

比总收益可靠得多的判据。把每日预测按分位切成 10 层, 看各层的未来收益:

  - 真 alpha  -> 各层收益单调递减, 且 多空价差(第1层-第10层) 显著为正
  - 假 alpha  -> 只有第1层(或某几层)突出, 其余杂乱无章
                 说明收益来自少数极端样本, 不是系统性排序能力

同时输出多空价差的逐年表现, 检验时间稳定性。用缓存预测, 不重训。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="preds_P1BASE_oldfeats.pkl")
ap.add_argument("--hold-days", type=int, default=10)
ap.add_argument("--n-groups", type=int, default=10)
args = ap.parse_args()

from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
KLINE_DIR = DATA_DIR / "raw" / "kline"

print("加载缓存预测与K线 ...")
cache = pickle.load(open(DATA_DIR / "processed" / args.preds, "rb"))
preds = cache["preds"]

frames = []
for p in sorted(KLINE_DIR.glob("*.parquet")):
    kl = pd.read_parquet(p, columns=["date", "close"])
    kl["date"] = pd.to_datetime(kl["date"])
    kl["code6"] = p.stem
    frames.append(kl)
px = (pd.concat(frames, ignore_index=True)
      .drop_duplicates(["code6", "date"])
      .pivot(index="date", columns="code6", values="close").sort_index())
all_dates = px.index
H = args.hold_days
G = args.n_groups

rows = []
for dp in preds:
    d = pd.Timestamp(dp["date"])
    i = all_dates.searchsorted(d)
    if i + 1 + H >= len(all_dates):
        continue
    d_buy, d_sell = all_dates[i + 1], all_dates[i + 1 + H]
    ranked = [str(c)[:6] for c in dp["ranked"]]
    ranked = [c for c in ranked if c in px.columns]
    if len(ranked) < G * 5:
        continue
    pb, ps = px.loc[d_buy, ranked], px.loc[d_sell, ranked]
    ok = pb.notna() & ps.notna() & (pb > 0)
    if ok.sum() < G * 5:
        continue
    codes = np.array(ranked)[ok.values]
    fwd = (ps[ok].values / pb[ok].values - 1)
    n = len(codes)
    # ranked 已按预测降序; 第0组 = 模型最看好
    grp = (np.arange(n) * G // n)
    rows.append(pd.DataFrame({"date": d, "grp": grp, "fwd": fwd}))

if not rows:
    raise SystemExit("无可用样本")
panel = pd.concat(rows, ignore_index=True)
n_days = panel["date"].nunique()
print(f"  {n_days} 个信号日\n")

daily = panel.groupby(["date", "grp"])["fwd"].mean().unstack()

print("=" * 78)
print(f"分层收益 (第1层=模型最看好, 持有 {H} 天, 共 {n_days} 期)")
print("=" * 78)
print(f"{'层':>4} {'平均收益':>10} {'年化':>9} {'t值':>7} {'胜率':>8}")
print("-" * 78)
per_year = 252 / H
means = []
for g in range(G):
    if g not in daily.columns:
        continue
    v = daily[g].dropna()
    m = v.mean()
    t = m / v.std() * np.sqrt(len(v))
    means.append(m)
    print(f"{g+1:>4} {m:>+9.3%} {m*per_year:>+8.1%} {t:>7.2f} {(v>0).mean():>7.1%}")

# 单调性: 分层收益与层序号的秩相关
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(np.arange(len(means)), means).statistic
print("-" * 78)
print(f"单调性(层序号 vs 收益 的秩相关): {rho:+.3f}  "
      f"(理想 = -1.000, 即层数越大收益越低)")

ls = (daily[0] - daily[G - 1]).dropna()
t_ls = ls.mean() / ls.std() * np.sqrt(len(ls))
print(f"多空价差(第1层 - 第{G}层): {ls.mean():+.3%}/期, 年化 {ls.mean()*per_year:+.1%}, "
      f"t={t_ls:.2f}, 胜率 {(ls>0).mean():.1%}")

print()
print("=" * 78)
print("多空价差逐年")
print("=" * 78)
print(f"{'年份':>6} {'期数':>6} {'平均价差':>11} {'年化':>9} {'t值':>7} {'胜率':>8}")
print("-" * 78)
for y, v in ls.groupby(ls.index.year):
    if len(v) < 5:
        continue
    t = v.mean() / v.std() * np.sqrt(len(v)) if v.std() else np.nan
    print(f"{y:>6} {len(v):>6} {v.mean():>+10.3%} {v.mean()*per_year:>+8.1%} "
          f"{t:>7.2f} {(v>0).mean():>7.1%}")

print()
print("=" * 78)
print("头部集中度: 收益是否只来自第1层?")
print("=" * 78)
top = daily[0].dropna()
rest = daily[[c for c in daily.columns if c != 0]].mean(axis=1).dropna()
print(f"  第1层        : {top.mean():+.3%}/期")
print(f"  第2~{G}层平均 : {rest.mean():+.3%}/期")
excess = (daily[0] - daily[[c for c in daily.columns if c != 0]].mean(axis=1)).dropna()
t_ex = excess.mean() / excess.std() * np.sqrt(len(excess))
print(f"  第1层超额    : {excess.mean():+.3%}/期, t={t_ex:.2f}")
print()
print("  判读: 单调性秩相关接近 -1 且多空 t>2 -> 排序能力真实;")
print("        若只有第1层突出而中间层杂乱 -> 收益靠少数极端样本, 不稳健。")
