"""小资金买得起的低价股池里, 模型还有没有 alpha?

背景: pos=3/regime=off 的对照显示, 模型盈利几乎全部来自买入价 >¥70 的股票
(10万本金在该区间 76 个回合赚 ¥129,051), 而 ¥20,000 本金按 3 只持仓,
每只预算 ¥6,667, 受 A股最小 100 股限制只能买 ¥66.7 以下的票, 完全够不到。

本脚本直接回答: 把股票池按"某本金买不买得起"切开后, 各子池的 IC 和
前3名实际收益分别是多少。如果低价子池 IC≈0, 那么 ¥20,000 这个本金
在当前策略下【结构上就不可行】, 再怎么调参也没用。

用缓存预测, 不重训模型。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="preds_P1BASE_oldfeats.pkl")
ap.add_argument("--hold-days", type=int, default=10)
ap.add_argument("--positions", type=int, default=3)
ap.add_argument("--capitals", default="20000,50000,100000,300000")
args = ap.parse_args()

from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
KLINE_DIR = DATA_DIR / "raw" / "kline"
COL = {"date": "date", "close": "close"}

print("加载缓存预测 ...")
cache = pickle.load(open(DATA_DIR / "processed" / args.preds, "rb"))
preds = cache["preds"]
print(f"  {len(preds)} 个预测日")

print("加载K线, 构建 日期x代码 收盘价矩阵 ...")
frames = []
for p in sorted(KLINE_DIR.glob("*.parquet")):
    kl = pd.read_parquet(p, columns=list(COL))[list(COL)].rename(columns=COL)
    kl["date"] = pd.to_datetime(kl["date"])
    kl["code6"] = p.stem
    frames.append(kl)
px = pd.concat(frames, ignore_index=True)
px = px.drop_duplicates(["code6", "date"]).pivot(index="date", columns="code6",
                                                 values="close").sort_index()
print(f"  {px.shape[0]} 个交易日 x {px.shape[1]} 只股票")

all_dates = px.index
H = args.hold_days
CAPS = [float(x) for x in args.capitals.split(",")]

rows = []
for dp in preds:
    d = pd.Timestamp(dp["date"])
    i = all_dates.searchsorted(d)
    # T日收盘出信号 -> T+1 收盘买入 -> 持有 H 天后收盘卖出
    if i + 1 + H >= len(all_dates):
        continue
    d_buy, d_sell = all_dates[i + 1], all_dates[i + 1 + H]
    ranked = [str(c)[:6] for c in dp["ranked"]]
    ranked = [c for c in ranked if c in px.columns]
    if len(ranked) < 30:
        continue
    p_buy = px.loc[d_buy, ranked]
    p_sell = px.loc[d_sell, ranked]
    ok = p_buy.notna() & p_sell.notna() & (p_buy > 0)
    if ok.sum() < 30:
        continue
    sub = pd.DataFrame({
        "code": np.array(ranked)[ok.values],
        "score": -np.arange(len(ranked))[ok.values],   # 排名越靠前分越高
        "price": p_buy[ok].values,
        "fwd": (p_sell[ok].values / p_buy[ok].values - 1),
    })
    rows.append((d, sub))

print(f"  可用信号日 {len(rows)} 个\n")

print("=" * 82)
print(f"按本金切分股票池 (持仓 {args.positions} 只, 最小100股 -> 可买最高价 = 本金/持仓数/100)")
print("=" * 82)
print(f"{'本金':>9} {'可买最高价':>10} {'池内占比':>9} {'IC':>8} {'IC t值':>8} "
      f"{'前3名平均收益':>13} {'胜率':>7}")
print("-" * 82)

summary = []
for cap in CAPS:
    maxpx = cap / args.positions / 100
    ics, top_rets, shares = [], [], []
    for d, sub in rows:
        aff = sub[sub["price"] <= maxpx]
        shares.append(len(aff) / len(sub))
        if len(aff) < 20:
            continue
        ic = spearmanr(aff["score"], aff["fwd"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
        top = aff.nlargest(args.positions, "score")
        top_rets.append(top["fwd"].mean())
    ics, top_rets = np.array(ics), np.array(top_rets)
    t = ics.mean() / ics.std() * np.sqrt(len(ics)) if len(ics) > 1 else np.nan
    summary.append({"cap": cap, "maxpx": maxpx, "ic": ics.mean(), "t": t,
                    "top": top_rets.mean(), "win": (top_rets > 0).mean(),
                    "share": np.mean(shares), "n": len(ics)})
    print(f"{int(cap):>9,} {maxpx:>10.1f} {np.mean(shares):>8.1%} "
          f"{ics.mean():>+8.4f} {t:>8.2f} {top_rets.mean():>+12.3%} "
          f"{(top_rets>0).mean():>6.1%}")

# 全池(不受资金限制)基准
ics, top_rets = [], []
for d, sub in rows:
    ic = spearmanr(sub["score"], sub["fwd"]).statistic
    if not np.isnan(ic):
        ics.append(ic)
    top_rets.append(sub.nlargest(args.positions, "score")["fwd"].mean())
ics, top_rets = np.array(ics), np.array(top_rets)
t = ics.mean() / ics.std() * np.sqrt(len(ics))
print("-" * 82)
print(f"{'无限制':>9} {'-':>10} {1.0:>8.1%} {ics.mean():>+8.4f} {t:>8.2f} "
      f"{top_rets.mean():>+12.3%} {(top_rets>0).mean():>6.1%}")

print()
print("=" * 82)
print("按价格分层的 IC (模型在哪个价格段有效?)")
print("=" * 82)
bands = [(0, 10), (10, 20), (20, 40), (40, 70), (70, 150), (150, 1e9)]
names = ["<¥10", "¥10-20", "¥20-40", "¥40-70", "¥70-150", ">¥150"]
print(f"{'价格段':>10} {'平均只数':>9} {'IC':>9} {'IC t值':>8} "
      f"{'前3名平均收益':>13} {'该段整体平均收益':>15}")
print("-" * 82)
for (lo, hi), nm in zip(bands, names):
    ics, top_rets, cnts, base = [], [], [], []
    for d, sub in rows:
        b = sub[(sub["price"] >= lo) & (sub["price"] < hi)]
        cnts.append(len(b))
        if len(b) < 15:
            continue
        ic = spearmanr(b["score"], b["fwd"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
        top_rets.append(b.nlargest(args.positions, "score")["fwd"].mean())
        base.append(b["fwd"].mean())
    if len(ics) < 20:
        print(f"{nm:>10} {np.mean(cnts):>9.1f}  (有效日不足, 跳过)")
        continue
    ics, top_rets, base = np.array(ics), np.array(top_rets), np.array(base)
    t = ics.mean() / ics.std() * np.sqrt(len(ics))
    print(f"{nm:>10} {np.mean(cnts):>9.1f} {ics.mean():>+9.4f} {t:>8.2f} "
          f"{top_rets.mean():>+12.3%} {base.mean():>+14.3%}")

print()
print("=" * 82)
print("解读")
print("=" * 82)
s20 = next(x for x in summary if x["cap"] == 20000) if any(
    x["cap"] == 20000 for x in summary) else None
if s20:
    print(f"  ¥20,000 只能覆盖池子的 {s20['share']:.0%}, "
          f"IC {s20['ic']:+.4f} (t={s20['t']:.2f}), 前3名平均 {s20['top']:+.3%}/期")
    print(f"  对比无限制: IC {ics.mean():+.4f}, 前3名平均 {top_rets.mean():+.3%}/期"
          if False else "")
print("  若低价池 IC 的 t 值 < 2 且前3名平均收益接近 0, 说明当前策略在 ¥20,000")
print("  这个本金下【结构性不可行】—— 应转向单价更低的标的(ETF)或先积累本金。")
