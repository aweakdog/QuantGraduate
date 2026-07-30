"""为什么同一份预测, 2万本金亏钱而10万本金赚钱?

对照 pos=3 / regime=off 两个回测(唯一差别是本金):
  cap=20000  -> -17.6%
  cap=100000 -> +182.0%
成本占比几乎相同(10.3% vs 12.2%), 部署率也接近(93.4% vs 96.1%),
所以差异不可能只来自"手续费更贵"。真正的怀疑对象:

  A股最小交易单位是100股。¥20,000/3只 = 每只预算 ¥6,667,
  意味着股价 > ¥66.7 的票【一手都买不起】, 模型选出来也只能跳过,
  被迫用排名更靠后的低价股替代。
  -> 如果低价股整体表现更差, 小资金就会系统性地拿到更差的组合。

本脚本检验: 两者实际买到的股票有多大差异, 买入价分布如何, 以及
被迫替代的部分贡献了多少盈亏。
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

RUNS = {
    20000: "wf_daily_P1SW_h10_n3_off_ts2022-09-01_te2026-07-27_cap20000.json",
    100000: "wf_daily_P1SW_h10_n3_off_ts2022-09-01_te2026-07-27_cap100000.json",
}

data = {}
for cap, fn in RUNS.items():
    p = PROC / fn
    if not p.exists():
        raise SystemExit(f"缺失 {p}")
    data[cap] = json.load(open(p, encoding="utf-8"))


def round_trips(trades):
    books, rts = defaultdict(deque), []
    for t in sorted(trades, key=lambda x: x["date"]):
        code, sh = t["code"], t["shares"]
        if t["action"] == "buy":
            books[code].append({"shares": sh, "cost": -t["net"], "px": t["price"],
                                "date": t["date"]})
        else:
            proceeds, remain = t["net"], sh
            while remain > 0 and books[code]:
                lot = books[code][0]
                take = min(remain, lot["shares"])
                fc = lot["cost"] * take / lot["shares"]
                fp = proceeds * take / sh
                rts.append({"code": code, "buy_px": lot["px"],
                            "buy_date": lot["date"], "sell_date": t["date"],
                            "pnl": fp - fc, "cost": fc,
                            "ret_pct": (fp / fc - 1) * 100 if fc else 0})
                lot["shares"] -= take
                lot["cost"] -= fc
                remain -= take
                if lot["shares"] <= 0:
                    books[code].popleft()
    return pd.DataFrame(rts)


print("=" * 76)
print("一、买入价分布 (最小100股 -> 小资金买不起高价股)")
print("=" * 76)
buys = {}
for cap, d in data.items():
    b = pd.DataFrame([t for t in d["trades"] if t["action"] == "buy"])
    buys[cap] = b
    budget = cap / 3
    print(f"\n  本金 ¥{cap:,} (每只预算 ¥{budget:,.0f} -> 理论可买最高股价 ¥{budget/100:.1f})")
    print(f"    买入笔数 {len(b)} | 买入价: 中位数 ¥{b['price'].median():.2f} "
          f"| 均值 ¥{b['price'].mean():.2f} | 最高 ¥{b['price'].max():.2f}")
    for q in [0.5, 0.75, 0.9, 0.99]:
        print(f"      {int(q*100)}分位 ¥{b['price'].quantile(q):.2f}")
    over = (b["price"] * 100 > budget).sum()
    print(f"    买入价 x100股 超过预算的笔数: {over}")

print()
print("=" * 76)
print("二、两者实际买到的股票差异")
print("=" * 76)
s20 = set(buys[20000]["code"])
s100 = set(buys[100000]["code"])
print(f"  2万买过 {len(s20)} 只 | 10万买过 {len(s100)} 只")
print(f"  交集 {len(s20 & s100)} 只 | Jaccard {len(s20&s100)/len(s20|s100):.1%}")
only100 = s100 - s20
print(f"\n  只有10万买过、2万从未买过的 {len(only100)} 只 (即被资金门槛挡掉的):")
b100 = buys[100000]
sub = b100[b100["code"].isin(only100)].groupby("code")["price"].agg(["mean", "count"])
sub = sub.sort_values("mean", ascending=False)
for c, r in sub.head(12).iterrows():
    print(f"    {c:<11} 均价 ¥{r['mean']:>8.2f}  买过 {int(r['count'])} 次")
if len(sub):
    print(f"  这批股票均价中位数 ¥{sub['mean'].median():.2f} "
          f"(对比 2万组合买入价中位数 ¥{buys[20000]['price'].median():.2f})")

print()
print("=" * 76)
print("三、盈亏归因: 高价股 vs 低价股")
print("=" * 76)
for cap, d in data.items():
    rt = round_trips(d["trades"])
    if rt.empty:
        continue
    print(f"\n  --- 本金 ¥{cap:,} (共 {len(rt)} 个回合) ---")
    bins = [0, 10, 20, 40, 70, 1e9]
    labels = ["<¥10", "¥10-20", "¥20-40", "¥40-70", ">¥70"]
    rt["px_bin"] = pd.cut(rt["buy_px"], bins=bins, labels=labels)
    g = rt.groupby("px_bin", observed=False).agg(
        回合数=("pnl", "size"), 净盈亏=("pnl", "sum"),
        平均收益率=("ret_pct", "mean"), 胜率=("pnl", lambda x: (x > 0).mean() * 100))
    g["净盈亏"] = g["净盈亏"].round(0)
    g["平均收益率"] = g["平均收益率"].round(2)
    g["胜率"] = g["胜率"].round(1)
    print(g.to_string())

print()
print("=" * 76)
print("四、结论指标")
print("=" * 76)
for cap, d in data.items():
    s = d["summary"]
    rt = round_trips(d["trades"])
    print(f"  ¥{cap:>7,}: 收益 {s['total_return_pct']:>7.1f}% | "
          f"平均回合收益率 {rt['ret_pct'].mean():>+6.2f}% | "
          f"胜率 {(rt['pnl']>0).mean():>5.1%} | 成本 {s['total_cost_pct']:.1f}%")
