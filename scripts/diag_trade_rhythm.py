"""从成交记录反推实际操作节奏: 多久操作一次? 买卖是否同日? 信号日与成交日差几天?"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed/wf_daily_cap100000_3pos_ts2022-09-01_te2026-07-24_cap100000.json"
j = json.loads(P.read_text())
t = pd.DataFrame(j["trades"])
t["date"] = pd.to_datetime(t["date"])
t["signal_date"] = pd.to_datetime(t["signal_date"])

print(f"exec_mode={j['exec_mode']}  portfolio_mode={j['portfolio_mode']}  "
      f"hold_days={j['hold_days']}  target_positions={j['target_positions']}")

print("\n=== 信号日 -> 成交日 间隔(自然日) ===")
print((t["date"] - t["signal_date"]).dt.days.value_counts().sort_index().to_string())

print("\n=== 有操作的日子 ===")
days = sorted(t["date"].unique())
print(f"  共 {len(days)} 个操作日")
gaps = pd.Series(days).diff().dt.days.dropna()
print(f"  相邻操作日间隔(自然日): 中位 {gaps.median():.0f}  众数 {gaps.mode().tolist()}")

print("\n=== 每个操作日的动作构成 (前10个) ===")
for d in days[:10]:
    g = t[t["date"] == d]
    acts = g.groupby("action").size().to_dict()
    print(f"  {pd.Timestamp(d).date()}  {acts}")

print("\n=== 买卖是否同日发生 ===")
per = t.groupby("date")["action"].apply(lambda s: set(s))
both = per.apply(lambda s: "buy" in s and ("sell" in s or "force_sell" in s))
print(f"  同日既买又卖: {both.sum()}/{len(per)} 天 ({100*both.mean():.1f}%)")
print(f"  只买不卖: {per.apply(lambda s: 'buy' in s and 'sell' not in s and 'force_sell' not in s).sum()} 天")

print("\n=== 最近6个操作日明细 ===")
for d in days[-6:]:
    g = t[t["date"] == d].sort_values("action")
    print(f"\n  【{pd.Timestamp(d).date()}】(信号日 {g['signal_date'].iloc[0].date()})")
    for _, r in g.iterrows():
        print(f"    {r['action']:10s} {str(r['code'])[:6]}  {int(r['shares']):>5d}股 "
              f"@ {r['price']:>8.2f}  金额 {r['gross']:>10,.0f}  {r['reason']}")

print("\n=== 全年操作日频次 ===")
s = pd.Series(1, index=pd.DatetimeIndex(days))
print(s.groupby(s.index.year).size().to_string())
