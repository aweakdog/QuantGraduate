"""最低佣金反事实测算"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed/wf_daily_small3_t1open_ts2022-09-01_te2026-07-24_cap10000.json"
j = json.loads(P.read_text())
t = pd.DataFrame(j["trades"])
g = pd.to_numeric(t["gross"])
f = pd.to_numeric(t["fee"])
prop = 0.0006 * g.sum()

print("=== 费用反事实 ===")
print(f"  成交金额合计 {g.sum():>12,.0f} 元 | 平均单笔 {g.mean():,.0f} 元")
print(f"  实收费用     {f.sum():>12,.0f} 元  (实际费率 {100*f.sum()/g.sum():.3f}%)")
print(f"  纯比例应收   {prop:>12,.0f} 元  (名义费率 0.060%)")
print(f"  最低佣金多收 {f.sum()-prop:>12,.0f} 元 = 本金的 {100*(f.sum()-prop)/10000:.1f}%")

print("\n=== summary ===")
for k, v in j["summary"].items():
    print(f"  {k:26s} {v}")

print("\n=== 买单被拒原因分布 ===")
if "reason" in t.columns:
    print(t.groupby(["action", "reason"]).size().to_string())

print("\n=== 每笔买入股数分布 (整手约束) ===")
b = t[t["action"] == "buy"]
sh = pd.to_numeric(b["shares"])
print(sh.value_counts().sort_index().head(8).to_string())
print(f"  只买到最小一手(100股)的比例: {100*(sh==100).mean():.1f}%")
