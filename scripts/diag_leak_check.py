"""验证旧训练集的基本面列是否为"未来快照广播"式泄露

判据: 若某只股票在全部历史日期上 revenue 恒为同一个值, 说明是用某个
时点(如2026年)的快照值回填了整段历史 -> 严重未来泄露。
真实季报数据应随报告期逐季变化。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

old = pd.read_parquet(PROC / "backup" / sys.argv[1],
                      columns=["date", "code", "revenue", "profit", "total_assets", "roe"])
old["date"] = pd.to_datetime(old["date"])

print("=== 旧训练集: 每只股票的基本面取值个数 ===")
print("(nunique==1 表示整段历史恒定 -> 快照广播泄露)")
for col in ["revenue", "profit", "total_assets", "roe"]:
    sub = old[old[col].notna()]
    if sub.empty:
        print(f"  {col}: 全为 NaN")
        continue
    nun = sub.groupby("code")[col].nunique()
    span = sub.groupby("code")["date"].agg(["min", "max", "count"])
    n_const = int((nun == 1).sum())
    print(f"\n  {col}: {len(nun)} 只有值")
    print(f"    取值恒定(nunique==1)的股票: {n_const}/{len(nun)} = {n_const/len(nun):.1%}")
    print(f"    每只平均取值个数: {nun.mean():.2f}  中位 {nun.median():.0f}")
    print(f"    每只平均覆盖天数: {span['count'].mean():.0f}")
    ex = nun.index[0]
    e = sub[sub["code"] == ex].sort_values("date")
    print(f"    样例 {ex}: {len(e)} 行, 取值 {e[col].unique()[:5]}, "
          f"日期 {e['date'].min().date()} ~ {e['date'].max().date()}")

print("\n\n=== 新训练集同项对照 (应随季度变化) ===")
new = pd.read_parquet(PROC / "training_data_pit_v24.parquet",
                      columns=["date", "code", "revenue", "profit", "total_assets", "roe"])
new["date"] = pd.to_datetime(new["date"])
for col in ["revenue", "total_assets"]:
    sub = new[new[col].notna()]
    nun = sub.groupby("code")[col].nunique()
    print(f"  {col}: 每只平均取值个数 {nun.mean():.1f} (恒定的 {(nun==1).sum()}/{len(nun)})")

print("\n\n=== 关键: 旧 revenue 是否与'未来涨幅'相关 (泄露的直接证据) ===")
o2 = pd.read_parquet(PROC / "backup" / sys.argv[1], columns=["date", "code", "revenue", "fwd_5d_ret"])
o2["date"] = pd.to_datetime(o2["date"])
o2 = o2.dropna(subset=["revenue", "fwd_5d_ret"])
print(f"  旧: revenue 非空且有标签的行 {len(o2):,}")
if len(o2) > 1000:
    # 每只股票的 revenue(近乎恒定) vs 该股票整段未来收益均值
    g = o2.groupby("code").agg(rev=("revenue", "median"), fwd=("fwd_5d_ret", "mean"))
    print(f"  截面相关(股票级 revenue 中位数 vs 平均未来5日收益): "
          f"spearman={g['rev'].corr(g['fwd'], method='spearman'):+.4f}  (n={len(g)})")
    n2 = pd.read_parquet(PROC / "training_data_pit_v24.parquet",
                         columns=["date", "code", "revenue", "fwd_5d_ret"]).dropna()
    g2 = n2.groupby("code").agg(rev=("revenue", "median"), fwd=("fwd_5d_ret", "mean"))
    print(f"  新: 同样的相关性                                    "
          f"spearman={g2['rev'].corr(g2['fwd'], method='spearman'):+.4f}  (n={len(g2)})")
