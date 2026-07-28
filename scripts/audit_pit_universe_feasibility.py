"""可行性检查: 能否用现有数据构建 point-in-time 无偏股票池

核心问题:
  1. kline 目录是否含已退市股票? (否则幸存者偏差无法消除)
  2. 有无历史市值? 无则用"日均成交额"作 PIT 安全的规模/流动性代理
  3. ST 状态历史是否可得?
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "data/raw/kline"

files = sorted(KL.glob("*.parquet"))
print(f"kline 文件数: {len(files)}")

rows = []
for p in files:
    try:
        k = pd.read_parquet(p, columns=["date"])
    except Exception:
        continue
    if not len(k):
        continue
    d = pd.to_datetime(k["date"])
    rows.append((p.stem, d.min(), d.max(), len(k)))
m = pd.DataFrame(rows, columns=["code", "first", "last", "n"])
print(f"可读: {len(m)} 只")

LATEST = m["last"].max()
print(f"全局最新交易日: {LATEST.date()}")

print("\n=== 幸存者偏差检查: 最后交易日分布 ===")
for cut, lbl in [(7, "1周内"), (30, "1月内"), (90, "3月内"), (365, "1年内")]:
    n = (m["last"] >= LATEST - pd.Timedelta(days=cut)).sum()
    print(f"  {lbl:6s}仍在交易: {n:>5} 只 ({100*n/len(m):5.1f}%)")
dead = m[m["last"] < LATEST - pd.Timedelta(days=90)]
print(f"\n  停止更新>90天(疑似退市/长停): {len(dead)} 只 ({100*len(dead)/len(m):.1f}%)")
if len(dead):
    print(f"  其最后交易日分布(按年):")
    print(dead["last"].dt.year.value_counts().sort_index().to_string())
    print(f"  样例: {dead.nsmallest(8,'last')[['code','first','last','n']].to_string(index=False)}")

print("\n=== 上市时间分布 (首个K线日) ===")
print(m["first"].dt.year.value_counts().sort_index().to_string())

print("\n=== 历史长度 ===")
for thr, lbl in [(500, "≥500日(约2年)"), (250, "≥250日(约1年)"), (120, "≥120日")]:
    print(f"  {lbl:16s}: {(m['n']>=thr).sum():>5} 只")

print("\n=== kline 可用列 (决定能否算成交额) ===")
k0 = pd.read_parquet(files[0])
print(f"  {list(k0.columns)}")
has_amt = "amount" in k0.columns
print(f"  含成交额(amount): {has_amt}")
if has_amt:
    print(f"  样例(最近3行):")
    print(k0.tail(3)[["date", "close", "volume", "amount"]].to_string(index=False))

print("\n=== ST 状态: 当前清单里的名称 ===")
p = ROOT / "data/raw/all_stock_list.parquet"
if p.exists():
    a = pd.read_parquet(p)
    print(f"  all_stock_list: {len(a)} 行, 列 {list(a.columns)}")
    st = a[a["name"].astype(str).str.contains("ST", na=False)]
    print(f"  当前名称含ST: {len(st)} 只 (仅当前快照, 无历史ST状态)")
