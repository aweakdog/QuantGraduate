"""对比新旧训练集里基本面列的实际数值, 判断哪个口径正确"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DATA = ROOT / "data"

OLD = PROC / "backup" / sys.argv[1] if len(sys.argv) > 1 else None
NEW = PROC / "training_data_pit_v24.parquet"

cols = ["date", "code", "revenue", "profit", "total_assets", "roe", "eps"]
old = pd.read_parquet(OLD, columns=cols)
new = pd.read_parquet(NEW, columns=cols)
old["date"] = pd.to_datetime(old["date"])
new["date"] = pd.to_datetime(new["date"])

code = "600519.SH"
print(f"===== {code} 基本面对比 (每季度取一行) =====")
o = old[old["code"] == code].set_index("date")
n = new[new["code"] == code].set_index("date")
sample = [d for d in o.index if d.day <= 3][:14]
print(f"{'日期':12s}{'revenue旧':>16}{'revenue新':>16}{'profit旧':>14}{'profit新':>14}")
for d in sample:
    if d not in n.index:
        continue
    print(f"{str(d.date()):12s}{o.loc[d,'revenue']:>16,.0f}{n.loc[d,'revenue']:>16,.0f}"
          f"{o.loc[d,'profit']:>14,.0f}{n.loc[d,'profit']:>14,.0f}")

print(f"\n===== 原始源文件 raw/fundamentals/600519.parquet =====")
h = pd.read_parquet(DATA / "raw" / "fundamentals" / "600519.parquet")
h["date"] = pd.to_datetime(h["date"])
print(h[["date", "revenue", "profit", "total_assets", "roe", "eps"]]
      .tail(10).to_string(index=False))

print(f"\n===== 全局量级对比 (中位数) =====")
print(f"{'列':16s}{'旧中位数':>20}{'新中位数':>20}{'新/旧':>10}")
for c in ["revenue", "profit", "total_assets", "roe", "eps"]:
    mo, mn = old[c].median(), new[c].median()
    ratio = (mn / mo) if (pd.notna(mo) and mo != 0) else float("nan")
    print(f"{c:16s}{mo:>20,.4f}{mn:>20,.4f}{ratio:>10.4g}")

print(f"\n非空覆盖率:")
for c in ["revenue", "profit", "total_assets", "roe", "eps"]:
    print(f"  {c:16s} 旧={old[c].notna().mean():6.1%}  新={new[c].notna().mean():6.1%}")
