"""诊断重建漂移: (1) 基本面列为何丢失 (2) con_* 漂移是"数据追加"还是"口径不同" """
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"
DATA = ROOT / "data"

print("=" * 70)
print("  1. 基本面特征为何缺失")
print("=" * 70)
from pipeline.feature_engine import (calc_fundamental_features, read_fund_flow,
                                     read_kline)

code6 = "600519"
print(f"raw/fundamentals/{code6}.parquet 存在: "
      f"{(DATA / 'raw' / 'fundamentals' / f'{code6}.parquet').exists()}")
print(f"universe/stock_list.parquet 存在: "
      f"{(DATA / 'universe' / 'stock_list.parquet').exists()}")
fp = DATA / "raw" / "fundamentals" / f"{code6}.parquet"
if fp.exists():
    h = pd.read_parquet(fp)
    print(f"  内容: {len(h)} 行, 列={list(h.columns)[:12]}")
    if "date" in h.columns:
        print(f"  日期范围: {pd.to_datetime(h['date']).min()} ~ {pd.to_datetime(h['date']).max()}")

dk = read_kline(code6)
res = calc_fundamental_features(code6, dk["date"])
print(f"calc_fundamental_features 返回: "
      f"{'None (这就是丢列原因)' if res is None else f'{len(res.columns)} 列 {list(res.columns)[:8]}'}")

print()
print("=" * 70)
print("  2. con_* 漂移按年份分布 (若全历史都漂 -> 口径/代码不同; 只近期漂 -> 数据追加)")
print("=" * 70)
old = pd.read_parquet(PROC / "training_data_pit_v24.parquet",
                      columns=["date", "code", "con_amount_ma5", "con_mf_net_ma5"])
new = pd.read_parquet(PROC / "_smoke_test.parquet",
                      columns=["date", "code", "con_amount_ma5", "con_mf_net_ma5"])
old["date"] = pd.to_datetime(old["date"])
new["date"] = pd.to_datetime(new["date"])
m = old.merge(new, on=["date", "code"], suffixes=("_o", "_n"))
m["year"] = m["date"].dt.year
for col in ["con_amount_ma5", "con_mf_net_ma5"]:
    print(f"\n  {col}:")
    for y, g in m.groupby("year"):
        a, b = g[f"{col}_o"], g[f"{col}_n"]
        both = a.notna() & b.notna()
        if not both.any():
            continue
        eq = (a[both] == b[both]).mean()
        d = (a[both] - b[both]).abs().max()
        print(f"    {y}: 完全相等占比={eq:6.1%}  最大绝对偏差={d:.6g}")

print()
print("  非空覆盖率对比 (旧 vs 新):")
for col in ["con_amount_ma5", "con_mf_net_ma5"]:
    print(f"    {col}: 旧={old[col].notna().mean():.1%}  新={new[col].notna().mean():.1%}")

print()
print("=" * 70)
print("  3. 概念特征的输入源现状")
print("=" * 70)
ff = read_fund_flow("600519")
print(f"read_fund_flow(600519): "
      f"{'None' if ff is None else f'{len(ff)} 行, max={pd.to_datetime(ff[chr(100)+chr(97)+chr(116)+chr(101)]).max()}'}")
kl = read_kline("600519")
print(f"read_kline(600519): {len(kl)} 行, max={kl['date'].max()}")
