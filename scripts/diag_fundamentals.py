"""复现 calc_fundamental_features 静默失败的真实异常"""
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.feature_engine import DATA_DIR, _fund_pub_date, read_kline  # noqa

code6 = "600519"
hist = pd.read_parquet(Path(DATA_DIR) / "raw" / "fundamentals" / f"{code6}.parquet")
hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
hist = hist[hist["date"].dt.year.between(2010, 2030)]
hist = hist.sort_values("date").drop_duplicates(subset=["date"], keep="first")
hist = hist.set_index("date")

print("报告期截止日 (前8):", [str(d.date()) for d in hist.index[:8]])
mapped = hist.index.map(_fund_pub_date)
print("映射后发布日 (前8):", [str(pd.Timestamp(d).date()) for d in mapped[:8]])
print(f"映射后单调递增: {pd.DatetimeIndex(mapped).is_monotonic_increasing}")
print(f"映射后有重复: {pd.DatetimeIndex(mapped).duplicated().sum()} 个")
print(f"映射后有 NaT: {pd.isna(pd.DatetimeIndex(mapped)).sum()} 个")

hist.index = mapped
dk = read_kline(code6)
target = dk["date"].to_frame("date").set_index("date").index

print("\n逐列 reindex 试验:")
for col in ["pe", "pb", "roe", "eps"]:
    if col not in hist.columns:
        print(f"  {col}: 源数据无此列")
        continue
    s = pd.to_numeric(hist[col], errors="coerce").sort_index()
    try:
        r = s.reindex(target, method="ffill", limit=250)
        print(f"  {col}: OK 非空 {r.notna().sum()}/{len(r)}")
    except Exception:
        print(f"  {col}: 抛异常 ↓")
        traceback.print_exc(limit=1)
        break
