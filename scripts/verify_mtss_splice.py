"""验证 mtss_balance 补数后的接缝连续性: 逐股检查 06-30 -> 07-01 的跳变"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"

d = pd.read_parquet(FF)
d["date"] = pd.to_datetime(d["date"])
m = d[d["mtss_balance"].notna()][["date", "code", "mtss_balance"]].copy()
print(f"mtss_balance: {len(m):,} 行, {m['code'].nunique()} 只, "
      f"{m['date'].min().date()} ~ {m['date'].max().date()}")

SEAM = pd.Timestamp("2026-07-01")
piv = m.pivot_table(index="date", columns="code", values="mtss_balance")
piv = piv.sort_index()
ret = piv.pct_change()

hist = ret[ret.index < SEAM].tail(120)
seam = ret[ret.index == SEAM]
if not len(seam):
    print("无接缝日数据")
    raise SystemExit

mu, sd = hist.mean(), hist.std()
z = ((seam.iloc[0] - mu) / sd).dropna()
print(f"\n接缝日 {SEAM.date()} 相对前120日分布的 z 值 ({len(z)} 只):")
print(f"  |z|<3  : {(z.abs()<3).sum():>4} 只 ({100*(z.abs()<3).mean():5.1f}%)")
print(f"  3~5    : {((z.abs()>=3)&(z.abs()<5)).sum():>4} 只")
print(f"  >5     : {(z.abs()>=5).sum():>4} 只")
print(f"  中位|z| = {z.abs().median():.3f}   均值 = {z.abs().mean():.3f}")

sr = seam.iloc[0].dropna()
print(f"\n接缝日涨跌幅分布: 中位 {sr.median()*100:+.3f}%  "
      f"P5 {sr.quantile(0.05)*100:+.2f}%  P95 {sr.quantile(0.95)*100:+.2f}%")

# 对照: 历史上普通交易日的日变化分布
hr = hist.stack().dropna()
print(f"历史普通日分布:   中位 {hr.median()*100:+.3f}%  "
      f"P5 {hr.quantile(0.05)*100:+.2f}%  P95 {hr.quantile(0.95)*100:+.2f}%")

bad = z[z.abs() >= 5]
if len(bad):
    print(f"\n|z|>=5 的股票 (前10):")
    for c, v in bad.abs().sort_values(ascending=False).head(10).items():
        print(f"  {c}  z={z[c]:+.2f}  变化 {seam.iloc[0][c]*100:+.2f}%")
else:
    print("\n无 |z|>=5 异常 — 接缝平滑, 口径一致")

print(f"\n=== 逐日覆盖 (最近12个交易日) ===")
print(m[m["date"] >= "2026-06-20"].groupby("date").size().to_string())
