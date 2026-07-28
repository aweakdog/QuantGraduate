"""1) 验证中行牌价交叉推导 USDJPY/USDCNH 的精度  2) 排查基本面脏日期"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

print("=" * 60)
print("【1. 汇率交叉推导精度验证】")
S, E = "20250101", "20260727"
usd = ak.currency_boc_sina(symbol="美元", start_date=S, end_date=E)
jpy = ak.currency_boc_sina(symbol="日元", start_date=S, end_date=E)
print(f"  中行牌价: 美元 {len(usd)} 行, 日元 {len(jpy)} 行")

u = usd[["日期", "央行中间价"]].rename(columns={"央行中间价": "u"})
j = jpy[["日期", "央行中间价"]].rename(columns={"央行中间价": "j"})
m = u.merge(j, on="日期")
m["u"] = pd.to_numeric(m["u"], errors="coerce")
m["j"] = pd.to_numeric(m["j"], errors="coerce")
m = m.dropna()
m = m[(m["u"] > 0) & (m["j"] > 0)]
m["日期"] = pd.to_datetime(m["日期"])
# 中行报价单位: 每100外币兑人民币
m["usdcny"] = m["u"] / 100                 # USD/CNY
m["usdjpy"] = m["u"] / m["j"]              # (u/100) / (j/100)

print(f"  推导样例(最近3天):")
print(m.tail(3)[["日期", "usdcny", "usdjpy"]].to_string(index=False))

for name, col in [("USDJPY", "usdjpy"), ("USDCNH", "usdcny")]:
    p = ROOT / f"data/raw/macro/{name}.parquet"
    loc = pd.read_parquet(p)
    loc["日期"] = pd.to_datetime(loc["日期"])
    loc["v"] = pd.to_numeric(loc["最新值"], errors="coerce")
    c = loc.merge(m[["日期", col]], on="日期").dropna(subset=["v", col])
    if not len(c):
        print(f"  {name}: 无重叠日期")
        continue
    d = (c[col] / c["v"] - 1).abs() * 100
    corr = c[col].pct_change().corr(c["v"].pct_change())
    print(f"\n  {name}: 重叠 {len(c)} 天 | 水平偏差 平均 {d.mean():.3f}% 最大 {d.max():.3f}% "
          f"| 日收益相关性 {corr:.4f}")
    print(f"       本地末值 {c['v'].iloc[-1]:.4f} vs 推导 {c[col].iloc[-1]:.4f} @ {c['日期'].iloc[-1].date()}")

print("\n" + "=" * 60)
print("【2. 基本面脏日期排查】")
fd = ROOT / "data" / "raw" / "fundamentals"
fs = sorted(fd.glob("*.parquet"))
print(f"  {len(fs)} 个文件")
d0 = pd.read_parquet(fs[0])
print(f"  样例 {fs[0].stem}: {len(d0)} 行")
print(f"  列: {list(d0.columns)}")
print(f"\n  前3行:\n{d0.head(3).to_string()[:900]}")

dcols = [c for c in d0.columns if "date" in str(c).lower() or "日期" in str(c) or "期" in str(c)]
print(f"\n  日期候选列: {dcols}")
for dc in dcols:
    bad, mx = 0, []
    for f in fs:
        try:
            d = pd.read_parquet(f, columns=[dc])
        except Exception:
            continue
        s = pd.to_datetime(d[dc], errors="coerce")
        bad += int((s > pd.Timestamp("2030-01-01")).sum())
        v = s[s < pd.Timestamp("2030-01-01")].max()
        if pd.notna(v):
            mx.append(v)
    print(f"    列 '{dc}': 未来脏值 {bad} 条 | 合理值最新 = "
          f"{max(mx).date() if mx else '-'} (覆盖 {len(mx)} 文件)")
