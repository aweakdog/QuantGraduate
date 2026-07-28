"""用美股 FX ETF 作为汇率代理, 与本地序列对比选出最佳源

候选:
  UUP  Invesco 美元指数基金        -> usdind
  FXY  Invesco 日元信托 (反向)     -> usdjpy  (FXY 涨 = 日元升 = usdjpy 跌)
  CYB  WisdomTree 人民币          -> usdcnh (反向)
  FXE  欧元信托
评价: 与本地序列在重叠期的【日收益相关性】(水平值不可比, ETF 有费率损耗)
"""
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "raw" / "macro"

CAND = [
    ("UUP", "USDIND", +1),
    ("FXY", "USDJPY", -1),
    ("CYB", "USDCNH", -1),
    ("FXE", "USDIND", -1),
    ("USDU", "USDIND", +1),
]

print("=== FX ETF 代理源评估 ===\n")
for sym, local_name, sign in CAND:
    try:
        d = ak.stock_us_daily(symbol=sym)
    except Exception as e:
        print(f"  {sym:5s} -> {local_name:7s}  FAIL {type(e).__name__}: {str(e)[:40]}")
        continue
    d["date"] = pd.to_datetime(d["date"])
    d = d[["date", "close"]].rename(columns={"close": "etf"}).sort_values("date")

    p = MACRO / f"{local_name}.parquet"
    if not p.exists():
        print(f"  {sym:5s} -> {local_name:7s}  本地缺失")
        continue
    loc = pd.read_parquet(p)
    loc["date"] = pd.to_datetime(loc["日期"])
    loc["v"] = pd.to_numeric(loc["最新值"], errors="coerce")
    loc = loc[["date", "v"]].dropna().sort_values("date")

    # ETF 是美股收盘, 本地是 iFinD -> 直接按自然日 merge
    m = loc.merge(d, on="date", how="inner").dropna()
    if len(m) < 30:
        print(f"  {sym:5s} -> {local_name:7s}  重叠仅 {len(m)} 天, 跳过")
        continue
    r_loc = m["v"].pct_change()
    r_etf = m["etf"].pct_change() * sign
    corr = r_loc.corr(r_etf)
    print(f"  {sym:5s} -> {local_name:7s}  重叠 {len(m):>4} 天  "
          f"日收益相关性 {corr:+.4f}   ETF最新 {d['date'].max().date()}")
    time.sleep(0.5)

print("\n=== 对照: 中行牌价推导 (已知) ===")
print("  中行 -> USDJPY   日收益相关性 +0.5332")
print("  中行 -> USDCNH   日收益相关性 +0.4177")

print("\n=== 本地 SOX vs 新浪 .SOX 校验 (确认可直接续接) ===")
try:
    sox = ak.index_us_stock_sina(symbol=".SOX")
    sox["date"] = pd.to_datetime(sox["date"])
    sox = sox[["date", "close"]].rename(columns={"close": "new"})
    loc = pd.read_parquet(MACRO / "全球半导体SOX.parquet")
    loc["date"] = pd.to_datetime(loc["日期"])
    loc["v"] = pd.to_numeric(loc["最新值"], errors="coerce")
    m = loc[["date", "v"]].merge(sox, on="date").dropna()
    dev = (m["new"] / m["v"] - 1).abs() * 100
    print(f"  重叠 {len(m)} 天 | 水平偏差 平均 {dev.mean():.4f}% 最大 {dev.max():.4f}% "
          f"| 收益相关性 {m['v'].pct_change().corr(m['new'].pct_change()):.4f}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")

print("\n=== 本地 大宗商品指数 vs akshare 校验 ===")
try:
    cc = ak.macro_china_commodity_price_index()
    cc["date"] = pd.to_datetime(cc["日期"])
    cc = cc[["date", "最新值"]].rename(columns={"最新值": "new"})
    loc = pd.read_parquet(MACRO / "中国大宗商品价格指数.parquet")
    loc["date"] = pd.to_datetime(loc["日期"])
    loc["v"] = pd.to_numeric(loc["最新值"], errors="coerce")
    m = loc[["date", "v"]].merge(cc, on="date").dropna()
    m["new"] = pd.to_numeric(m["new"], errors="coerce")
    dev = (m["new"] / m["v"] - 1).abs() * 100
    print(f"  重叠 {len(m)} 天 | 水平偏差 平均 {dev.mean():.4f}% 最大 {dev.max():.4f}%")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")

print("\n=== 本地 CN2Y vs bond_zh_us_rate 校验 ===")
try:
    b = ak.bond_zh_us_rate(start_date="20250101")
    b["date"] = pd.to_datetime(b["日期"])
    b = b[["date", "中国国债收益率2年"]].rename(columns={"中国国债收益率2年": "new"})
    loc = pd.read_parquet(MACRO / "CN2Y.parquet")
    loc["date"] = pd.to_datetime(loc["日期"])
    loc["v"] = pd.to_numeric(loc["最新值"], errors="coerce")
    m = loc[["date", "v"]].merge(b, on="date").dropna()
    dev = (m["new"] - m["v"]).abs()
    print(f"  重叠 {len(m)} 天 | 绝对偏差(bp) 平均 {dev.mean()*100:.2f} 最大 {dev.max()*100:.2f}")
    print(f"  本地末值 {m['v'].iloc[-1]:.4f} vs 新源 {m['new'].iloc[-1]:.4f}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")
