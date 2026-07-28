"""审计股票池前视偏差: 手工挑选的216只热门题材股 vs 真实宽基指数

watchlist_216.json 首次入库 2026-07-09, 但回测区间从 2022-09 开始。
若该池子显著跑赢同期宽基指数, 说明"等权买入持有基准"本身不可复现,
"跑赢/接近基准"这个结论就失去意义。
"""
import json
import signal
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "2023-09-20", "2026-07-20"   # 与回测持仓期一致


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))

# ── 1. 池子构成 ──
wl = json.loads((ROOT / "data/universe/watchlist_216.json").read_text())["watchlist"]
print(f"股票池: {len(wl)} 只 (watchlist_216.json, 2026-07-09 入库)")
themes = pd.Series([s.get("theme", "未分类") for s in wl]).value_counts()
print(f"覆盖 {len(themes)} 个题材, 最集中的 12 个:")
for t, n in themes.head(12).items():
    print(f"    {t:22s} {n:>3} 只")

boards = pd.Series([s["code"][:3] for s in wl]).value_counts()
print(f"\n板块分布:")
name = {"688": "科创板", "300": "创业板", "000": "深主板", "002": "中小板",
        "600": "沪主板", "601": "沪主板", "603": "沪主板", "605": "沪主板"}
for b, n in boards.items():
    print(f"    {name.get(b, b):8s}({b}) {n:>3} 只 ({100*n/len(wl):4.1f}%)")

# ── 2. 池子等权买入持有收益 (回测基准口径) ──
codes = [s["code"][:6] for s in wl]
rets = []
for c in codes:
    p = ROOT / f"data/raw/kline/{c}.parquet"
    if not p.exists():
        continue
    k = pd.read_parquet(p, columns=["date", "close"])
    k["date"] = pd.to_datetime(k["date"])
    k = k[(k["date"] >= START) & (k["date"] <= END)].sort_values("date")
    if len(k) < 100:
        continue
    rets.append(pd.Series(k["close"].values, index=k["date"], name=c))
px = pd.concat(rets, axis=1).ffill()
eq = px.pct_change().mean(axis=1)
basket = (1 + eq).prod() - 1
yrs = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
print(f"\n=== 216池 等权买入持有 ({len(px.columns)} 只有效) ===")
print(f"    总收益 {basket*100:+.1f}%   年化 {((1+basket)**(1/yrs)-1)*100:+.1f}%")

# ── 3. 真实宽基指数 ──
IDX = [("000300", "沪深300"), ("000905", "中证500"), ("399006", "创业板指"),
       ("000688", "科创50"), ("000001", "上证指数"), ("399001", "深证成指")]
print(f"\n=== 同期真实指数 ({START} ~ {END}) ===")
out = []
for code, nm in IDX:
    signal.alarm(30)
    try:
        d = ak.index_zh_a_hist(symbol=code, period="daily",
                               start_date=START.replace("-", ""),
                               end_date=END.replace("-", ""))
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        print(f"    {nm:10s} 取数失败 {type(e).__name__}")
        continue
    if d is None or len(d) < 50:
        print(f"    {nm:10s} 数据不足")
        continue
    c = pd.to_numeric(d["收盘"], errors="coerce").dropna()
    r = c.iloc[-1] / c.iloc[0] - 1
    out.append((nm, r))
    print(f"    {nm:10s} 总收益 {r*100:+8.1f}%   年化 {((1+r)**(1/yrs)-1)*100:+6.1f}%")

# ── 4. 结论 ──
if out:
    best = max(out, key=lambda x: x[1])
    print(f"\n=== 偏差量化 ===")
    print(f"    216池等权      {basket*100:+.1f}%")
    print(f"    最强宽基指数   {best[0]} {best[1]*100:+.1f}%")
    print(f"    超出幅度       {(basket-best[1])*100:+.1f} 个百分点")
    print(f"    倍数           {(1+basket)/(1+best[1]):.2f}x")
