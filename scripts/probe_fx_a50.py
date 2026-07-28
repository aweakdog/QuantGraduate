"""汇率交叉推导 + A50 代理源探测"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(name, fn, sec=25):
    signal.alarm(sec)
    try:
        r = fn()
        signal.alarm(0)
        return r
    except T:
        signal.alarm(0)
        print(f"  {name}: 超时")
    except Exception as e:
        signal.alarm(0)
        print(f"  {name}: FAIL {type(e).__name__} {str(e)[:60]}")
    return None


print("=== 中行牌价 (用于交叉推导 USDJPY) ===")
S, E = "20260601", "20260727"
usd = go("美元", lambda: ak.currency_boc_sina(symbol="美元", start_date=S, end_date=E))
time.sleep(1)
jpy = go("日元", lambda: ak.currency_boc_sina(symbol="日元", start_date=S, end_date=E))

if usd is not None and jpy is not None:
    print(f"  美元 {len(usd)} 行, 日元 {len(jpy)} 行")
    print(f"  日元列: {list(jpy.columns)}")
    u = usd[["日期", "央行中间价"]].rename(columns={"央行中间价": "usdcny"})
    j = jpy[["日期", "央行中间价"]].rename(columns={"央行中间价": "jpycny100"})
    m = u.merge(j, on="日期").dropna()
    m["usdcny"] = pd.to_numeric(m["usdcny"], errors="coerce")
    m["jpycny100"] = pd.to_numeric(m["jpycny100"], errors="coerce")
    # 中行日元报价是 每100日元兑人民币 -> USDJPY = usdcny / (jpycny100/100)
    m["usdjpy_derived"] = m["usdcny"] / (m["jpycny100"] / 100)
    print(m.tail(5).to_string(index=False))

    loc = pd.read_parquet("data/raw/macro/USDJPY.parquet")
    loc["日期"] = pd.to_datetime(loc["日期"])
    m["日期"] = pd.to_datetime(m["日期"])
    cmp = loc.merge(m[["日期", "usdjpy_derived"]], on="日期").dropna()
    if len(cmp):
        d = (cmp["usdjpy_derived"] / pd.to_numeric(cmp["最新值"]) - 1).abs()
        print(f"\n  与本地 USDJPY 对比: 重叠 {len(cmp)} 天, "
              f"最大偏差 {d.max()*100:.3f}%, 平均 {d.mean()*100:.3f}%")
    else:
        print("\n  与本地无重叠日期")

print("\n=== A50 代理源 ===")
for name, fn in [
    ("stock_zh_index_daily sh000016(上证50)",
     lambda: ak.stock_zh_index_daily(symbol="sh000016")),
    ("stock_zh_index_daily sh000300(沪深300)",
     lambda: ak.stock_zh_index_daily(symbol="sh000300")),
]:
    r = go(name, fn)
    if r is not None:
        dc = "date" if "date" in r.columns else r.columns[0]
        print(f"  {name}: OK {len(r)} 行, 最新={pd.to_datetime(r[dc]).max().date()}, 列={list(r.columns)}")
    time.sleep(1)

print("\n=== 本地 A50 与 上证50 相关性检查 ===")
try:
    a50 = pd.read_parquet("data/raw/macro/A50期货.parquet")
    a50["日期"] = pd.to_datetime(a50["日期"])
    a50 = a50[["日期", "最新值"]].rename(columns={"最新值": "a50"})
    idx = ak.stock_zh_index_daily(symbol="sh000016")
    idx["日期"] = pd.to_datetime(idx["date"])
    idx = idx[["日期", "close"]].rename(columns={"close": "sse50"})
    m = a50.merge(idx, on="日期").dropna().sort_values("日期")
    m["a50"] = pd.to_numeric(m["a50"], errors="coerce")
    r1 = m["a50"].pct_change()
    r2 = m["sse50"].pct_change()
    print(f"  重叠 {len(m)} 天, 日收益相关性 = {r1.corr(r2):.4f}")
except Exception as e:
    print(f"  FAIL {type(e).__name__} {str(e)[:80]}")
