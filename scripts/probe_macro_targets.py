"""探测 c1_base 实际需要的宏观序列: sox/usdjpy/usdcnh/cn2y/a50_futures/cn_commodity_idx"""
import pathlib
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
    t0 = time.time()
    signal.alarm(sec)
    try:
        df = fn()
        signal.alarm(0)
    except T:
        signal.alarm(0)
        print(f"  {name:44s} 超时>{sec}s")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"  {name:44s} FAIL {type(e).__name__}: {str(e)[:45]}")
        return None
    try:
        dc = next((c for c in df.columns if str(c) in ("日期", "date", "index")), None)
        latest = pd.to_datetime(df[dc], errors="coerce").max().date() if dc else "?"
        print(f"  {name:44s} OK {len(df):>6}行 {time.time()-t0:4.1f}s 最新={latest}")
        print(f"      列: {list(df.columns)[:8]}")
    except Exception as e:
        print(f"  {name:44s} OK 但解析异常 {e}")
    time.sleep(1)
    return df


print("=== 本地现状 (c1_base 需要的) ===")
for f in ["全球半导体SOX", "USDJPY", "USDCNH", "USDIND", "CN2Y",
          "A50期货", "中国大宗商品价格指数"]:
    p = pathlib.Path(f"data/raw/macro/{f}.parquet")
    if p.exists():
        d = pd.read_parquet(p)
        dc = next((c for c in d.columns if "日期" in str(c)), d.columns[0])
        print(f"  {f:22s} {len(d):>5}行  最新={pd.to_datetime(d[dc], errors='coerce').max().date()}")
    else:
        print(f"  {f:22s} 缺失")

print("\n=== 候选源探测 ===")
print("[SOX 费城半导体]")
go("index_us_stock_sina('.SOX')", lambda: ak.index_us_stock_sina(symbol=".SOX"))
go("stock_us_daily('SOXX')  ETF代理", lambda: ak.stock_us_daily(symbol="SOXX"))

print("[汇率]")
go("fx_pair_quote", lambda: ak.fx_pair_quote())
go("currency_hist_sina USDJPY", lambda: ak.currency_hist(symbol="usdjpy"))
go("forex_hist_em USDJPY", lambda: ak.forex_hist_em(symbol="USDJPY"))
go("index_investing_global 美元指数",
   lambda: ak.index_investing_global(country="美国", index_name="美元指数",
                                     period="每日", start_date="2026-06-01",
                                     end_date="2026-07-27"))

print("[A50 期货]")
go("futures_global_hist A50", lambda: ak.futures_global_hist(symbol="富时中国A50"))
go("index_investing_global 富时中国A50",
   lambda: ak.index_investing_global(country="中国", index_name="富时中国A50",
                                     period="每日", start_date="2026-06-01",
                                     end_date="2026-07-27"))

print("[中国大宗商品价格指数]")
go("macro_china_commodity_price_index", lambda: ak.macro_china_commodity_price_index())
