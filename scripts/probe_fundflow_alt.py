"""个股资金流的备选路径 + 深交所两融明细"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(name, fn, sec=45):
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
        print(f"  {name:44s} FAIL {type(e).__name__}: {str(e)[:36]}")
        return None
    if df is None or not len(df):
        print(f"  {name:44s} 空")
        return None
    dc = next((c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()), None)
    mx = ""
    if dc:
        s = pd.to_datetime(df[dc], errors="coerce")
        if s.notna().any():
            mx = f" 最新={s.max().date()}"
    print(f"  {name:44s} OK {len(df):>5}行 {time.time()-t0:4.1f}s{mx}")
    print(f"      列: {list(df.columns)[:9]}")
    time.sleep(0.8)
    return df


print("=== 深交所两融明细 (补全 mtss) ===")
go("stock_margin_detail_szse 20260724",
   lambda: ak.stock_margin_detail_szse(date="20260724"))
go("stock_margin_szse 汇总",
   lambda: ak.stock_margin_szse(date="20260724"))

print("\n=== 个股资金流备选路径 ===")
go("stock_fund_flow_individual 3日排行",
   lambda: ak.stock_fund_flow_individual(symbol="3日排行"), sec=60)
go("stock_individual_fund_flow_rank 今日",
   lambda: ak.stock_individual_fund_flow_rank(indicator="今日"))

print("\n=== 逐笔/分时 (可重构主力资金流) ===")
go("stock_zh_a_tick_tx 000063",
   lambda: ak.stock_zh_a_tick_tx(symbol="sz000063"))
go("stock_intraday_em 000063",
   lambda: ak.stock_intraday_em(symbol="000063"))
go("stock_zh_a_hist_min_em 000063 1分钟",
   lambda: ak.stock_zh_a_hist_min_em(symbol="000063", period="1",
                                     start_date="2026-07-24 09:30:00",
                                     end_date="2026-07-24 15:00:00"))

print("\n=== 北向资金 (可作替代资金面因子) ===")
go("stock_hsgt_hist_em 沪股通", lambda: ak.stock_hsgt_hist_em(symbol="沪股通"))
go("stock_hsgt_fund_flow_summary_em", lambda: ak.stock_hsgt_fund_flow_summary_em())
