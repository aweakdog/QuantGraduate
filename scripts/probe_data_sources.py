"""探测哪些行情源在当前网络下可用"""
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak

TESTS = [
    ("东财 stock_zh_a_hist",
     lambda: ak.stock_zh_a_hist(symbol="000063", period="daily",
                                start_date="20260701", end_date="20260728", adjust="qfq")),
    ("腾讯 stock_zh_a_hist_tx",
     lambda: ak.stock_zh_a_hist_tx(symbol="sz000063",
                                   start_date="20260701", end_date="20260728", adjust="qfq")),
    ("新浪 stock_zh_a_daily",
     lambda: ak.stock_zh_a_daily(symbol="sz000063",
                                 start_date="20260701", end_date="20260728", adjust="qfq")),
    ("东财 实时快照 stock_zh_a_spot_em",
     lambda: ak.stock_zh_a_spot_em()),
    ("交易日历 tool_trade_date_hist_sina",
     lambda: ak.tool_trade_date_hist_sina()),
]

for name, fn in TESTS:
    t0 = time.time()
    try:
        df = fn()
        n = len(df) if df is not None else 0
        tail = ""
        if df is not None and n:
            dc = next((c for c in df.columns if c in ("日期", "date", "trade_date")), None)
            if dc is not None:
                tail = f"  最新={df[dc].max()}"
        print(f"  {name:34s} OK    {n:>6} 行  {time.time()-t0:5.1f}s{tail}")
    except Exception as e:
        print(f"  {name:34s} FAIL  {type(e).__name__}  {str(e)[:80]}")
    time.sleep(3)
