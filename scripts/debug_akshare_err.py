import time
import traceback

import akshare as ak

CASES = [
    ("短区间 20260601-20260728", dict(symbol="000063", period="daily",
                                   start_date="20260601", end_date="20260728", adjust="qfq")),
    ("长区间 20210101-20260728", dict(symbol="000063", period="daily",
                                   start_date="20210101", end_date="20260728", adjust="qfq")),
]

for label, kw in CASES:
    t0 = time.time()
    try:
        df = ak.stock_zh_a_hist(**kw)
        print(f"{label}: OK  {len(df)} 行  {time.time()-t0:.1f}s  最新 {df['日期'].max()}")
    except Exception as e:
        print(f"{label}: FAIL {type(e).__name__}  {time.time()-t0:.1f}s")
        print("   ", str(e)[:300])
    time.sleep(2)

print("\n--- 完整堆栈(长区间) ---")
try:
    ak.stock_zh_a_hist(symbol="000063", period="daily",
                       start_date="20210101", end_date="20260728", adjust="qfq")
except Exception:
    traceback.print_exc()
