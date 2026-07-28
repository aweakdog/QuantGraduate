"""验证 新浪/腾讯 的前复权口径能否与本地 data/raw/kline 对齐"""
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

TESTS = [("000063", "sz000063"), ("600519", "sh600519"), ("300750", "sz300750")]
S, E = "20260601", "20260717"

for code, sym in TESTS:
    old = pd.read_parquet(f"data/raw/kline/{code}.parquet")
    old["date"] = pd.to_datetime(old["date"])
    old = old.set_index("date")
    print(f"\n=== {code} ===")

    # 新浪
    try:
        sn = ak.stock_zh_a_daily(symbol=sym, start_date=S, end_date=E, adjust="qfq")
        sn["date"] = pd.to_datetime(sn["date"])
        sn = sn.set_index("date")
        j = old[["close"]].join(sn[["close"]], how="inner", rsuffix="_new").dropna()
        d = (j["close_new"] / j["close"] - 1).abs()
        print(f"  新浪 qfq : 重叠{len(j):3d}天  最大偏差 {d.max()*100:8.4f}%  "
              f"{'一致' if d.max() < 0.001 else '不一致'}")
        print(f"    列: {list(sn.reset_index().columns)}")
    except Exception as e:
        print(f"  新浪 FAIL {type(e).__name__} {str(e)[:70]}")

    # 腾讯
    try:
        tx = ak.stock_zh_a_hist_tx(symbol=sym, start_date=S, end_date=E, adjust="qfq")
        dc = "date" if "date" in tx.columns else tx.columns[0]
        tx[dc] = pd.to_datetime(tx[dc])
        tx = tx.set_index(dc)
        cc = "close" if "close" in tx.columns else None
        if cc:
            j = old[["close"]].join(tx[[cc]], how="inner", rsuffix="_new").dropna()
            d = (j[f"{cc}_new"] / j["close"] - 1).abs() if f"{cc}_new" in j else None
            if d is not None:
                print(f"  腾讯 qfq : 重叠{len(j):3d}天  最大偏差 {d.max()*100:8.4f}%  "
                      f"{'一致' if d.max() < 0.001 else '不一致'}")
        print(f"    列: {list(tx.reset_index().columns)}")
    except Exception as e:
        print(f"  腾讯 FAIL {type(e).__name__} {str(e)[:70]}")

print("\n--- 新浪最近数据样例 ---")
s = ak.stock_zh_a_daily(symbol="sz000063", start_date="20260715", end_date="20260728", adjust="qfq")
print(s.to_string(index=False))
