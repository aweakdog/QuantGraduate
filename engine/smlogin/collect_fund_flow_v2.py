"""
历史资金面数据采集器 — v2 单进程版
运行: ths/Scripts/python.exe collect_fund_flow_v2.py

连接 thsdk 一次，循环查询每日6个资金面指标，保存为 parquet。
"""
import pandas as pd
import numpy as np
from thsdk import THS
import time
from datetime import datetime, timedelta
from pathlib import Path

STOCK = "601689.SH"
DATA_DIR = Path(r"D:\myAI\Hermes-Workspace\data")
DATA_DIR.mkdir(exist_ok=True)

METRICS = [
    ("主力资金流向", "main_force_net"),
    ("dde大单净额", "dde_net"),
    ("ddx", "ddx"),
    ("融资融券余额", "mtss_balance"),
    ("资金流向(万元)", "money_flow_wan"),
    ("主力增仓占比", "main_force_pct"),
]

def query(stock, date, metric):
    try:
        r = ths.wencai_nlp(f"{stock} {date} {metric}")
        if r.success and r.data:
            df = r.df
            cols = [c for c in df.columns if c not in ['股票代码','股票简称','最新价','最新涨跌幅']]
            if cols:
                val = str(df[cols[0]].iloc[0])
                if val and val != "None":
                    return float(val.replace(",", ""))
        time.sleep(0.35)
        return None
    except:
        time.sleep(0.35)
        return None

# 连接
print("Connecting thsdk...")
ths = THS()
ths.connect()
print("Connected.")

# 日期范围
end = datetime.now()
start = end - timedelta(days=180)
dates = []
d = start
while d <= end:
    if d.weekday() < 5:
        dates.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)

total = len(dates) * len(METRICS)
print(f"Trading days: {len(dates)}, queries: {total}")

rows = []
done = 0
t0 = time.time()
for date in dates:
    row = {"date": date}
    for metric_name, col_name in METRICS:
        val = query(STOCK, date, metric_name)
        row[col_name] = val
        done += 1
        if done % 20 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remain = (total - done) / rate if rate > 0 else 9999
            print(f"  [{done}/{total}] {date} {metric_name}={val}  ({elapsed:.0f}s, ~{remain:.0f}s)")
    rows.append(row)

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
path = DATA_DIR / f"{STOCK}_fund_flow_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
df.to_parquet(path)
print(f"\nSaved: {path}")
print(f"Rows: {len(df)}, Cols: {list(df.columns)}")
print(f"Time: {time.time()-t0:.0f}s")
