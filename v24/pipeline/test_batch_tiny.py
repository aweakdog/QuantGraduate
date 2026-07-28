"""快速测试: batch_fund_flow 是否能在前台运行"""
import sys, time
sys.stdout.reconfigure(line_buffering=True)

print("step 1: imports...", flush=True)
import json
from datetime import datetime, timedelta
from pathlib import Path
print("step 2: pandas...", flush=True)
import pandas as pd
print("step 3: thsdk...", flush=True)
from thsdk import THS
print("step 4: connecting...", flush=True)

from pipeline.config import settings
KQ = {"username": settings.THS_USERNAME, "password": settings.THS_PASSWORD}

# 读 watchlist
WATCHLIST_PATH = str(settings.WATCHLIST_PATH)
with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
    watch_data = json.load(f)
stocks = watch_data["watchlist"]
print(f"watchlist: {len(stocks)} stocks", flush=True)

# 交易日
END = datetime.now()
START = END - timedelta(days=90)
trade_days = []
d = START
while d <= END:
    if d.weekday() < 5:
        trade_days.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)
print(f"trade days: {len(trade_days)} ({trade_days[0]}~{trade_days[-1]})", flush=True)

# 只做前3只
with THS(KQ) as ths:
    print("connected!", flush=True)
    for i, s in enumerate(stocks[:3]):
        code = s["code"]
        print(f"stock {i+1}/3: {code} {s['name']}", flush=True)
        
        r = ths.wencai_nlp(f"{code} {trade_days[0]} 主力资金流向,主力增仓占比,dde大单净额,融资融券余额,资金流向(万元)")
        if r.success and r.data:
            print(f"  OK: {r.df.columns.tolist()}", flush=True)
            print(f"  row: {r.data[0]}", flush=True)
        else:
            print(f"  FAIL: {r.error}", flush=True)
        time.sleep(0.35)
        
        # 第2天
        if len(trade_days) > 1:
            r2 = ths.wencai_nlp(f"{code} {trade_days[1]} 主力资金流向,主力增仓占比,dde大单净额,融资融券余额,资金流向(万元)")
            if r2.success and r2.data:
                print(f"  day2 OK", flush=True)
            else:
                print(f"  day2 FAIL", flush=True)
            time.sleep(0.35)

print("DONE", flush=True)
