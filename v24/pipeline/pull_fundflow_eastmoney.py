"""
东财 push2his API — 批量拉取198只股票历史资金流日线
覆盖: 2015-01-01 ~ 2026-07-02
字段: date, main_net, small_net, medium_net, large_net, super_large_net, main_pct

输出: data/raw/fund_flow_full/{code6}.parquet
"""
import os, sys, json, time, requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = settings.DATA_DIR
OUT_DIR = DATA / "raw" / "fund_flow_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WATCH_PATH = DATA / "universe" / "watchlist.json"
with open(str(WATCH_PATH), encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]

API = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}
PARAMS = {
    "klt": "101",       # daily
    "lmt": "5000",      # max records
    "fields1": "f1,f2,f3,f7",
    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
    "ut": "b2884a393a59ad64002292a3e90d46a5",
}

def to_secid(code6: str) -> str:
    """6位代码 -> 东财secid (1.600000 / 0.000001)"""
    if code6.startswith(("60", "68", "11", "13")):
        return f"1.{code6}"
    return f"0.{code6}"

def pull_one(code6: str) -> pd.DataFrame:
    """拉单只股票资金流日线"""
    secid = to_secid(code6)
    params = {**PARAMS, "secid": secid}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=15)
            data = r.json().get("data")
            if not data or not data.get("klines"):
                if attempt < 2:
                    time.sleep(1)
                    continue
                return pd.DataFrame()
            rows = []
            for line in data["klines"]:
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                rows.append({
                    "date": parts[0],
                    "main_net": float(parts[1]) if parts[1] else None,
                    "small_net": float(parts[2]) if parts[2] else None,
                    "medium_net": float(parts[3]) if parts[3] else None,
                    "large_net": float(parts[4]) if parts[4] else None,
                    "super_large_net": float(parts[5]) if parts[5] else None,
                    "main_pct": float(parts[6]) if parts[6] else None,
                })
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            print(f"  ERR {code6}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# Main
print(f"Pulling fund flow for {len(stocks)} stocks...")
done = 0
skipped = 0
for i, s in enumerate(stocks):
    code6 = s["code"][:6]
    out = OUT_DIR / f"{code6}.parquet"
    
    # Skip if already pulled today
    if out.exists():
        old = pd.read_parquet(out)
        if len(old) > 100 and old["date"].max() >= pd.Timestamp("2026-06-25"):
            skipped += 1
            done += 1
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(stocks)}] skipped {skipped}, pulled {done-skipped}")
            continue
    
    df = pull_one(code6)
    if len(df) > 0:
        df.to_parquet(out, index=False)
    
    done += 1
    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(stocks)}] {s['name']} ({code6}): {len(df)} rows")
    
    time.sleep(0.15)  # rate limit

# Summary
files = list(OUT_DIR.glob("*.parquet"))
total_rows = sum(len(pd.read_parquet(f)) for f in files)
print(f"\nDone: {len(files)} files, {total_rows} total rows, skipped {skipped}")
