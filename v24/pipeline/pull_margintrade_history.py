"""
火力全开: 拉取 196只关注圈 历史融资融券明细

数据源: datacenter-web.eastmoney.com (已验证可用)
策略:
  - 每只股票单次查询, pageSize=500
  - 翻页直到拉完所有历史
  - 输出: raw/MainNetFlow/margintrade_history.parquet

关键字段:
  RZYE(融资余额), RZCHE(融资偿还), RZMRE(融资买入), RZJME(融资净买入)
  RQYE(融券余额), RQMCL(融券卖出), RQCHL(融券偿还), RQJMG(融券净卖出)
  RZRQYE(两融余额), SPJ(收盘价), ZDF(涨跌幅)
"""

import sys, time, requests, json
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd

BASE = Path("D:/myAI/WorkBuddy-workspace/quant-strategy")
OUT  = BASE / "data" / "raw" / "MainNetFlow" / "margintrade_history.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

KEY_COLS = ["DATE","SCODE","SECNAME","SPJ","ZDF","RZYE","RZCHE","RZMRE","RZJME",
            "RQYE","RQYL","RQMCL","RQCHL","RQJMG","RZRQYE","FIN_BALANCE_GR"]

def pull_one(code: str, name: str) -> list:
    """Pull full margin trade history for one stock"""
    rows = []
    page = 1
    while True:
        params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX",
            "columns": "ALL",
            "filter": f'(SCODE="{code}")',
            "pageNumber": str(page),
            "pageSize": "500",
            "sortColumns": "DATE",
            "sortTypes": "-1",
            "source": "WEB",
            "client": "WEB",
        }
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
            d = r.json()
        except Exception as e:
            print(f"    ⚠ {code} page {page}: {e}")
            time.sleep(2)
            break

        if d.get("result") and d["result"].get("data"):
            data = d["result"]["data"]
            rows.extend(data)
            if len(data) < 500:
                break  # last page
            page += 1
            time.sleep(0.3)
        else:
            break
    return rows

def main():
    # Load stock codes
    cf = pd.read_parquet(str(BASE / "data" / "processed" / "training_data_v3.parquet"))
    codes = sorted(cf["code"].unique())
    print(f"[STOCKS] {len(codes)}")

    # Resume check
    done_codes = set()
    if OUT.exists():
        old = pd.read_parquet(str(OUT))
        done_codes = set(old["code"].str.split(".").str[0].unique())
        print(f"[RESUME] {len(old)} rows, {len(done_codes)} codes done")

    all_rows = []
    total = len(codes)
    t0 = time.time()

    for i, code_full in enumerate(codes):
        code = code_full.split(".")[0]
        if code in done_codes:
            continue

        rows = pull_one(code, code_full)
        if not rows:
            print(f"  [{i+1}/{total}] ⚠ {code}: empty")
            continue

        # Standardize: select key columns
        for row in rows:
            rec = {"code": code, "date": str(row.get("DATE",""))[:10]}
            for k in KEY_COLS:
                if k in ("DATE","SCODE","SECNAME"):
                    continue
                v = row.get(k)
                if v is not None:
                    try:
                        rec[k.lower()] = float(v)
                    except:
                        rec[k.lower()] = 0.0
                else:
                    rec[k.lower()] = 0.0
            all_rows.append(rec)

        if len(all_rows) >= 10000:
            flush(all_rows)
            all_rows = []

        elapsed = time.time() - t0
        print(f"  [{i+1}/{total}] ✅ {code} {len(rows)} rows  |  {elapsed/60:.1f}min", flush=True)
        time.sleep(0.5)  # rate limit

    if all_rows:
        flush(all_rows)

    elapsed = time.time() - t0
    final = pd.read_parquet(str(OUT))
    print(f"\n[DONE] {elapsed/60:.1f}min | {len(final)} rows, {final['code'].nunique()} stocks")
    print(f"  date: {final['date'].min()} ~ {final['date'].max()}")

def flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","code"])
    if OUT.exists():
        old = pd.read_parquet(str(OUT))
        df = pd.concat([old,df]).drop_duplicates(["date","code"],keep="last").sort_values(["date","code"])
    df.to_parquet(str(OUT))
    print(f"  flushed {len(df)} total rows", flush=True)

if __name__ == "__main__":
    main()
