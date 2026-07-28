"""
Pull margin trade data for 49 new stocks (eastmoney datacenter API)
"""
import sys, time, requests, json, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

BASE = Path("D:/myAI/WorkBuddy-workspace/quant-strategy")
OUT = BASE / "data" / "raw" / "MainNetFlow" / "oos_margintrade.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

KEY_COLS = ["rzye","rzche","rzmre","rzjme","rqye","rqyl","rqmcl","rqchl","rqjmg","rzrqye","spj","zdf","fin_balance_gr"]

new_codes = ['002445','000695','300915','300643','300449','002708','002363','688048','002222',
             '688561','600903','000788','000757','002284','300409','300534','603656','688686',
             '000712','688206','300220','920964','688039','600874','300423','601828','688549',
             '300942','000301','002899','600917','301696','300940','002862','300376','301608',
             '002361','002275','600463','002312','600207','600036','688702','300824','000935',
             '603040','605268','002573','600439']

def pull_one(code):
    rows = []
    page = 1
    while True:
        params = {
            "reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
            "filter": f'(SCODE="{code}")', "pageNumber": str(page), "pageSize": "500",
            "sortColumns": "DATE", "sortTypes": "-1", "source": "WEB", "client": "WEB",
        }
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
            d = r.json()
        except:
            break
        if d.get("result") and d["result"].get("data"):
            data = d["result"]["data"]
            for row in data:
                rec = {"code": code, "date": str(row.get("DATE",""))[:10]}
                for k in KEY_COLS:
                    v = row.get(k.upper())
                    if v is not None:
                        try: rec[k] = float(v)
                        except: rec[k] = 0.0
                    else:
                        rec[k] = 0.0
                rows.append(rec)
            if len(data) < 500:
                break
            page += 1
            time.sleep(0.3)
        else:
            break
    return rows

all_rows = []
t0 = time.time()
for i, code in enumerate(new_codes):
    rows = pull_one(code)
    if rows:
        all_rows.extend(rows)
        el = time.time()-t0
        print(f"  [{i+1}/{len(new_codes)}] {code}: {len(rows)} rows  |  {el/60:.1f}min", flush=True)
    else:
        print(f"  [{i+1}/{len(new_codes)}] {code}: empty", flush=True)
    time.sleep(0.5)

if all_rows:
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","code"])
    df.to_parquet(str(OUT), index=False)
    el = time.time()-t0
    print(f"\nSaved: {OUT} ({len(df)} rows, {df['code'].nunique()} stocks, {el/60:.1f}min)")
