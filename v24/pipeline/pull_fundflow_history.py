"""
火力全开: 拉取 196只关注圈 历史 DDE/资金流向 (2020-01 ~ 2026-06)
每300天重连THS, 每10天flush parquet, 断点续采
"""

import sys, time, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
from thsdk import THS

BASE = Path("D:/myAI/WorkBuddy-workspace/quant-strategy")
OUT  = BASE / "data" / "raw" / "fund_flow_full" / "fundflow_history.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)
KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}

FIELDS = "主力资金流向,主力增仓占比,DDE大单净额,融资融券余额,资金流向(万元)"
FNAMES = ["main_force_net","main_force_pct","dde_net","mtss_balance","fund_flow"]
FKEYS  = ["主力资金流向","主力增仓占比","dde大单净额","融资融券余额","资金流向"]

def flush(rows):
    if not rows: return
    df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","code"])
    if OUT.exists():
        old = pd.read_parquet(str(OUT))
        df = pd.concat([old,df]).drop_duplicates(["date","code"],keep="last").sort_values(["date","code"])
    df.to_parquet(str(OUT))
    return len(df)

# Load codes
cf = pd.read_parquet(str(BASE/"data"/"processed"/"training_data_v3.parquet"))
codes = sorted(cf["code"].unique())
wc = ",".join(c.split(".")[0] for c in codes)
N = len(codes)

# Trade days
days = []
d=datetime(2020,1,1); end=datetime(2026,6,30)
while d<=end:
    if d.weekday()<5: days.append(d.strftime("%Y-%m-%d"))
    d+=timedelta(days=1)
T=len(days)

# Resume
done=set()
if OUT.exists():
    old=pd.read_parquet(str(OUT))
    done=set(old["date"].astype(str).unique())
    print(f"[RESUME] {len(old)} rows, {len(done)} dates")

rows=[]; t0=time.time()
step=300  # reconnect every 300 days

for batch_start in range(0, T, step):
    batch_days = days[batch_start:batch_start+step]
    with THS(KQ) as ths:
        for i, day in enumerate(batch_days):
            idx = batch_start + i
            if day in done: continue

            try:
                r = ths.wencai_nlp(f"{wc} {day} {FIELDS}")
            except Exception as e:
                print(f"  [{idx}/{T}] ERR {day}: {e}", flush=True)
                time.sleep(1); continue

            if r.success and r.data:
                dk=day.replace("-","")
                for rd in r.data:
                    code=rd.get("股票代码","").split(".")[0]
                    if not code: continue
                    rec={"date":day,"code":code}
                    for fk,fn in zip(FKEYS,FNAMES):
                        v=rd.get(f"{fk}[{dk}]")
                        if v is not None:
                            try: rec[fn]=float(str(v).replace(",",""))
                            except: rec[fn]=None
                        else: rec[fn]=None
                    rows.append(rec)

            if len(rows)>=10*N:
                tot=flush(rows); rows=[]
                el=time.time()-t0
                print(f"  [{idx+1}/{T}] saved {tot} rows  |  {el/60:.1f}min", flush=True)

            if (idx+1)%100==0:
                el=time.time()-t0
                print(f"  [{idx+1}/{T}] {el/60:.1f}min  |  buffer {len(rows)} rows", flush=True)

            time.sleep(0.3)

    # end of batch: flush remaining, re-enter with-block
    if rows:
        tot=flush(rows); rows=[]; print(f"  batch flush: {tot} rows", flush=True)

# Final
if rows: flush(rows)
el=time.time()-t0
final=pd.read_parquet(str(OUT))
print(f"\n[DONE] {el/60:.1f}min | {len(final)} rows, {final['date'].nunique()} dates")
print(f"  {final['date'].min()} ~ {final['date'].max()}")
cols=[c for c in final.columns if c not in ("date","code")]
print(f"  cols: {cols}")
