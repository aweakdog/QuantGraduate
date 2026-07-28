"""每日收盘后批量拉取：1min K线 + 主力资金 + 事件"""
import requests, time, os, subprocess, pandas as pd, json
from datetime import datetime

BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy/data'
TODAY = datetime.now().strftime('%Y-%m-%d')
TODAY8 = datetime.now().strftime('%Y%m%d')
THS_PY = "C:/Users/admin/.workbuddy/binaries/python/envs/ths/Scripts/python.exe"
TMP = os.environ.get('TEMP', 'C:/tmp')

for d in ['kline_1min','fund_flow_daily','events_daily']:
    os.makedirs(f'{BASE}/raw/{d}', exist_ok=True)
os.makedirs(f'{BASE}/raw/sectors_daily', exist_ok=True)

stocks_df = pd.read_parquet(f'{BASE}/universe/stock_list.parquet')
all_codes = sorted(stocks_df['股票代码'].str[:6].unique())
all_codes_str = json.dumps(all_codes)
print(f'All A-share stocks: {len(all_codes)}')

# ─── 1. 1分钟K线 via thsdk KQ2026 ───
print('1min K-line: via thsdk KQ2026...')
try:
    r = subprocess.run([THS_PY, 'pipeline/pull_1min_thsdk.py'], capture_output=True, text=True, timeout=300)
    print(r.stdout[-300:] if r.stdout else '')
    if r.returncode: print(f'  ERR: {r.stderr[-200:]}')
except Exception as e:
    print(f'  SKIP (timeout/error): {e}')

# ─── 2. Fund flow ───
print('\nFund flow: via thsdk...')
fund_flow_py = f"""{TMP}/daily_fundflow_{TODAY8}.py"""
with open(fund_flow_py, 'w') as f:
    f.write(f'''
import pandas as pd, time, json, os
from thsdk import THS
KQ = {{'username': os.environ.get('THS_USERNAME', ''), 'password': os.environ.get('THS_PASSWORD', '')}}
codes = {all_codes_str}
BATCH=50; FIELDS='主力资金流向,主力增仓占比,DDE大单净额,融资融券余额,资金流向(万元)'
FNAMES=['main_force_net','main_force_pct','dde_net','mtss_balance','fund_flow']
FKEYS=['主力资金流向','主力增仓占比','dde大单净额','融资融券余额','资金流向']
rows=[]; TD='{TODAY}'; TD8='{TODAY8}'
with THS(KQ) as ths:
    for i in range(0,len(codes),BATCH):
        wc=','.join(codes[i:i+BATCH])
        try:
            r=ths.wencai_nlp(f'{{wc}} {{TD}} {{FIELDS}}')
            if r.success and r.data:
                for rd in r.data:
                    c=rd.get('股票代码','').split('.')[0]
                    if not c: continue
                    rec={{"date":TD,"code":c}}
                    for fk,fn in zip(FKEYS,FNAMES):
                        v=rd.get(f'{{fk}}[{{TD8}}]')
                        if v is not None:
                            try: rec[fn]=float(str(v).replace(',',''))
                            except: rec[fn]=None
                        else: rec[fn]=None
                    rows.append(rec)
        except: pass
        time.sleep(0.3)
if rows:
    pd.DataFrame(rows).to_parquet('{BASE}/raw/fund_flow_daily/{TODAY8}.parquet',index=False)
    print(f'OK:{{len(rows)}}')
else: print('NO_DATA')
''')
r1 = subprocess.run([THS_PY, fund_flow_py], capture_output=True, text=True, timeout=300)
print(f'  Fund flow: {r1.stdout.strip()[-80:]}')
if r1.returncode: print(f'  ERR: {r1.stderr[-200:]}')
os.remove(fund_flow_py)

# ─── 3. Events ───
print('\nEvents: via thsdk...')
events_py = f"""{TMP}/daily_events_{TODAY8}.py"""
with open(events_py, 'w') as f:
    f.write(f'''
import pandas as pd, json, os
from thsdk import THS
KQ = {{"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}}
codes = {all_codes_str}
wc=",".join(codes); TD="{TODAY}"
rows=[]
with THS(KQ) as ths:
    r=ths.wencai_nlp(f"{{wc}} 重大事件")
    if r.success and r.data:
        for d in r.data:
            c=d.get("股票代码","").split(".")[0]
            enames=str(d.get("重要事件名称","")).split("|")
            etimes=str(d.get("重要事件公告时间","")).split("|")
            for en, et in zip(enames, etimes):
                en=en.strip()
                if not en or en=="None": continue
                rows.append({{"code":c,"date":TD,"event_type":en[:50],"event_time":et.strip()[:10]}})
if rows:
    pd.DataFrame(rows).to_parquet("{BASE}/raw/events_daily/{TODAY8}.parquet",index=False)
    print(f"OK:{{len(rows)}}")
else: print("NO_DATA")
''')
r2 = subprocess.run([THS_PY, events_py], capture_output=True, text=True, timeout=180)
print(f'  Events: {r2.stdout.strip()[-80:]}')
if r2.returncode: print(f'  ERR: {r2.stderr[-200:]}')
os.remove(events_py)

# ─── 4. 板块/概念数据 (日线级别) ───
print('\nSectors: via thsdk...')
sectors_py = f"""{TMP}/daily_sectors_{TODAY8}.py"""
with open(sectors_py, 'w') as f:
    f.write(f'''
import pandas as pd, json, time, os
from thsdk import THS
KQ = {{"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}}
BASE = "{BASE}"; TD = "{TODAY}"; TD8 = "{TODAY8}"
os.makedirs(f"{{BASE}}/raw/sectors_daily", exist_ok=True)

with THS(KQ) as ths:
    # Concepts
    rc = ths.ths_concept()
    if rc.success and rc.data:
        concepts = [{{"code":d["代码"],"name":d["名称"],"type":"concept"}} for d in rc.data]
        pd.DataFrame(concepts).to_parquet(f"{{BASE}}/raw/sectors_daily/concepts_{{TD8}}.parquet", index=False)
        print(f"Concepts: {{len(concepts)}}")
    
    time.sleep(0.3)
    
    # Industries
    ri = ths.ths_industry()
    if ri.success and ri.data:
        industries = [{{"code":d["代码"],"name":d["名称"],"type":"industry"}} for d in ri.data]
        pd.DataFrame(industries).to_parquet(f"{{BASE}}/raw/sectors_daily/industries_{{TD8}}.parquet", index=False)
        print(f"Industries: {{len(industries)}}")
    
    time.sleep(0.3)
    
    # Top sectors by fund flow
    for stype, sname in [("concept","概念"),("industry","行业")]:
        try:
            r = ths.wencai_nlp(f"今日{{sname}}板块资金流向排名前20")
            if r.success and r.data:
                pd.DataFrame(r.data).to_parquet(f"{{BASE}}/raw/sectors_daily/{{stype}}_flow_{{TD8}}.parquet", index=False)
                print(f"{{stype}} flow: {{len(r.data)}}")
        except: pass
        time.sleep(0.3)
''')
r3 = subprocess.run([THS_PY, sectors_py], capture_output=True, text=True, timeout=180)
print(f'  Sectors: {r3.stdout.strip()[-200:]}')
if r3.returncode: print(f'  ERR: {r3.stderr[-200:]}')
os.remove(sectors_py)

print(f'\nDone: {TODAY}')
