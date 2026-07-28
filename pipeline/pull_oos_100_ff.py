"""Pull fund flow for 100 OOS ChiNext/STAR stocks - test period only"""
import sys, time, json, os
import pandas as pd
from datetime import datetime, timedelta
from thsdk import THS

KQ = {'username': os.environ.get('THS_USERNAME', ''), 'password': os.environ.get('THS_PASSWORD', '')}
BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy'

with open(f'{BASE}/data/universe/oos_100_list.json') as f:
    codes = json.load(f)

wc = ','.join(codes)
OUT = f'{BASE}/data/raw/fund_flow_full/oos_100_ff.parquet'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Test period only: 2025-03-01 ~ 2026-07-03
days = []
d = datetime(2025, 3, 1)
end = datetime(2026, 7, 3)
while d <= end:
    if d.weekday() < 5:
        days.append(d.strftime('%Y-%m-%d'))
    d += timedelta(days=1)

print(f"Pulling fund flow for {len(codes)} stocks x {len(days)} days")

FIELDS = '主力资金流向,主力增仓占比,DDE大单净额,资金流向(万元)'
FNAMES = ['main_force_net', 'main_force_pct', 'dde_net', 'fund_flow']
FKEYS = ['主力资金流向', '主力增仓占比', 'dde大单净额', '资金流向']

all_rows = []
t0 = time.time()
with THS(KQ) as ths:
    for idx, day in enumerate(days):
        try:
            r = ths.wencai_nlp(f'{wc} {day} {FIELDS}')
        except:
            time.sleep(0.5)
            continue
        if r.success and r.data:
            dk = day.replace('-', '')
            for rd in r.data:
                code = rd.get('股票代码', '').split('.')[0]
                if not code:
                    continue
                rec = {'date': day, 'code': code}
                for fk, fn in zip(FKEYS, FNAMES):
                    v = rd.get(f'{fk}[{dk}]')
                    if v is not None:
                        try:
                            rec[fn] = float(str(v).replace(',', ''))
                        except:
                            rec[fn] = None
                    else:
                        rec[fn] = None
                all_rows.append(rec)
        time.sleep(0.3)
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  [{idx+1}/{len(days)}] {(idx+1)/elapsed*60:.0f}day/min, {len(all_rows)} rows', flush=True)

if all_rows:
    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'])
    codes_pulled = df['code'].nunique()
    df.to_parquet(OUT, index=False)
    print(f'\nSaved: {OUT} ({len(df)} rows, {codes_pulled}/{len(codes)} stocks)')
else:
    print('No data pulled')
