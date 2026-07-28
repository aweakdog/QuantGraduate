"""使用 thsdk KQ2026 正式账号拉取全量1分钟K线"""
import sys, json, time, os
sys.path.insert(0, 'C:/Users/admin/.workbuddy/skills/ths-all-in-one/scripts')
from thsdk import THS
import pandas as pd

BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy/data'
from datetime import datetime
TODAY = datetime.now().strftime('%Y-%m-%d')
TODAY8 = datetime.now().strftime('%Y%m%d')

# Target codes: ALL A-share stocks
sl = pd.read_parquet(f'{BASE}/universe/stock_list.parquet')
all_stocks = sorted(sl['股票代码'].str[:6].unique())
print(f'All A-share stocks: {len(all_stocks)}')

indices = {'000001':'上证指数','399001':'深证成指','399006':'创业板指','000688':'科创50','000300':'沪深300','000016':'上证50'}
etfs = {'510050':'上证50ETF','510300':'300ETF','510500':'500ETF','159915':'创业板ETF','588000':'科创50ETF',
        '512480':'半导体ETF','159845':'中证1000ETF','512010':'医药ETF','515000':'科技ETF','159949':'创业板50ETF','512880':'证券ETF'}

targets = {}
for c in all_stocks: targets[c] = c

# Add indices
for code, name in [('000001','上证指数'),('399001','深证成指'),('399006','创业板指'),
    ('000688','科创50'),('000300','沪深300'),('000016','上证50')]:
    targets[code] = name

# Add ETFs (initial set)
etf_basics = {'510050':'上证50ETF','510300':'300ETF','510500':'500ETF','159915':'创业板ETF','588000':'科创50ETF',
        '512480':'半导体ETF','159845':'中证1000ETF','512010':'医药ETF','515000':'科技ETF','159949':'创业板50ETF','512880':'证券ETF'}
for c,n in etf_basics.items(): targets[c] = n

print(f'Initial targets: {len(targets)}')

KQ = {'username': os.environ.get('THS_USERNAME', ''), 'password': os.environ.get('THS_PASSWORD', '')}
all_rows = []
t0 = time.time()

with THS(KQ) as ths:
    # Get all ETFs and add to targets
    etf_resp = ths.fund_etf_lists()
    if etf_resp.success and etf_resp.data:
        etc = 0
        for etf in etf_resp.data:
            code = etf.get('代码','')
            name = etf.get('名称','')
            if code and code.startswith('USZJ'):
                code6 = code[4:]
                if code6 not in targets:
                    targets[code6] = name
                    etc += 1
        print(f'  ETFs added: {etc}')
    print(f'Final targets: {len(targets)}')
    for i, (code, name) in enumerate(targets.items()):
        # search_symbols first to get THSCODE
        sym = ths.search_symbols(code)
        time.sleep(0.05)  # respect thsdk rate limit
        if not sym.success or not sym.data:
            continue
        # Prefer A-share
        cand = [d for d in sym.data if d.get('MarketStr','').startswith(('USZA','USHA','UBJA'))]
        if not cand: cand = sym.data
        ths_code = cand[0].get('THSCODE', '')

        # Get 1-minute klines (latest ~240 bars = 1 trading day)
        k = ths.klines(ths_code, count=240, interval='1m')
        if k.success and k.data:
            for row in k.data:
                all_rows.append({
                    'code': name,
                    'time': str(row.get('时间',''))[11:16],
                    'open': row.get('开盘价'),
                    'close': row.get('收盘价'),
                    'high': row.get('最高价'),
                    'low': row.get('最低价'),
                    'vol': row.get('成交量',0),
                    'amount': row.get('总金额',0),
                    'date': TODAY,
                })
        
        elapsed = time.time() - t0
        if (i+1) % 30 == 0:
            done = len(set(r['code'] for r in all_rows))
            print(f'  [{i+1}/{len(targets)}] {done} done, {len(all_rows)} bars, {elapsed:.0f}s', flush=True)

if all_rows:
    df = pd.DataFrame(all_rows)
    out = f'{BASE}/raw/kline_1min/{TODAY8}.parquet'
    df.to_parquet(out, index=False)
    print(f'\nSaved: {out}')
    print(f'Bars: {len(df)}, Targets: {df["code"].nunique()}, Time: {time.time()-t0:.0f}s')
else:
    print('No data')
