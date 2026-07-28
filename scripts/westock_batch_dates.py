"""使用westock批量查询（按日期）拉取主力资金流向"""
import os, json, subprocess, pandas as pd, time

FF_DIR = r'D:\myAI\Hermes-Workspace\quant-strategy\data\raw\fund_flow'
WATCHLIST = r'D:\myAI\Hermes-Workspace\quant-strategy\data\universe\watchlist.json'

with open(WATCHLIST) as f:
    stocks = json.load(f)['watchlist']

# 按代码批量分组（10只/组）
batches = []
for i in range(0, len(stocks), 10):
    batch = stocks[i:i+10]
    codes = []
    for s in batch:
        c6 = s['code'][:6]
        mkt = {'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj'}.get(s['code'][7:], 'sh')
        codes.append(f'{mkt}{c6}')
    batches.append((codes, batch))

print(f'共 {len(batches)} 个批次，每批 10 只')

# 最近 60 个交易日
DATES = ['2026-06-30','2026-06-29','2026-06-26','2026-06-25','2026-06-24',
         '2026-06-23','2026-06-22','2026-06-19','2026-06-18','2026-06-17',
         '2026-06-16','2026-06-15','2026-06-12','2026-06-11','2026-06-10',
         '2026-06-09','2026-06-08','2026-06-05','2026-06-04','2026-06-03',
         '2026-06-02','2026-06-01','2026-05-29','2026-05-28','2026-05-27',
         '2026-05-26','2026-05-25','2026-05-22','2026-05-21','2026-05-20']

t0 = time.time()
total_fetched = 0

for date in DATES:
    batch_new = 0
    for codes_grp, stock_grp in batches:
        cmd = f'npx westock-data-clawhub@1.0.4 asfund {",".join(codes_grp)} --date {date}'
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=True)
        except:
            continue

        # 解析所有行
        for line in r.stdout.split('\n'):
            parts = [p.strip() for p in line.split('|')]
            for i, p in enumerate(parts):
                if not p.startswith('sz') and not p.startswith('sh') and not p.startswith('bj'):
                    continue
                if p not in [f'{mkt}{s["code"][:6]}' for s in stocks for mkt in ['sh','sz','bj']]:
                    continue
                # 找到了！code 在 parts[i]
                c6 = p[2:]  # sz002463 -> 002463
                nf = parts[i+12] if len(parts) > i+12 else ''
                nf_clean = nf.replace(',', '')
                if not nf_clean.lstrip('-').replace('.', '').isdigit():
                    continue
                val = float(nf_clean)
                path = os.path.join(FF_DIR, f'{c6}.parquet')
                date_fmt = date.replace('-', '')
                # 追加
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df['date'] = df['date'].astype(str)
                    if date_fmt not in df['date'].values:
                        df_new = pd.DataFrame([{'date': date_fmt, 'main_force_net': val}])
                        combined = pd.concat([df, df_new]).sort_values('date')
                        combined.to_parquet(path, index=False)
                        batch_new += 1
                else:
                    pd.DataFrame([{'date': date_fmt, 'main_force_net': val}]).to_parquet(path, index=False)
                    batch_new += 1

        time.sleep(1.5)  # 避免限流

    total_fetched += batch_new
    files = len([f for f in os.listdir(FF_DIR) if f.endswith('.parquet')])
    elapsed = time.time() - t0
    print(f'{date}: +{batch_new}条, 累计 {files}/200 只 ({elapsed:.0f}s)')

total_time = time.time() - t0
print(f'\n完成! {total_fetched} 条数据, {total_time:.0f}s')
