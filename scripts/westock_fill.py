"""填充缺失的主力资金流向数据（westock-data）"""
import os, json, subprocess, pandas as pd, time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

FF_DIR = r'D:\myAI\Hermes-Workspace\quant-strategy\data\raw\fund_flow'
WATCHLIST = r'D:\myAI\Hermes-Workspace\quant-strategy\data\universe\watchlist.json'

# 最近 60 个交易日
TRADING_DATES = [
    '2026-06-30','2026-06-29','2026-06-26','2026-06-25','2026-06-24',
    '2026-06-23','2026-06-22','2026-06-19','2026-06-18','2026-06-17',
    '2026-06-16','2026-06-15','2026-06-12','2026-06-11','2026-06-10',
    '2026-06-09','2026-06-08','2026-06-05','2026-06-04','2026-06-03',
    '2026-06-02','2026-06-01','2026-05-29','2026-05-28','2026-05-27',
    '2026-05-26','2026-05-25','2026-05-22','2026-05-21','2026-05-20',
    '2026-05-19','2026-05-18','2026-05-15','2026-05-14','2026-05-13',
    '2026-05-12','2026-05-11','2026-05-08','2026-05-07','2026-05-06',
    '2026-04-30','2026-04-29','2026-04-28','2026-04-27','2026-04-24',
    '2026-04-23','2026-04-22','2026-04-21','2026-04-20','2026-04-17',
    '2026-04-16','2026-04-15','2026-04-14','2026-04-11','2026-04-10',
    '2026-04-09','2026-04-08','2026-04-07','2026-04-03','2026-04-02',
]

def fetch_stock(s):
    """单只股票全部日期拉取，已存在的跳过"""
    c6 = s['code'][:6]
    mkt = {'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj'}.get(s['code'][7:], 'sh')
    path = os.path.join(FF_DIR, f'{c6}.parquet')

    # 已有数据的追加缺失日期
    existing_set = set()
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df['date'] = df['date'].astype(str)
        existing_set = set(df['date'].values)
    
    needed = [d for d in TRADING_DATES if d.replace('-', '') not in existing_set]
    if not needed:
        return c6, len(existing_set), '已有全部'

    recs = []
    for date in needed:
        try:
            r = subprocess.run(
                f'npx westock-data-clawhub@1.0.4 asfund {mkt}{c6} --date {date}',
                capture_output=True, text=True, timeout=15, shell=True
            )
            for line in r.stdout.split('\n'):
                parts = [p.strip() for p in line.split('|')]
                for i, p in enumerate(parts):
                    if p == f'{mkt}{c6}':
                        nf = parts[i + 12] if len(parts) > i + 12 else ''
                        nf = nf.replace(',', '')
                        if nf.lstrip('-').replace('.', '').isdigit():
                            recs.append({'date': date.replace('-', ''), 'main_force_net': float(nf)})
                        break
        except:
            pass
        time.sleep(0.3)

    if recs:
        df_new = pd.DataFrame(recs)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['date'] = df['date'].astype(str)
            df_new['date'] = df_new['date'].astype(str)
            # 合并去重
            combined = pd.concat([df, df_new]).drop_duplicates(subset='date').sort_values('date')
            combined.to_parquet(path, index=False)
            return c6, len(combined), f'+{len(recs)}行'
        else:
            df_new.to_parquet(path, index=False)
            return c6, len(recs), f'新建{len(recs)}行'
    return c6, 0, '无数据'


if __name__ == '__main__':
    with open(WATCHLIST) as f:
        stocks = json.load(f)['watchlist']
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_stock, s): s['code'] for s in stocks}
        done, _ = wait(futures.keys())
        for f in done:
            c, n, msg = f.result()
            if n:
                print(f'{c}: {n}行 {msg} ({time.time()-t0:.0f}s)')

    files = [f for f in os.listdir(FF_DIR) if f.endswith('.parquet')]
    print(f'\n总计: {len(files)}/200 文件 ({time.time()-t0:.0f}s)')
