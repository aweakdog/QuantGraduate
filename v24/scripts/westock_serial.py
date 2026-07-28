"""逐个股票拉取westock主力资金流向（单线程防限流）"""
import os, json, subprocess, pandas as pd, time

FF_DIR = r'D:\myAI\Hermes-Workspace\quant-strategy\data\raw\fund_flow'
WATCHLIST = r'D:\myAI\Hermes-Workspace\quant-strategy\data\universe\watchlist.json'

with open(WATCHLIST) as f:
    stocks = json.load(f)['watchlist']

# 最近 10 天验证管道
DATES = ['2026-06-30','2026-06-29','2026-06-26','2026-06-25','2026-06-24',
         '2026-06-23','2026-06-22','2026-06-19','2026-06-18','2026-06-17']

def parse_line(line, codes_map):
    """解析单行，返回 (c6, date_str, main_force_net) 或 None"""
    parts = [p.strip() for p in line.split('|')]
    for i, p in enumerate(parts):
        if p in codes_map:
            nf = parts[i+12] if len(parts) > i+12 else ''
            nf = nf.replace(',', '')
            if nf.lstrip('-').replace('.', '').isdigit():
                return codes_map[p], parts[i+4], float(nf)
    return None

# 预计算 code -> c6 映射
codes_map = {}
for s in stocks:
    c6 = s['code'][:6]
    mkt = {'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj'}.get(s['code'][7:], 'sh')
    codes_map[f'{mkt}{c6}'] = c6

t0 = time.time()
done = 0
total_new = 0

for s in stocks:
    c6 = s['code'][:6]
    mkt = {'SH': 'sh', 'SZ': 'sz', 'BJ': 'bj'}.get(s['code'][7:], 'sh')
    code_key = f'{mkt}{c6}'
    path = os.path.join(FF_DIR, f'{c6}.parquet')

    # 已有数据的跳过
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df['date'] = df['date'].astype(str)
        existing = set(df['date'].values)
    else:
        existing = set()

    # 只拉缺失的日期
    needed = [d for d in DATES if d.replace('-', '') not in existing]
    if not needed:
        done += 1
        if done % 20 == 0:
            print(f'{c6}: 已全部 ({time.time()-t0:.0f}s)')
        continue

    recs = []
    for date in needed:
        cmd = f'npx westock-data-clawhub@1.0.4 asfund {code_key} --date {date}'
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, shell=True)
            for line in r.stdout.split('\n'):
                result = parse_line(line, codes_map)
                if result and result[0] == c6:
                    recs.append({'date': date.replace('-', ''), 'main_force_net': result[2]})
                    break
        except:
            pass
        time.sleep(0.5)

    if recs:
        df_new = pd.DataFrame(recs)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['date'] = df['date'].astype(str)
            df_new['date'] = df_new['date'].astype(str)
            combined = pd.concat([df, df_new]).drop_duplicates(subset='date').sort_values('date')
            combined.to_parquet(path, index=False)
        else:
            df_new.to_parquet(path, index=False)
        total_new += len(recs)

    done += 1
    elapsed = time.time() - t0
    if done % 10 == 0:
        files_ok = len([f for f in os.listdir(FF_DIR) if f.endswith('.parquet')])
        print(f'{done}/200: {c6} +{len(recs)}行 -> {files_ok}只 ({elapsed:.0f}s)')

total = len([f for f in os.listdir(FF_DIR) if f.endswith('.parquet')])
print(f'\n完成! +{total_new}行, 共{total}/200只 ({time.time()-t0:.0f}s)')
