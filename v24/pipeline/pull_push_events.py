"""10jqka 快讯事件拉取 — 近1月全量 + crontab 每日追加"""
import os, json, time, subprocess
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

EVENT_DIR = r'D:\myAI\Hermes-Workspace\quant-strategy\data\raw\events'
WATCHLIST = r'D:\myAI\Hermes-Workspace\quant-strategy\data\universe\watchlist.json'
os.makedirs(EVENT_DIR, exist_ok=True)

API = 'https://news.10jqka.com.cn/tapp/news/push/stock'
date_str = datetime.now().strftime('%Y%m%d')

def fetch_page(page: int) -> list:
    """拉取单页快讯"""
    cmd = f'curl -s -H "User-Agent: Mozilla/5.0" -H "Referer: https://news.10jqka.com.cn/" "{API}?date={date_str}&page={page}"'
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=True)
        d = json.loads(r.stdout)
        items = d.get('data', {}).get('list', [])
        return items
    except:
        return []

def parse_item(item: dict) -> dict:
    """解析单条快讯"""
    ts = int(item.get('ctime', 0))
    dt = datetime.fromtimestamp(ts) if ts else None
    return {
        'id': item.get('id'),
        'seq': item.get('seq'),
        'title': item.get('title', ''),
        'digest': item.get('digest', ''),
        'url': item.get('url', ''),
        'datetime': dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None,
        'timestamp': ts,
        'source': item.get('source', ''),
        'tag': item.get('tag', ''),
        'tags': json.dumps([t.get('name','') for t in item.get('tags',[])], ensure_ascii=False),
        'stocks': json.dumps([{'code':s.get('code',''),'name':s.get('name','')} for s in item.get('stock',[])], ensure_ascii=False),
        'importance': item.get('import', '0'),
    }

# ─── 获取总页数 ───
first = fetch_page(1)
if not first:
    print('API 无响应，退出')
    exit(1)

total = len(first)  # 第一页数
# 试第 630 页看末端
last = fetch_page(630)
if last:
    # 找空页确定总页数
    for p in range(100, 700, 50):
        data = fetch_page(p)
        if not data:
            total_pages = p - 1
            break
    else:
        total_pages = 630
else:
    total_pages = 630

print(f'总页数: ~{total_pages}, 约 {total_pages * 20} 条快讯')

# ─── 全量拉取 ───
t0 = time.time()
all_items = []
pages = list(range(1, total_pages + 1))

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(fetch_page, p): p for p in pages}
    for f in as_completed(futures):
        p = futures[f]
        items = f.result()
        parsed = [parse_item(i) for i in items]
        all_items.extend(parsed)
        if p % 50 == 0:
            print(f'  page {p}/{total_pages} ({len(parsed)}条, {time.time()-t0:.0f}s)')
        time.sleep(0.1)

print(f'拉取完成: {len(all_items)} 条 ({time.time()-t0:.0f}s)')

# ─── 写入 parquet ───
out = os.path.join(EVENT_DIR, 'push_news.parquet')
df = pd.DataFrame(all_items)
# 按时间排序，去重
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp', ascending=False).drop_duplicates(subset=['id']).reset_index(drop=True)
df.to_parquet(out, index=False)
print(f'写入 {out}: {len(df)} 行')

# ─── 按股票拆分（供 feature engine 使用） ───
stock_dir = os.path.join(EVENT_DIR, 'by_stock')
os.makedirs(stock_dir, exist_ok=True)

# 解析 stocks 列
for _, row in df.iterrows():
    stocks_list = json.loads(row['stocks']) if row['stocks'] else []
    for s in stocks_list:
        code = s.get('code', '')[:6]
        if code and code.isdigit():
            rec = {'date': row['datetime'][:10] if row['datetime'] else '',
                   'time': row['datetime'],
                   'title': row['title'],
                   'digest': row['digest'],
                   'source': row['source'],
                   'event_type': 'news',
                   'severity': 'P3' if row['importance'] == '1' else 'P4',
                   'url': row['url']}
            out_stock = os.path.join(stock_dir, f'{code}_events.parquet')
            if os.path.exists(out_stock):
                existing = pd.read_parquet(out_stock)
                combined = pd.concat([existing, pd.DataFrame([rec])])
                combined = combined.drop_duplicates(subset=['date','title']).sort_values('date', ascending=False)
                combined.to_parquet(out_stock, index=False)
            else:
                pd.DataFrame([rec]).to_parquet(out_stock, index=False)

stock_count = len([f for f in os.listdir(stock_dir) if f.endswith('.parquet')])
print(f'按股票拆分: {stock_count} 只有事件数据')
print(f'完成! 总耗时 {time.time()-t0:.0f}s')
