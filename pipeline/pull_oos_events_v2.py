"""用 thsdk 批量拉新49只的 重大事件+关键词资讯"""
import sys, time, base64, json, os
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE, "data", "raw", "events_ifind")

new_codes = ['002445','000695','300915','300643','300449','002708','002363','688048','002222',
             '688561','600903','000788','000757','002284','300409','300534','603656','688686',
             '000712','688206','300220','920964','688039','600874','300423','601828','688549',
             '300942','000301','002899','600917','301696','300940','002862','300376','301608',
             '002361','002275','600463','002312','600207','600036','688702','300824','000935',
             '603040','605268','002573','600439']

from thsdk import THS
KQ = {'username': os.environ.get('THS_USERNAME', ''), 'password': os.environ.get('THS_PASSWORD', '')}

def parse_date(s):
    if not s or len(s) < 8:
        return None
    try:
        if len(s) >= 10:
            return datetime.strptime(s[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
        return datetime.strptime(s[:8], '%Y%m%d').strftime('%Y-%m-%d')
    except:
        return None

def map_event(ename):
    e = ename.lower()
    if '涨停' in e: return ('limit_up', 1, 2)
    if '跌停' in e: return ('limit_down', -1, 2)
    if '解禁' in e: return ('unlock', -1, 2)
    if '增持' in e: return ('increase', 1, 2)
    if '减持' in e: return ('reduction', -1, 2)
    if '回购' in e: return ('buyback_plan', 1, 3)
    if '中标' in e or '合同' in e: return ('big_contract', 1, 2)
    if '立案' in e or '处罚' in e: return ('lawsuit', -1, 0)
    if '预增' in e: return ('earnings_revise', 1, 2)
    return ('unknown', 0, 3)

all_events = []

with THS(KQ) as ths:
    for idx, c in enumerate(new_codes):
        # Pull 重大事件
        r = ths.wencai_nlp(f'{c} 重大事件')
        if r.success and r.data:
            for d in r.data:
                for ename, etime in [(n.strip(), t.strip()) for n, t in 
                    zip(
                        str(d.get('重要事件名称','')).split('|'),
                        str(d.get('重要事件公告时间','')).split('|')
                    )]:
                    if not ename or ename == 'None': continue
                    e_date = parse_date(etime)
                    if not e_date: continue
                    etype, dir_h, pl = map_event(ename)
                    all_events.append({
                        'code': c, 'date': e_date, 'event_type': etype,
                        'p_level': pl, 'direction': dir_h, 'title': ename
                    })
        
        time.sleep(0.25)
        
        # Pull 关键词资讯
        r2 = ths.wencai_nlp(f'{c} 关键词资讯')
        if r2.success and r2.data:
            for d in r2.data:
                raw = d.get('关键词资讯','')
                if not raw or raw == 'None': continue
                try:
                    decoded = base64.b64decode(raw).decode('utf-8')
                except: continue
                try:
                    items = json.loads(decoded)
                except: continue
                for item in items:
                    pub = item.get('PublishTime','')
                    title = item.get('PageRawTitle','')
                    e_date = parse_date(str(pub)[:10]) or parse_date(str(pub)[:8])
                    if not e_date or not title: continue
                    all_events.append({
                        'code': c, 'date': e_date,
                        'event_type': 'keyword_news', 'p_level': 3, 'direction': 0,
                        'title': title
                    })
        
        time.sleep(0.25)
        
        if (idx+1) % 10 == 0:
            print(f'  [{idx+1}/{len(new_codes)}] {len(all_events)} events')

df = pd.DataFrame(all_events)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code','date']).drop_duplicates(subset=['code','date','event_type'], keep='last').reset_index(drop=True)
print(f'\nTotal: {len(df)} rows, {df["code"].nunique()} stocks')
print(f'Types: {df["event_type"].value_counts().to_dict()}')

# Merge with events_v2
ev_v2 = pd.read_parquet(os.path.join(OUT_DIR, 'events_v2.parquet'))
combined = pd.concat([ev_v2, df], ignore_index=True)
combined = combined.drop_duplicates(subset=['code','date','event_type'], keep='last')
combined = combined.sort_values(['code','date']).reset_index(drop=True)
combined.to_parquet(os.path.join(OUT_DIR, 'events_v2.parquet'), index=False)
print(f'Merged events_v2: {combined.shape}, {combined["code"].nunique()} codes')
