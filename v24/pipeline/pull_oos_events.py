"""快速拉取新49只的 重大事件 从 thsdk wencai（~30s完成）"""
import sys, time, json, base64, os
import pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASE = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(BASE, "data", "raw", "events_ifind", "oos_events.parquet")

new_codes = ['002445','000695','300915','300643','300449','002708','002363','688048','002222',
             '688561','600903','000788','000757','002284','300409','300534','603656','688686',
             '000712','688206','300220','920964','688039','600874','300423','601828','688549',
             '300942','000301','002899','600917','301696','300940','002862','300376','301608',
             '002361','002275','600463','002312','600207','600036','688702','300824','000935',
             '603040','605268','002573','600439']

wc = ','.join(new_codes)

from thsdk import THS
KQ = {'username': os.environ.get('THS_USERNAME', ''), 'password': os.environ.get('THS_PASSWORD', '')}

events_rows = []
with THS(KQ) as ths:
    r = ths.wencai_nlp(f'{wc} 重大事件')
    if r.success and r.data:
        for d in r.data:
            code_full = d.get('股票代码','')
            code6 = code_full[:6]
            ev_names = d.get('重要事件名称','')
            ev_times = d.get('重要事件公告时间','')
            ev_content = d.get('重要事件内容','')
            
            # These can be pipe-separated or single values
            if not ev_names or ev_names == 'None':
                continue
            
            names_list = [ev_names] if isinstance(ev_names, str) else ev_names
            times_list = [ev_times] if isinstance(ev_times, str) else ev_times
            content_list = [ev_content] if isinstance(ev_content, str) else ev_content
            
            # Handle both single event and multiple events
            if isinstance(ev_names, str) and '|' in ev_names:
                names_list = ev_names.split('|')
                times_list = ev_times.split('|') if isinstance(ev_times, str) and '|' in ev_times else [ev_times]*len(names_list)
                content_list = ev_content.split('|') if isinstance(ev_content, str) and '|' in ev_content else [ev_content]*len(names_list)
            
            for i in range(len(names_list)):
                ename = names_list[i].strip() if i < len(names_list) else ''
                etime = times_list[i].strip() if i < len(times_list) else ''
                econtent = content_list[i].strip() if i < len(content_list) else ''
                
                if not ename:
                    continue
                
                # Parse date from event time (format: YYYYMMDD or YYYYMMDD HHMMSS)
                e_date = None
                if etime and len(etime) >= 8:
                    try:
                        e_date = datetime.strptime(etime[:8], '%Y%m%d').strftime('%Y-%m-%d')
                    except:
                        pass
                
                if not e_date:
                    continue
                
                # Map event type
                etype = 'unknown'
                direction = 0
                p_level = 3
                
                ename_lower = ename.lower()
                if '涨停' in ename: etype, direction, p_level = 'limit_up', 1, 2
                elif '跌停' in ename: etype, direction, p_level = 'limit_down', -1, 2
                elif '限售解禁' in ename or '解禁' in ename: etype, direction, p_level = 'unlock', -1, 2
                elif '增持' in ename: etype, direction, p_level = 'increase', 1, 2
                elif '减持' in ename: etype, direction, p_level = 'reduction', -1, 2
                elif '回购' in ename: etype, direction, p_level = 'buyback_plan', 1, 3
                elif '中标' in ename or '合同' in ename: etype, direction, p_level = 'big_contract', 1, 2
                elif '立案' in ename or '处罚' in ename: etype, direction, p_level = 'lawsuit', -1, 0
                elif '业绩' in ename and '预增' in ename: etype, direction, p_level = 'earnings_revise', 1, 2
                elif '亏损' in ename: etype, direction, p_level = 'earnings_revise', -1, 2
                
                events_rows.append({
                    'code': code6,
                    'date': e_date,
                    'event_type': etype,
                    'p_level': p_level,
                    'direction': direction,
                    'title': ename,
                    'content': econtent[:200] if econtent else '',
                })

if events_rows:
    df = pd.DataFrame(events_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved: {OUT} ({len(df)} rows, {df['code'].nunique()} stocks)")
    print(f"Event types: {df['event_type'].value_counts().to_dict()}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
else:
    print("No events pulled")
