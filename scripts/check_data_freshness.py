"""各数据源新鲜度体检 —— 看看离今天差多少"""
import pathlib
from datetime import datetime

import pandas as pd

ROOT = pathlib.Path('.')
DATA = ROOT / 'data'
TODAY = pd.Timestamp(datetime.now().date())


def latest_in(paths, date_cols=('date', '时间', 'pub_time', '日期')):
    best, n = None, 0
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        n += 1
        col = next((c for c in date_cols if c in df.columns), None)
        if col is None:
            continue
        try:
            m = pd.to_datetime(df[col], errors='coerce').max()
        except Exception:
            continue
        if pd.notna(m) and (best is None or m > best):
            best = m
    return best, n


SOURCES = [
    ('日K线',        list((DATA / 'raw' / 'kline').glob('*.parquet'))[:400]),
    ('1分钟K线',     list((DATA / 'raw' / 'kline_1min').glob('*.parquet'))),
    ('资金流(daily)', list((DATA / 'raw' / 'fund_flow').glob('*.parquet'))[:200]),
    ('资金流(full)',  list((DATA / 'raw' / 'fund_flow_full').glob('*.parquet'))),
    ('基本面',       list((DATA / 'raw' / 'fundamentals').glob('*.parquet'))[:200]),
    ('事件(daily)',  list((DATA / 'raw' / 'events_daily').glob('*.parquet'))[:200]),
    ('事件(akshare)', list((DATA / 'raw' / 'akshare_events').glob('*.parquet'))[:200]),
    ('公告',         list((DATA / 'raw' / 'announcements').glob('*.parquet'))[:200]),
    ('板块日频',     list((DATA / 'raw' / 'sectors_daily').glob('*.parquet'))),
    ('宏观/外盘',    list((DATA / 'raw' / 'macro').glob('*.parquet'))),
]

print(f"今天: {TODAY.date()}\n")
print(f"{'数据源':<16}{'文件数':>7}{'最新日期':>14}{'滞后天数':>10}   状态")
print("-" * 62)
for name, paths in SOURCES:
    if not paths:
        print(f"{name:<16}{0:>7}{'-':>14}{'-':>10}   缺失")
        continue
    m, n = latest_in(paths)
    if m is None:
        print(f"{name:<16}{len(paths):>7}{'无日期列':>14}{'-':>10}   ?")
        continue
    lag = (TODAY - m.normalize()).days
    st = '最新' if lag <= 3 else ('偏旧' if lag <= 14 else '过期')
    print(f"{name:<16}{len(paths):>7}{str(m.date()):>14}{lag:>10}   {st}")

print()
tp = DATA / 'processed' / 'training_data_v24.parquet'
if tp.exists():
    d = pd.read_parquet(tp, columns=['date', 'code'])
    d['date'] = pd.to_datetime(d['date'])
    lag = (TODAY - d['date'].max().normalize()).days
    print(f"训练集 training_data_v24: {d['date'].min().date()} ~ {d['date'].max().date()}"
          f" | {d['date'].nunique()} 交易日 | {d['code'].nunique()} 只 | 滞后 {lag} 天")
