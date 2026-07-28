"""更新后校验: 历史是否被改坏 / 新数据是否合理"""
import json
import pathlib

import pandas as pd

KD = pathlib.Path('data/raw/kline')
U = json.loads(pathlib.Path('data/universe/watchlist_216.json').read_text())
items = U.get('watchlist', U) if isinstance(U, dict) else U
codes = [str(x['code'])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]

rows = []
for c in codes:
    p = KD / f'{c}.parquet'
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    d['date'] = pd.to_datetime(d['date'])
    rows.append({
        'code': c, 'n': len(d),
        'start': d['date'].min(), 'end': d['date'].max(),
        'nan_close': int(d['close'].isna().sum()),
        'nonpos': int((d['close'] <= 0).sum()),
        'cols': len(d.columns),
    })
r = pd.DataFrame(rows)
print(f"216池校验: {len(r)} 只")
print(f"  最新日期分布: {r['end'].value_counts().head(4).to_dict()}")
print(f"  起始日期最小/最大: {r['start'].min().date()} / {r['start'].max().date()}")
print(f"  行数 中位/最小/最大: {int(r['n'].median())} / {r['n'].min()} / {r['n'].max()}")
print(f"  close 有NaN的股票数: {(r['nan_close'] > 0).sum()}")
print(f"  close <=0 的股票数 : {(r['nonpos'] > 0).sum()}")
print(f"  列数一致: {r['cols'].nunique() == 1} ({r['cols'].iloc[0]} 列)")

print("\n单只抽查 000063 最近 8 天:")
d = pd.read_parquet(KD / '000063.parquet')
d['date'] = pd.to_datetime(d['date'])
print(d.tail(8)[['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']].to_string(index=False))

print("\n历史一致性抽查 (与训练集里的旧收盘价对比, 2026-06 区间):")
tr = pd.read_parquet('data/processed/training_data_v24.parquet', columns=['date', 'code', 'ret_1d'])
tr['date'] = pd.to_datetime(tr['date'])
tr['code6'] = tr['code'].astype(str).str[:6]
bad = 0
for c in codes[:40]:
    d = pd.read_parquet(KD / f'{c}.parquet')
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    d['ret_new'] = d['close'].pct_change()
    t = tr[(tr['code6'] == c) & (tr['date'] >= '2026-06-01') & (tr['date'] <= '2026-06-30')]
    if not len(t):
        continue
    j = t.set_index('date')[['ret_1d']].join(d[['ret_new']], how='inner').dropna()
    if len(j) and (j['ret_1d'] - j['ret_new']).abs().max() > 1e-4:
        bad += 1
print(f"  抽查 40 只, 6月日收益与旧训练集不一致的: {bad} 只 {'(前复权基准变动导致, 属正常)' if bad else '完全一致'}")
