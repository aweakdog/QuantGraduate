"""确认 K线 -> 训练集 的重建链路在 Mac 上是否完整"""
import json
import pathlib

import pandas as pd

P = pathlib.Path('data/processed')
U = pathlib.Path('data/universe')

print("=== processed 训练集文件 ===")
for v in ['v15', 'v22', 'v23', 'v24']:
    f = P / f'training_data_{v}.parquet'
    if f.exists():
        d = pd.read_parquet(f, columns=['date', 'code'])
        d['date'] = pd.to_datetime(d['date'])
        import pyarrow.parquet as pq
        ncol = len(pq.ParquetFile(f).schema.names)
        print(f"  {v}: {len(d):>8,} 行  {d['code'].nunique():>4} 只  {ncol:>4} 列  "
              f"{d['date'].min().date()} ~ {d['date'].max().date()}")
    else:
        print(f"  {v}: 不存在")

print("\n=== build_all 使用的股票池 ===")
for name in ['watchlist_top120.json', 'watchlist.json', 'watchlist_216.json']:
    f = U / name
    if f.exists():
        w = json.loads(f.read_text())
        items = w.get('watchlist', w) if isinstance(w, dict) else w
        print(f"  {name}: {len(items)} 只  <- {'build_all 优先用这个' if name == 'watchlist_top120.json' else ''}")
    else:
        print(f"  {name}: 不存在")

print("\n=== feature_engine 依赖的数据文件 ===")
deps = [
    ('资金流 consolidated', 'data/raw/fund_flow_full/fundflow_history.parquet'),
    ('两融 consolidated', 'data/raw/MainNetFlow/margintrade_history.parquet'),
    ('事件 v2', 'data/raw/events_ifind/events_v2.parquet'),
    ('概念映射', 'data/universe/concept_stock_map.json'),
    ('供应链图谱', 'data/universe/supply_chain_map.json'),
]
for label, rel in deps:
    f = pathlib.Path(rel)
    if not f.exists():
        print(f"  {label:22s} 缺失  {rel}")
        continue
    if f.suffix == '.parquet':
        d = pd.read_parquet(f, columns=['date'] if 'date' in pd.read_parquet(f).columns[:20] else None)
        try:
            m = pd.to_datetime(d['date']).max().date()
            print(f"  {label:22s} OK    最新 {m}")
        except Exception:
            print(f"  {label:22s} OK    (无date列)")
    else:
        print(f"  {label:22s} OK")

print("\n=== 日K线目录 ===")
kd = pathlib.Path('data/raw/kline')
files = list(kd.glob('*.parquet'))
print(f"  {len(files)} 个文件")
uni = json.loads((U / 'watchlist_216.json').read_text())
items = uni.get('watchlist', uni) if isinstance(uni, dict) else uni
codes = [str(x['code'])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]
have = sum(1 for c in codes if (kd / f'{c}.parquet').exists())
print(f"  216池覆盖: {have}/{len(codes)}")
