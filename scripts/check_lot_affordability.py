"""10000元本金下, 一手门槛会砍掉多少股票池?"""
import pathlib

import numpy as np
import pandas as pd

KD = pathlib.Path('data/raw/kline')
CM = {'时间': 'date', '收盘价': 'close'}

df = pd.read_parquet('data/processed/training_data_v24.parquet', columns=['date', 'code'])
codes = sorted(df['code'].astype(str).str[:6].unique())

px = []
for c in codes:
    p = KD / f'{c}.parquet'
    if not p.exists():
        continue
    k = pd.read_parquet(p).rename(columns=CM)
    k['date'] = pd.to_datetime(k['date'])
    k = k[k['date'] >= '2023-09-19']
    if len(k):
        px.append(k['close'].median())
px = pd.Series(px)

print(f"股票池 {len(px)} 只, 回测期内收盘价中位数分布:")
for q in [.1, .25, .5, .75, .9]:
    print(f"    P{int(q*100):<3d}: {px.quantile(q):7.1f} 元  (一手 {px.quantile(q)*100:8,.0f} 元)")

print("\n不同本金 x 持仓数 下, 每个仓位的预算 和 可买股票占比:")
print(f"{'本金':>8} {'持仓数':>6} {'每仓预算':>10} {'可买一手的股票占比':>18}")
print("-" * 50)
for cap in (10000, 30000, 50000, 100000):
    for n in (3, 5, 10):
        alloc = cap / n
        pct = (px * 100 <= alloc).mean() * 100
        mark = "  <-- 可用" if pct >= 50 else ("  <-- 勉强" if pct >= 25 else "  <-- 不可用")
        print(f"{cap:>8,} {n:>6} {alloc:>10,.0f} {pct:>17.1f}%{mark}")

print("\n结论: 10000元本金下")
for n in (1, 2, 3, 5, 10):
    alloc = 10000 / n
    pct = (px * 100 <= alloc).mean() * 100
    print(f"    持仓 {n:2d} 只 -> 每仓 {alloc:6,.0f} 元 -> 只能买价格 <= {alloc/100:5.1f} 元的股票, 占池子 {pct:5.1f}%")
