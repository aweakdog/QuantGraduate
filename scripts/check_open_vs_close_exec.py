#!/usr/bin/env python3
"""开盘价 vs 尾盘价 的成交风险对比 (只看策略实际交易过的股票)

日线数据无法直接测买卖价差, 但可以测两个与执行风险高度相关的量:
  1. 相对当日均价(VWAP=amount/volume)的偏离度 —— 越偏离, 择时运气成分越大
  2. 隔夜跳空幅度 —— 开盘价承担全部隔夜风险, 尾盘价不承担
"""
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

f = glob.glob('data/processed/wf_daily_em_t1open_s002_*_cap20000.json')[0]
d = json.load(open(f))
codes = sorted({str(t['code'])[:6] for t in d['trades']})
dates = sorted({t['date'] for t in d['trades']})
d0, d1 = pd.Timestamp(min(dates)), pd.Timestamp(max(dates))
print(f'策略实际交易过 {len(codes)} 只, 区间 {d0.date()} ~ {d1.date()}\n')

rows = []
for c in codes:
    p = ROOT / 'data' / 'raw' / 'kline' / f'{c}.parquet'
    if not p.exists():
        g = list((ROOT / 'data' / 'raw' / 'kline').glob(f'*{c}*.parquet'))
        if not g:
            continue
        p = g[0]
    kl = pd.read_parquet(p)
    kl['date'] = pd.to_datetime(kl['date'])
    kl = kl[(kl['date'] >= d0) & (kl['date'] <= d1)].sort_values('date')
    kl = kl[(kl['volume'] > 0) & (kl['amount'] > 0)]
    if len(kl) < 30:
        continue
    vwap = kl['amount'] / kl['volume']
    vwap = vwap.where(vwap.between(kl['low'] * 0.9, kl['high'] * 1.1))
    rows.append({
        'code': c,
        'open_dev': float(((kl['open'] - vwap).abs() / vwap).mean() * 100),
        'close_dev': float(((kl['close'] - vwap).abs() / vwap).mean() * 100),
        'gap': float((kl['open'] / kl['close'].shift(1) - 1).abs().mean() * 100),
    })

df = pd.DataFrame(rows).dropna()
print(f'有效样本 {len(df)} 只\n')
print('=== 相对当日成交均价(VWAP)的平均绝对偏离 ===')
print(f"  开盘价: {df['open_dev'].mean():.3f}%   (中位 {df['open_dev'].median():.3f}%)")
print(f"  收盘价: {df['close_dev'].mean():.3f}%   (中位 {df['close_dev'].median():.3f}%)")
diff = df['open_dev'].mean() - df['close_dev'].mean()
print(f"  开盘比收盘多偏离: {diff:+.3f}pp")
print(f"  开盘偏离更大的股票占比: {(df['open_dev'] > df['close_dev']).mean()*100:.0f}%")

print('\n=== 隔夜跳空 (开盘价独有的风险) ===')
print(f"  平均绝对跳空: {df['gap'].mean():.3f}%")
print('  尾盘成交不承担这部分风险。')

print('\n=== 结论 ===')
print(f'  按 VWAP 偏离度衡量, 开盘执行的价格不确定性比尾盘高约 {diff:.2f}pp。')
print('  注意: 这是"价格不确定性"而非真实买卖价差, 真实价差需分钟/tick数据才能测。')
