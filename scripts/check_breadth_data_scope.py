#!/usr/bin/env python3
"""广度择时对数据范围的敏感度

策略的择时信号 = 全市场 5533 只中收盘价站上 MA20 的占比。
若数据源受限(如 iFinD 个人版配额), 只能拉 300 只股票池, 信号还成立吗?
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE = ROOT / 'data' / 'raw' / 'kline'
MA, THRESH = 20, 0.35

u = pd.read_parquet(ROOT / 'data' / 'universe' / 'universe_pit.parquet')
u['effective_date'] = pd.to_datetime(u['effective_date'])
pool = set(u['code'].astype(str).str.zfill(6))
cur_pool = set(u[u['effective_date'] == u['effective_date'].max()]['code']
               .astype(str).str.zfill(6))
print(f'全市场 {len(list(KLINE.glob("*.parquet")))} 只 | '
      f'历史池合计 {len(pool)} 只 | 当期池 {len(cur_pool)} 只\n')

rows_all, rows_pool, rows_cur = [], [], []
for p in sorted(KLINE.glob('*.parquet')):
    try:
        kl = pd.read_parquet(p, columns=['date', 'close'])
    except Exception:
        continue
    kl['date'] = pd.to_datetime(kl['date'])
    kl = kl.sort_values('date')
    kl['above'] = (kl['close'] > kl['close'].rolling(MA).mean()).astype(float)
    rec = kl[['date', 'above']]
    rows_all.append(rec)
    if p.stem in pool:
        rows_pool.append(rec)
    if p.stem in cur_pool:
        rows_cur.append(rec)


def breadth(rows):
    return pd.concat(rows, ignore_index=True).groupby('date')['above'].mean()


b_all, b_pool, b_cur = breadth(rows_all), breadth(rows_pool), breadth(rows_cur)
df = pd.DataFrame({'all': b_all, 'pool519': b_pool, 'cur300': b_cur}).dropna()
df = df[df.index >= '2022-09-01']
print(f'对比区间 {df.index.min().date()} ~ {df.index.max().date()}, {len(df)} 个交易日\n')

print('=== 广度序列相关性 ===')
for col, name in (('pool519', '历史池519只'), ('cur300', '当期池300只')):
    print(f'  全市场 vs {name}: 相关系数 {df["all"].corr(df[col]):.4f}, '
          f'平均差 {(df[col] - df["all"]).mean():+.4f}')

print(f'\n=== 择时信号一致性 (广度 > {THRESH} 视为可开仓) ===')
sig_all = df['all'] > THRESH
for col, name in (('pool519', '历史池519只'), ('cur300', '当期池300只')):
    sig = df[col] > THRESH
    agree = (sig == sig_all).mean()
    both_in = (sig & sig_all).sum()
    only_sub = (sig & ~sig_all).sum()
    only_all = (~sig & sig_all).sum()
    print(f'  {name}: 一致 {agree*100:.1f}%  '
          f'(同时开仓 {both_in} 天, 仅子集开仓 {only_sub} 天, 仅全市场开仓 {only_all} 天)')
    print(f'    -> 全市场开仓天数 {sig_all.sum()}, {name}开仓天数 {sig.sum()}')

print('\n=== iFinD MCP 个人版配额测算 (batch=10, 每月约20个交易日) ===')
for n, name in ((5533, '全市场'), (519, '历史池519只'), (300, '当期池300只')):
    per_day = int(np.ceil(n / 10))
    per_month = per_day * 20
    verdict = '够用' if per_month <= 5000 else f'超配额 {per_month/5000:.1f} 倍'
    print(f'  {name:12s} {per_day:>4} 次/天  {per_month:>6,} 次/月   '
          f'配额5000 -> {verdict}')
