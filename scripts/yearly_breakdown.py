#!/usr/bin/env python3
"""逐年拆解策略 vs 基准, 检查收益是否集中在个别时段"""
import json, glob, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def benchmark_series(dates, train_file, pit_universe):
    df = pd.read_parquet(ROOT / 'data' / 'processed' / train_file,
                         columns=['date', 'code', 'fwd_1d_ret'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['fwd_1d_ret'])
    if pit_universe:
        u = pd.read_parquet(ROOT / 'data' / 'universe' / pit_universe)
        u['effective_date'] = pd.to_datetime(u['effective_date'])
        eff = np.array(sorted(u['effective_date'].unique()))
        members = {d: set(g['code'].astype(str).str.zfill(6))
                   for d, g in u.groupby('effective_date')}
        c6 = df['code'].astype(str).str[:6]
        per = np.searchsorted(eff, df['date'].values, side='right') - 1
        keep = np.zeros(len(df), bool)
        for i, d in enumerate(eff):
            m = per == i
            if m.any():
                keep[m] = c6[m].isin(members[pd.Timestamp(d)]).values
        df = df[keep]
    b = df.groupby('date')['fwd_1d_ret'].mean().shift(1)
    return b.reindex(pd.to_datetime(dates)).fillna(0.0).values


def analyze(tag):
    f = glob.glob(f'data/processed/wf_daily_{tag}*_cap100000.json')
    if not f:
        print(f'  (找不到 {tag})')
        return
    d = json.load(open(f[0]))
    daily = pd.DataFrame(d['daily'])
    daily['date'] = pd.to_datetime(daily['date'])
    daily['bench'] = benchmark_series(daily['date'], d['train_file'], d.get('pit_universe'))
    daily['excess'] = daily['daily_ret'] - daily['bench']
    daily['year'] = daily['date'].dt.year

    print(f"\n  {tag}  (夏普 {d['summary']['sharpe']}, 总收益 {d['summary']['total_return_pct']}%)")
    print(f"    {'年份':>6}{'交易日':>7}{'策略%':>9}{'基准%':>9}{'超额%':>9}"
          f"{'空仓日':>7}{'年化超额%':>11}{'IR':>7}")
    for y, g in daily.groupby('year'):
        strat = (1 + g['daily_ret']).prod() - 1
        bench = (1 + g['bench']).prod() - 1
        ex = g['excess']
        ir = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
        cash = int(g['in_cash'].sum()) if 'in_cash' in g else 0
        print(f'    {y:>6}{len(g):>7}{strat*100:>9.1f}{bench*100:>9.1f}'
              f'{(strat-bench)*100:>9.1f}{cash:>7}{ex.mean()*252*100:>11.1f}{ir:>7.2f}')


if __name__ == '__main__':
    for tag in (sys.argv[1:] or ['clean_h10g00_', 'clean_h5g00_']):
        analyze(tag)
