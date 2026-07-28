#!/usr/bin/env python3
"""无泄漏预测缓存下的空仓择时网格结果"""
import json, glob, re
import numpy as np
import pandas as pd

rows = []
for f in glob.glob('data/processed/wf_daily_cgrid_B*_cap100000.json'):
    m = re.search(r'cgrid_B(\d+)C(\d+)_', f)
    if not m:
        continue
    d = json.load(open(f))
    s = d['summary']
    st = d.get('stability') or []
    rows.append({
        'thr': float('0.' + m.group(1)), 'confirm': int(m.group(2)),
        'sharpe': s['sharpe'], 'excess': s['excess_annual_pct'],
        'dd': s['max_dd_pct'], 'cash': s.get('cash_days_pct', 0),
        'total': s['total_return_pct'], 'ir': s['information_ratio'],
        'trades': s['n_trades'], 'cost': s['total_cost_pct'],
        'h1': st[0]['excess_annual_pct'] if st else np.nan,
        'h2': st[1]['excess_annual_pct'] if len(st) > 1 else np.nan,
    })

df = pd.DataFrame(rows).sort_values(['thr', 'confirm'])
print(f'共 {len(df)} 个组合\n')

for name, col in [('夏普', 'sharpe'), ('超额年化%', 'excess'), ('回撤%', 'dd'), ('空仓%', 'cash')]:
    print(f'=== {name} (行=广度阈值, 列=确认天数) ===')
    print(df.pivot(index='thr', columns='confirm', values=col).round(2).to_string())
    print()

print('=== 按夏普排序 ===')
p = df.sort_values('sharpe', ascending=False)
print(f"{'阈值':>6}{'确认':>5}{'夏普':>7}{'总收益%':>9}{'超额年化%':>10}{'IR':>6}"
      f"{'回撤%':>8}{'空仓%':>7}{'前半段':>8}{'后半段':>8}")
for _, r in p.iterrows():
    print(f"{r['thr']:>6.2f}{int(r['confirm']):>5}{r['sharpe']:>7.2f}{r['total']:>9.1f}"
          f"{r['excess']:>10.1f}{r['ir']:>6.2f}{r['dd']:>8.1f}{r['cash']:>7.1f}"
          f"{r['h1']:>8.1f}{r['h2']:>8.1f}")

off = glob.glob('data/processed/wf_daily_cgrid_off_*_cap100000.json')
if off:
    s = json.load(open(off[0]))['summary']
    print(f"\n对照 择时全关: 夏普 {s['sharpe']:.2f} 总收益 {s['total_return_pct']:.1f}% "
          f"超额年化 {s['excess_annual_pct']:.1f}% 回撤 {s['max_dd_pct']:.1f}%")

print('\n=== 稳健性诊断 ===')
best = df.loc[df['sharpe'].idxmax()]
print(f"最优: 阈值 {best['thr']:.2f} 确认 {int(best['confirm'])} 天, 夏普 {best['sharpe']:.2f}")
nb = df[(df['thr'].between(best['thr'] - 0.05, best['thr'] + 0.05)) &
        (df['confirm'].between(best['confirm'] - 1, best['confirm'] + 1))]['sharpe']
print(f"邻域({len(nb)}格) 夏普: 均值 {nb.mean():.2f} 标准差 {nb.std():.2f} 最低 {nb.min():.2f}")
print(f"全网格 夏普: 均值 {df['sharpe'].mean():.2f} 标准差 {df['sharpe'].std():.2f} "
      f"最低 {df['sharpe'].min():.2f} 最高 {df['sharpe'].max():.2f}")
pos = (df['excess'] > 0).sum()
print(f"{pos}/{len(df)} 个组合超额为正 ({pos/len(df)*100:.0f}%)")
both = ((df['h1'] > 0) & (df['h2'] > 0)).sum()
print(f"{both}/{len(df)} 个组合【前后两段都跑赢】 ({both/len(df)*100:.0f}%)  <-- 关键稳健性指标")
