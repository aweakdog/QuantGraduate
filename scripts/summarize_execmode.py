#!/usr/bin/env python3
"""开盘成交 vs 尾盘成交, 在不同滑点下的对比 (本金2万)"""
import json, glob

def get(mode, s):
    tag = f"em_{mode}_s{s}"
    f = glob.glob(f'data/processed/wf_daily_{tag}_*_cap20000.json')
    return json.load(open(f[0])) if f else None

slips = [('000', 0.0), ('001', 0.1), ('002', 0.2), ('003', 0.3)]

print('=== 总收益% / 夏普 ===')
print(f"{'单边滑点':>10}{'T+1开盘':>20}{'T+1尾盘':>20}{'尾盘优势':>14}")
print('-' * 66)
for code, pct in slips:
    a, b = get('t1open', code), get('t1close', code)
    if not a or not b:
        continue
    sa, sb = a['summary'], b['summary']
    d = sb['total_return_pct'] - sa['total_return_pct']
    print(f"{pct:>9.1f}%{sa['total_return_pct']:>13.1f}% {sa['sharpe']:>5.2f}"
          f"{sb['total_return_pct']:>13.1f}% {sb['sharpe']:>5.2f}"
          f"{d:>+11.1f}pp")

print('\n=== 明细 ===')
for mode, name in (('t1open', 'T+1开盘'), ('t1close', 'T+1尾盘')):
    print(f'\n{name}')
    print(f"  {'滑点':>7}{'总收益%':>10}{'年化%':>8}{'夏普':>7}{'回撤%':>8}"
          f"{'超额年化%':>11}{'期末资产':>11}{'成交':>6}")
    for code, pct in slips:
        d = get(mode, code)
        if not d:
            continue
        s = d['summary']
        print(f"  {pct:>6.1f}%{s['total_return_pct']:>10.1f}{s['annualized_return_pct']:>8.1f}"
              f"{s['sharpe']:>7.2f}{s['max_dd_pct']:>8.1f}{s['excess_annual_pct']:>11.1f}"
              f"{s['final_value']:>11,.0f}{s['n_trades']:>6}")

print('\n=== 现实情景: 开盘滑点高于尾盘 ===')
combos = [('开盘0.3% vs 尾盘0.1%', 't1open', '003', 't1close', '001'),
          ('开盘0.2% vs 尾盘0.1%', 't1open', '002', 't1close', '001'),
          ('开盘0.3% vs 尾盘0.2%', 't1open', '003', 't1close', '002')]
for label, m1, s1, m2, s2 in combos:
    a, b = get(m1, s1), get(m2, s2)
    if not a or not b:
        continue
    sa, sb = a['summary'], b['summary']
    win = '尾盘胜' if sb['sharpe'] > sa['sharpe'] else '开盘胜'
    print(f"  {label}: {sa['total_return_pct']:.1f}%/{sa['sharpe']:.2f} vs "
          f"{sb['total_return_pct']:.1f}%/{sb['sharpe']:.2f}  -> {win}")
