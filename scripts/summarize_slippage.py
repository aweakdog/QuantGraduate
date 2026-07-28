#!/usr/bin/env python3
"""滑点敏感性: 本金2万, h10 + 广度0.35/确认1天"""
import json, glob

rows = []
for f in glob.glob('data/processed/wf_daily_slip*_cap20000.json'):
    d = json.load(open(f))
    s = d['summary']
    st = d.get('stability') or []
    rows.append({
        'slip': d.get('slippage', 0), 's': s,
        'h1': st[0]['excess_annual_pct'] if st else None,
        'h2': st[1]['excess_annual_pct'] if len(st) > 1 else None,
    })
rows.sort(key=lambda r: r['slip'])

print(f"{'单边滑点':>9}{'往返成本':>10}{'总收益%':>10}{'年化%':>8}{'夏普':>7}"
      f"{'回撤%':>8}{'超额年化%':>11}{'期末资产':>11}{'费用%':>8}")
print('-' * 82)
base = None
for r in rows:
    s = r['s']
    rt = (r['slip'] + 0.0006) * 2 * 100
    if base is None:
        base = s['total_return_pct']
    print(f"{r['slip']*100:>8.2f}%{rt:>9.2f}%{s['total_return_pct']:>10.1f}"
          f"{s['annualized_return_pct']:>8.1f}{s['sharpe']:>7.2f}{s['max_dd_pct']:>8.1f}"
          f"{s['excess_annual_pct']:>11.1f}{s['final_value']:>11,.0f}{s['total_cost_pct']:>8.1f}")

print('\n=== 分段稳健性 ===')
for r in rows:
    ok = '两段都赢 ✓' if (r['h1'] or 0) > 0 and (r['h2'] or 0) > 0 else '✗'
    print(f"  滑点 {r['slip']*100:.2f}%: 前半段 {r['h1']:+.1f}% | 后半段 {r['h2']:+.1f}%   {ok}")

print('\n=== 每 0.1% 滑点的代价 ===')
for i in range(1, len(rows)):
    d_slip = (rows[i]['slip'] - rows[0]['slip']) * 100
    d_ret = rows[i]['s']['total_return_pct'] - rows[0]['s']['total_return_pct']
    d_sh = rows[i]['s']['sharpe'] - rows[0]['s']['sharpe']
    print(f"  滑点 0 -> {rows[i]['slip']*100:.2f}%: 总收益 {d_ret:+.1f}pp "
          f"({d_ret/d_slip*0.1:+.1f}pp per 0.1%) | 夏普 {d_sh:+.2f}")
