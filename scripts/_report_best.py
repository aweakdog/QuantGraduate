import json
import pandas as pd

F = 'data/processed/wf_daily_c1_base_ts2022-09-01_te2026-07-16_cap100000.json'
r = json.load(open(F))
s = r['summary']

print('最优方案 c1_base')
print('  区间      :', r['period'], f"({r['n_days']}个交易日)")
print('  配置      : 标签 fwd_5d_ret(按日期demean) / 持有5天分5档 / 每档2只 / 共10只 / L2回归')
print(f"  期末      : ¥{s['final_value']:,.0f}   总收益 {s['total_return_pct']:+.1f}%   年化 {s['annualized_return_pct']:+.1f}%")
print(f"  基准      : {s['benchmark_total_pct']:+.1f}%   年化 {s['benchmark_annual_pct']:+.1f}%")
print(f"  超额年化  : {s['excess_annual_pct']:+.1f}%   alpha {s['alpha_annual_pct']:+.1f}%   IR {s['information_ratio']}   beta {s['beta']}")
print(f"  夏普 {s['sharpe']}   最大回撤 {s['max_dd_pct']}%   成本 {s['total_cost_pct']}%   IC_t {s['ic_tstat']}")

d = pd.DataFrame(r['daily'])
d['date'] = pd.to_datetime(d['date'])
b = pd.read_parquet('data/processed/training_data_v24.parquet', columns=['date', 'fwd_1d_ret'])
b['date'] = pd.to_datetime(b['date'])
bm = b.groupby('date')['fwd_1d_ret'].mean().shift(1)
d['bench'] = pd.Series(bm.reindex(d['date']).values).fillna(0).values

print('\n  年度表现:')
for y, g in d.groupby(d['date'].dt.year):
    st = (1 + g['daily_ret']).prod() - 1
    bh = (1 + g['bench']).prod() - 1
    flag = '✓' if st > bh else '✗'
    print(f'    {y} ({len(g):3d}天): 策略 {st*100:+7.1f}%   基准 {bh*100:+7.1f}%   超额 {(st-bh)*100:+7.1f}% {flag}')

print('\n  分段稳健性:')
for h in r['stability']:
    print(f"    {h['segment']} {h['period']}: 策略 {h['strategy_pct']:+.1f}% vs 基准 {h['benchmark_pct']:+.1f}%"
          f" | 年化超额 {h['excess_annual_pct']:+.1f}% IR {h['ir']}")
