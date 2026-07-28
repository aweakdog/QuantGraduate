#!/usr/bin/env python3
"""对比 2万 vs 10万 本金下的实际可执行性"""
import json, glob

print(f"{'配置':>20}{'总收益%':>10}{'年化%':>8}{'夏普':>7}{'回撤%':>8}"
      f"{'超额年化%':>11}{'期末资产':>13}{'成交':>6}")
print('-' * 86)
for cap in (20000, 100000):
    for c in (1, 2):
        f = glob.glob(f'data/processed/wf_daily_cap{cap}_B35C{c}_*.json')
        if not f:
            continue
        s = json.load(open(f[0]))['summary']
        name = f"¥{cap:,} 确认{c}天"
        print(f"{name:>20}{s['total_return_pct']:>10.1f}{s['annualized_return_pct']:>8.1f}"
              f"{s['sharpe']:>7.2f}{s['max_dd_pct']:>8.1f}{s['excess_annual_pct']:>11.1f}"
              f"{s['final_value']:>13,.0f}{s['n_trades']:>6}")

print('\n=== 拒单分解 (确认1天) ===')
for cap in (20000, 100000):
    f = glob.glob(f'data/processed/wf_daily_cap{cap}_B35C1_*.json')
    if not f:
        continue
    s = json.load(open(f[0]))['summary']
    r = s['reject_breakdown']
    print(f"\n¥{cap:,}")
    print(f"  买入被拒: 停牌 {r['buy_halt']} | 涨停 {r['buy_limit_up']} | "
          f"买不起一手 {r['buy_lot_too_big']} | 现金不足 {r['buy_no_cash']}")
    print(f"  卖出被拒: 停牌 {r['sell_halt']} | 跌停 {r['sell_limit_down']}")
    print(f"  平均持仓 {s['avg_holdings']} 只 | 平均部署率 {s['avg_deployed_pct']}% | "
          f"总费用 {s['total_cost_pct']}%")
