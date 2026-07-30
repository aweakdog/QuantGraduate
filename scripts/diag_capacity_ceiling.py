"""算清楚当前配置的理论天花板在哪, 以及瓶颈是哪一个

核心工具: Grinold 基本定律  IR ≈ IC × sqrt(N)
  N = 每年独立下注次数 = 每次调仓的持仓数 × 每年调仓次数
只持3只、10天一调 -> N 很小 -> 即使 IC 是真的, IR 也上不去。

同时统计 IC 的逐年稳定性和 t 值, 判断信号到底有没有。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
NEW = PROC / "wf_daily_em_t1close_s001_fundfix_ts2022-09-01_te2026-07-27_cap20000.json"

d = json.load(open(NEW, encoding="utf-8"))
s = d["summary"]
daily = pd.DataFrame(d["daily"])
daily["date"] = pd.to_datetime(daily["date"])

print("=" * 66)
print("一、信号本身有多强 (IC = 模型排序与真实收益的秩相关)")
print("=" * 66)
ic = daily["ic"].dropna()
n = len(ic)
t = ic.mean() / ic.std(ddof=1) * np.sqrt(n)
print(f"  IC 均值 {ic.mean():+.4f} | 标准差 {ic.std():.4f} | 天数 {n}")
print(f"  t 值 {t:.2f}  -> {'显著' if abs(t) > 2 else '不显著 (需 >2)'}")
print(f"  IC>0 的天数占比 {(ic > 0).mean():.1%}  (随机应为 50%)")
print()
print("  逐年 IC:")
for y, g in daily.groupby(daily["date"].dt.year):
    gi = g["ic"].dropna()
    if len(gi) < 20:
        continue
    ty = gi.mean() / gi.std(ddof=1) * np.sqrt(len(gi))
    print(f"    {y}  IC {gi.mean():+.4f}  t {ty:+5.2f}  正比例 {(gi>0).mean():5.1%}  ({len(gi)}天)")

print()
print("=" * 66)
print("二、Grinold 基本定律: 就算 IC 是真的, 只持 3 只能到什么 IR?")
print("=" * 66)
IC = ic.mean()
HOLD = d["hold_days"]
POS = d["target_positions"]
rebal_per_year = 252 / HOLD
print(f"  当前: 持仓 {POS} 只, 持有 {HOLD} 天 -> 每年调仓 {rebal_per_year:.0f} 次")
for pos in [3, 5, 8, 10, 15, 20, 30]:
    N = pos * rebal_per_year
    ir = IC * np.sqrt(N)
    mark = "  <-- 当前" if pos == POS else ""
    print(f"    持仓 {pos:>2} 只 -> 每年 {N:>5.0f} 次独立下注 -> 理论 IR 上限 {ir:.2f}{mark}")
print(f"\n  实测 IR = {s['information_ratio']}  (理论上限 {IC*np.sqrt(POS*rebal_per_year):.2f})")
print("  注: 理论上限是无成本、IC完全稳定下的乐观值, 实测低于它属正常")

print()
print("=" * 66)
print("三、2万本金的结构性约束")
print("=" * 66)
CAP = d["initial_capital"]
FEE_MIN = 5.0
FEE_RATE = d.get("trade_cost", 0.0006)
print(f"  本金 ¥{CAP:,.0f}")
for pos in [3, 5, 8, 10, 15]:
    budget = CAP / pos
    max_price = budget / 100
    fee_pct = max(budget * FEE_RATE, FEE_MIN) / budget
    rt = fee_pct * 2 * 100
    flag = ""
    if max_price < 10:
        flag = "  <-- 只能买极低价股"
    print(f"    持仓 {pos:>2} 只: 单只预算 ¥{budget:>6,.0f} | "
          f"买得起的最高股价 ¥{max_price:>5.1f} | 单边费率 {fee_pct:.3%} | 往返 {rt:.2f}%{flag}")
print(f"\n  ¥5 最低佣金是硬约束: 仓位越小, 费率越高。")
print(f"  当前每年调仓 {rebal_per_year:.0f} 次, {POS} 只 -> 年换手成本约:")
for pos in [3, 10]:
    budget = CAP / pos
    fee_pct = max(budget * FEE_RATE, FEE_MIN) / budget
    annual = fee_pct * 2 * rebal_per_year * 100
    print(f"    持仓 {pos:>2} 只 -> 年成本约 {annual:.1f}% 的本金")

print()
print("=" * 66)
print("四、实际业绩 vs 基准")
print("=" * 66)
print(f"  策略总收益 {s['total_return_pct']:>7}%   年化 {s['annualized_return_pct']:>6}%")
print(f"  基准总收益 {s['benchmark_total_pct']:>7}%   年化 {s['benchmark_annual_pct']:>6}%")
print(f"  超额年化   {s['excess_annual_pct']:>7}%   IR {s['information_ratio']}")
print(f"  最大回撤   {s['max_dd_pct']:>7}%   夏普 {s['sharpe']}")
print(f"  总费用占比 {s.get('total_cost_pct','NA')}%   交易 {s['n_trades']} 笔")
print(f"  空仓天数占比 {s.get('cash_days_pct','NA')}%")
