"""诊断小资金回测: 最低佣金5元 与 整手拒单 对收益的侵蚀"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed/wf_daily_small3_t1open_ts2022-09-01_te2026-07-24_cap10000.json"
j = json.loads(P.read_text())

print("=== 结果文件顶层键 ===")
print(", ".join(list(j.keys())[:25]))

trades = j.get("trades") or j.get("trade_log") or []
print(f"\n交易记录: {len(trades)} 笔")
if trades:
    t = pd.DataFrame(trades)
    print(f"列: {list(t.columns)}")
    print(t.head(3).to_string())

    cost_col = next((c for c in t.columns if "cost" in c.lower() or "fee" in c.lower()), None)
    amt_col = next((c for c in t.columns if c.lower() in ("amount", "value", "notional", "amt")), None)
    if cost_col:
        print(f"\n=== 费用结构 ({cost_col}) ===")
        c = pd.to_numeric(t[cost_col], errors="coerce").dropna()
        print(f"  总费用 {c.sum():,.0f} 元 | 笔数 {len(c)} | 均值 {c.mean():.2f} 元/笔")
        print(f"  等于 5.00 元的笔数: {(c.round(2) == 5.00).sum()} "
              f"({100*(c.round(2)==5.00).mean():.1f}%)  <- 触发最低佣金")
        if amt_col:
            a = pd.to_numeric(t[amt_col], errors="coerce")
            print(f"  平均成交金额 {a.mean():,.0f} 元")
            print(f"  实际费率 {100*c.sum()/a.sum():.4f}%  (名义费率 0.06%)")
            print(f"  若无5元下限, 总费用应为 {0.0006*a.sum():,.0f} 元")
            print(f"  最低佣金多收 {c.sum()-0.0006*a.sum():,.0f} 元 "
                  f"= 本金的 {100*(c.sum()-0.0006*a.sum())/10000:.1f}%")

print("\n=== 关键指标 ===")
for k in ["ic_mean", "ic_t", "total_return", "annual_return", "sharpe", "max_dd",
          "final_value", "total_cost_pct", "n_trades", "reject_buy", "reject_sell",
          "bench_total_return", "alpha_annual", "ir", "beta", "avg_deploy"]:
    if k in j:
        print(f"  {k:22s} {j[k]}")

print("\n=== 反事实: 扣除最低佣金溢价后的净值 ===")
fv = j.get("final_value")
if trades and cost_col and amt_col:
    excess = c.sum() - 0.0006 * a.sum()
    print(f"  实际期末      ¥{fv:,.0f}")
    print(f"  若费率纯比例  ¥{fv + excess:,.0f}  (+{100*excess/10000:.1f}pct 本金)")
    print(f"  基准期末      ¥{10000*(1+j.get('bench_total_return',0)):,.0f}")

print("\n=== 不同本金下最低佣金占比推算 ===")
if trades and cost_col:
    n = len(c)
    min_fee_total = 5.0 * n
    for cap in (10000, 30000, 50000, 100000, 300000):
        # 成交金额随本金等比放大, 5元下限在多大金额下失效: 5/0.0006 = 8333元
        print(f"  本金 ¥{cap:>7,}: 单笔约 ¥{cap/3:>7,.0f} | "
              f"{'触发5元下限' if cap/3 < 8333 else '按比例收费':10s} | "
              f"最低佣金总额占本金 {100*min_fee_total/cap:5.1f}%")
    print(f"\n  临界: 单笔成交额需 > ¥8,333 才不触发5元下限 (5/0.0006)")
    print(f"        3仓 -> 本金需 > ¥25,000 ; 5仓 -> 本金需 > ¥41,667")
