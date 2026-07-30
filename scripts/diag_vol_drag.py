"""解释缺口: 分层分析估算毛收益为正, 为何 ¥20,000 实测 -17.6%?

怀疑: 波动率拖累。只持 3 只 x 10 天, 单笔波动极大, 几何收益远低于算术平均。
    几何 ≈ 算术 - σ²/2
若组合每期波动 9%, 则每期拖累 0.4%, 69 个周期就是 -28 个百分点。

本脚本用回测日频净值直接量化: 算术平均 / 几何平均 / 波动 / 拖累,
并对比不同本金与持仓数, 看分散化能减少多少拖累。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

RUNS = [
    ("¥20k  3只 off", "wf_daily_P1SW_h10_n3_off_ts2022-09-01_te2026-07-27_cap20000.json"),
    ("¥20k  5只 off", "wf_daily_P1SW_h10_n5_off_ts2022-09-01_te2026-07-27_cap20000.json"),
    ("¥20k 10只 off", "wf_daily_P1SW_h10_n10_off_ts2022-09-01_te2026-07-27_cap20000.json"),
    ("¥100k 3只 off", "wf_daily_P1SW_h10_n3_off_ts2022-09-01_te2026-07-27_cap100000.json"),
    ("¥100k 5只 off", "wf_daily_P1SW_h10_n5_off_ts2022-09-01_te2026-07-27_cap100000.json"),
    ("¥100k10只 off", "wf_daily_P1SW_h10_n10_off_ts2022-09-01_te2026-07-27_cap100000.json"),
]

print("=" * 96)
print("日频收益的算术 vs 几何 (年化), 以及波动率拖累")
print("=" * 96)
print(f"{'配置':<16} {'算术年化':>10} {'几何年化':>10} {'年化波动':>10} "
      f"{'拖累σ²/2':>10} {'总收益':>10} {'最大回撤':>9}")
print("-" * 96)

for name, fn in RUNS:
    p = PROC / fn
    if not p.exists():
        print(f"{name:<16}  (缺失 {fn})")
        continue
    d = json.load(open(p, encoding="utf-8"))
    daily = pd.DataFrame(d["daily"])
    r = daily["daily_ret"].astype(float).dropna().values
    if r.max() > 1.5:            # 万一是百分数
        r = r / 100.0
    arith = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    geo = (1 + r).prod() ** (252 / len(r)) - 1
    drag = vol ** 2 / 2
    s = d["summary"]
    print(f"{name:<16} {arith:>+9.1%} {geo:>+10.1%} {vol:>10.1%} "
          f"{drag:>10.1%} {s['total_return_pct']:>9.1f}% {s['max_dd_pct']:>8.1f}%")

print()
print("=" * 96)
print("只在持仓日(非空仓)统计, 排除空仓稀释")
print("=" * 96)
print(f"{'配置':<16} {'持仓天数':>9} {'算术年化':>10} {'几何年化':>10} "
      f"{'年化波动':>10} {'拖累':>9}")
print("-" * 96)
for name, fn in RUNS:
    p = PROC / fn
    if not p.exists():
        continue
    d = json.load(open(p, encoding="utf-8"))
    daily = pd.DataFrame(d["daily"])
    inv = daily[daily["n_holdings"].astype(float) > 0]
    r = inv["daily_ret"].astype(float).dropna().values
    if len(r) < 50:
        continue
    if r.max() > 1.5:
        r = r / 100.0
    arith, vol = r.mean() * 252, r.std() * np.sqrt(252)
    geo = (1 + r).prod() ** (252 / len(r)) - 1
    print(f"{name:<16} {len(r):>9} {arith:>+9.1%} {geo:>+10.1%} "
          f"{vol:>10.1%} {vol**2/2:>8.1%}")

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("  若 ¥20k 的算术年化为正而几何年化为负, 则 -17.6% 的主因是波动率拖累,")
print("  即【只持3只导致的极端集中】, 而非模型没有 alpha。")
print("  同时对比持仓数递增时拖累的下降幅度 —— 这是分散化的真实价值,")
print("  但在 ¥20k 下会被 ¥5 最低佣金造成的成本上升抵消。")
