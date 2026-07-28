"""汇总空仓择时参数网格: 判断最优点是平滑高原还是过拟合尖峰

输出三张矩阵 (行=广度阈值, 列=确认天数): 夏普 / 年化超额% / 最大回撤%
并给出邻域稳定性诊断: 最优点周围 8 格的均值与标准差。
"""
import glob
import json
import re

import numpy as np
import pandas as pd

rows = []
for f in sorted(glob.glob("data/processed/wf_daily_pit_gridB*_cap100000.json")):
    m = re.search(r"gridB(\d+)C(\d)", f)
    if not m:
        continue
    d = json.load(open(f, encoding="utf-8"))
    s = d["summary"]
    b = float("0." + m.group(1))
    rows.append({"阈值": b, "确认": int(m.group(2)),
                 "夏普": s["sharpe"], "总收益%": s["total_return_pct"],
                 "年化%": s["annualized_return_pct"], "超额年化%": s["excess_annual_pct"],
                 "IR": s["information_ratio"], "回撤%": s["max_dd_pct"],
                 "空仓%": s["cash_days_pct"], "费用%": s["total_cost_pct"],
                 "交易": s["n_trades"]})

if not rows:
    raise SystemExit("没有网格结果, 先跑 scripts/run_regime_grid.sh")

df = pd.DataFrame(rows).sort_values(["阈值", "确认"])
print(f"共 {len(df)} 个组合\n")

for metric in ["夏普", "超额年化%", "回撤%", "空仓%"]:
    p = df.pivot(index="阈值", columns="确认", values=metric)
    print(f"=== {metric} (行=广度阈值, 列=确认天数) ===")
    print(p.round(2).to_string())
    print()

print("=== 全部组合明细 (按夏普排序) ===")
print(df.sort_values("夏普", ascending=False).to_string(index=False))

# 邻域稳定性: 最优点周围是否也不错
piv = df.pivot(index="阈值", columns="确认", values="夏普")
bi, bj = np.unravel_index(np.nanargmax(piv.values), piv.shape)
best_b, best_c = piv.index[bi], piv.columns[bj]
neigh = piv.values[max(0, bi - 1):bi + 2, max(0, bj - 1):bj + 2].flatten()
neigh = neigh[~np.isnan(neigh)]
print(f"\n=== 稳健性诊断 ===")
print(f"最优组合: 阈值 {best_b:.2f} 确认 {best_c} 天, 夏普 {piv.values[bi,bj]:.2f}")
print(f"邻域({len(neigh)}格) 夏普: 均值 {neigh.mean():.2f} 标准差 {neigh.std():.2f} "
      f"最低 {neigh.min():.2f}")
print(f"全网格 夏普: 均值 {np.nanmean(piv.values):.2f} 标准差 {np.nanstd(piv.values):.2f} "
      f"最低 {np.nanmin(piv.values):.2f} 最高 {np.nanmax(piv.values):.2f}")
gap = piv.values[bi, bj] - neigh.mean()
print(f"最优点比邻域均值高 {gap:.2f}")
if gap > 0.25:
    print("-> 警告: 最优点显著高于邻域, 像是过拟合尖峰, 不可直接采信")
else:
    print("-> 最优点与邻域接近, 参数处在平滑高原, 结论较可信")

n_beat = (df["超额年化%"] > 0).sum()
print(f"\n{n_beat}/{len(df)} 个组合超额为正 "
      f"({n_beat/len(df)*100:.0f}%) —— 越高说明规则本身越有效, 而非参数挑得巧")
