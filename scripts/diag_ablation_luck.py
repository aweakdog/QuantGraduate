"""消融实验结果到底是"特征更好"还是"少数几笔运气"?

判据:
1. IC 对比 —— IC 衡量整个横截面排序能力。若两者 IC 相近但收益差 7 倍,
   则差异不在信号质量, 而在"恰好哪几只落到前3名"(运气)。
2. 收益集中度 —— 把每笔交易配对成回合(FIFO), 看前 N 笔贡献了多少总盈利。
   只持3只时, 极少数暴涨股能主导整个净值曲线。
3. 剔除最赚的几笔后还剩多少 —— 稳健策略不应依赖个别标的。
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

RUNS = {
    "新数据+旧特征(消融)": "wf_daily_ABLATION_newdata_oldfeats_ts2022-09-01_te2026-07-27_cap20000.json",
    "新数据+新特征(当前)": "wf_daily_em_t1close_s001_fundfix_ts2022-09-01_te2026-07-27_cap20000.json",
}

loaded = {}
for name, fn in RUNS.items():
    p = PROC / fn
    if not p.exists():
        print(f"!! 缺失 {fn}")
        continue
    loaded[name] = json.load(open(p, encoding="utf-8"))

print("=" * 74)
print("一、IC 对比 (信号排序能力 —— 这是判断'特征是否更好'的直接指标)")
print("=" * 74)
print(f"{'配置':<22} {'IC均值':>9} {'IC t值':>8} {'正比例':>8} {'总收益%':>9} {'IR':>7}")
print("-" * 74)
for name, d in loaded.items():
    s = d["summary"]
    daily = pd.DataFrame(d["daily"])
    ic = daily["ic"].dropna()
    print(f"{name:<22} {s['ic_mean']:>+9.4f} {s['ic_tstat']:>8.2f} "
          f"{(ic > 0).mean():>7.1%} {s['total_return_pct']:>9.1f} "
          f"{s['information_ratio']:>7.2f}")

print()
print("=" * 74)
print("二、收益集中度 (按回合配对, FIFO)")
print("=" * 74)


def round_trips(trades):
    """把 buy/sell 配成回合, 返回每回合净盈亏"""
    books = defaultdict(deque)
    rts = []
    for t in sorted(trades, key=lambda x: x["date"]):
        code, sh = t["code"], t["shares"]
        if t["action"] == "buy":
            # net 是负数(现金流出); 成本 = -net
            books[code].append({"shares": sh, "cost": -t["net"], "date": t["date"]})
        else:
            proceeds = t["net"]          # 正数(现金流入)
            remain = sh
            while remain > 0 and books[code]:
                lot = books[code][0]
                take = min(remain, lot["shares"])
                frac_cost = lot["cost"] * take / lot["shares"]
                frac_proc = proceeds * take / sh
                rts.append({
                    "code": code, "buy_date": lot["date"], "sell_date": t["date"],
                    "cost": frac_cost, "proceeds": frac_proc,
                    "pnl": frac_proc - frac_cost,
                    "ret_pct": (frac_proc / frac_cost - 1) * 100 if frac_cost else 0,
                })
                lot["shares"] -= take
                lot["cost"] -= frac_cost
                remain -= take
                if lot["shares"] <= 0:
                    books[code].popleft()
    return pd.DataFrame(rts)


for name, d in loaded.items():
    tr = d["trades"]
    init = d["initial_capital"]
    rt = round_trips(tr)
    if rt.empty:
        print(f"\n{name}: 无完整回合")
        continue
    rt = rt.sort_values("pnl", ascending=False).reset_index(drop=True)
    total_pnl = rt["pnl"].sum()
    win = (rt["pnl"] > 0).mean()
    print(f"\n--- {name} ---")
    print(f"  完整回合 {len(rt)} 个 | 胜率 {win:.1%} | 净盈亏合计 ¥{total_pnl:,.0f} "
          f"(本金 ¥{init:,.0f})")
    for k in [1, 3, 5, 10]:
        if k <= len(rt):
            share = rt.head(k)["pnl"].sum() / total_pnl if total_pnl else np.nan
            print(f"    最赚的 {k:>2} 个回合贡献 {share:>6.1%} 的总盈亏")
    print(f"  最赚的 5 个回合:")
    for _, r in rt.head(5).iterrows():
        print(f"    {r['code']:<11} {r['buy_date']} -> {r['sell_date']} "
              f"¥{r['pnl']:>+9,.0f} ({r['ret_pct']:>+7.1f}%)")
    # 剔除最赚的 top-k 后的总收益
    print(f"  剔除最赚回合后的总收益(近似, 按盈亏/本金):")
    base = total_pnl / init * 100
    print(f"    原始           {base:>7.1f}%")
    for k in [1, 3, 5]:
        adj = (total_pnl - rt.head(k)["pnl"].sum()) / init * 100
        print(f"    剔除 top{k:<2}      {adj:>7.1f}%")

print()
print("=" * 74)
print("三、持仓重叠度 (两个配置是否其实买的是同一批股票)")
print("=" * 74)
if len(loaded) == 2:
    names = list(loaded)
    sets = {}
    for n in names:
        codes = defaultdict(float)
        for t in loaded[n]["trades"]:
            if t["action"] == "buy":
                codes[t["code"]] += -t["net"]
        sets[n] = codes
    a, b = sets[names[0]], sets[names[1]]
    ka, kb = set(a), set(b)
    print(f"  {names[0]}: 买过 {len(ka)} 只")
    print(f"  {names[1]}: 买过 {len(kb)} 只")
    print(f"  交集 {len(ka & kb)} 只 | Jaccard {len(ka&kb)/len(ka|kb):.1%}")
    print(f"\n  {names[0]} 买入金额最大的 8 只:")
    for c, v in sorted(a.items(), key=lambda x: -x[1])[:8]:
        mark = "" if c in kb else "   <-- 另一配置从未买过"
        print(f"    {c:<11} ¥{v:>9,.0f}{mark}")
