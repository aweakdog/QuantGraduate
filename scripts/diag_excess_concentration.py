"""超额收益的集中度 —— 判断"跑赢基准"是持续能力还是几天的运气。

为什么要看这个
──────────────
按年拆出来的超额里, 2025 是 +32%, 但日胜率只有 52%。这两个数放在一起就说明
超额不是每天一点点攒出来的, 而是少数几天贡献的。如果剔掉最好的 10 天(不到总
天数的 1%)超额就归零, 那这个策略的"能力"在统计上就等同于买彩票中了几次。

用线性累加(而非复利)口径做剔除, 是为了让"某天贡献了多少"这件事可加、可比。
复利下剔除某天会改变后续所有基数, 归因就不干净了。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grid import (  # noqa: E402
    DEFAULT_SEEDS, WINDOWS, VARIANTS, TRAIN_FILE, PIT_UNIVERSE,
    ev_tag, out_path_cap,
)
from export_v35_excel import benchmark_series  # noqa: E402

VARIANT = "mb_dmw"
CFGS = ["g5_rg_slip05", "g5_rg_slip05_ph1", "g5_rg_slip05_ph2",
        "g5_rg_slip05_ph3", "g5_rg_slip05_ph4"]
CAP = 50000.0


def main():
    tf = VARIANTS[VARIANT].get("train_file", TRAIN_FILE)
    rows = []
    for win in WINDOWS:
        cols = {}
        for c in CFGS:
            for s in DEFAULT_SEEDS[:20]:
                p = out_path_cap(ev_tag(c, win, s, VARIANT), win, CAP)
                if not p.exists():
                    continue
                dd = json.load(open(p))["daily"]
                cols[f"{c}_{s}"] = pd.Series(
                    {pd.Timestamp(x["date"]): x["daily_ret"] for x in dd})
        m = pd.DataFrame(cols).sort_index().median(axis=1)
        b = benchmark_series(m.index, train_file=tf,
                             pit_universe=PIT_UNIVERSE, skip_boards=())
        rows.append(pd.DataFrame({"strat": m.values, "bench": b, "win": win},
                                 index=m.index))
    d = pd.concat(rows).sort_index()
    d["ex"] = d["strat"] - d["bench"]

    print(f"全期 {d.index.min().date()} ~ {d.index.max().date()}  {len(d)} 天\n")

    print("超额收益集中度 (剔除最好的 N 天后, 累计超额还剩多少)")
    print(f"{'剔除天数':<10}{'占总天数':>10}{'剩余累计超额%':>16}")
    s = d["ex"].sort_values(ascending=False)
    for n in [0, 5, 10, 20, 40]:
        print(f"{n:<10}{n / len(d) * 100:>9.1f}%{s.iloc[n:].sum() * 100:>16.1f}")

    print("\n日超额的分布")
    print(f"{'范围':<8}{'天数':>7}{'胜率':>8}{'日均%':>10}{'t值':>8}{'偏度':>8}")
    for win in ["A", "B", "全期"]:
        g = d if win == "全期" else d[d.win == win]
        e = g["ex"]
        t = e.mean() / e.std() * np.sqrt(len(e))
        print(f"{win:<8}{len(e):>7}{100 * (e > 0).mean():>7.1f}%"
              f"{e.mean() * 100:>10.4f}{t:>8.2f}{e.skew():>8.2f}")

    print("\n贡献最大的 10 天")
    print(f"{'日期':<12}{'窗口':<6}{'策略%':>9}{'基准%':>9}{'超额%':>9}")
    for dt, r in d.loc[s.index[:10]].sort_index().iterrows():
        print(f"{dt.date()!s:<12}{r['win']:<6}{r['strat'] * 100:>9.2f}"
              f"{r['bench'] * 100:>9.2f}{r['ex'] * 100:>9.2f}")


if __name__ == "__main__":
    main()
