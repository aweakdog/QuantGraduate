"""把"最近一周超额 +2.4%"放回历史分布里校准。

要防的错误
──────────
短期超额本身波动极大。一个长期没有 alpha 的策略, 照样会经常出现几天大幅跑赢
的片段 —— 尤其是当我们【因为它好看才去看它】的时候(在结果上做选择)。
所以唯一有意义的问题不是"这几天赚了没有", 而是:

    在这个策略的历史超额分布里, 一段 N 天 +X% 的超额, 处在什么分位?

如果它落在 70% 分位, 那它就是常态波动, 不含任何"最近变好了"的信息。

同时检查最近几天的推荐是不是同一批股票 —— 如果是, 那么"连续 3 天 100% 上涨"
其实只是【一次】押注被重复计了 3 遍, 有效样本量是 1 不是 3。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LIVE = ROOT / "data" / "live"

from eval_grid import (  # noqa: E402
    DEFAULT_SEEDS, WINDOWS, VARIANTS, TRAIN_FILE, PIT_UNIVERSE,
    ev_tag, out_path_cap,
)
from export_v35_excel import benchmark_series  # noqa: E402

CFGS = ["g5_rg_slip05", "g5_rg_slip05_ph1", "g5_rg_slip05_ph2",
        "g5_rg_slip05_ph3", "g5_rg_slip05_ph4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mb_dmw", choices=list(VARIANTS))
    ap.add_argument("--excess", type=float, default=2.40,
                    help="要校准的那段超额(%)")
    ap.add_argument("--hold", type=int, default=4, help="那段有几个交易日")
    ap.add_argument("--profile", default="base5w_steady")
    ap.add_argument("--topn", type=int, default=5)
    a = ap.parse_args()

    # ── 1. 最近推荐的重合度 ──
    plans = sorted(glob.glob(str(LIVE / f"plan_{a.profile}_[0-9]*.json")))[-6:]
    picks = {}
    for p in plans:
        d = json.load(open(p))
        picks[d["signal_date"]] = [str(x["code"]).zfill(6)
                                   for x in d["recommend"]
                                   if not x.get("blocked")][:a.topn]
    print(f"最近 {len(picks)} 个信号日的推荐前{a.topn}")
    for k, v in picks.items():
        print(f"  {k}: {' '.join(v)}")
    days = list(picks)
    if len(days) > 1:
        print("\n相邻两日重合数:")
        for i in range(1, len(days)):
            ov = set(picks[days[i]]) & set(picks[days[i - 1]])
            print(f"  {days[i - 1]} -> {days[i]}: {len(ov)}/{a.topn} 只相同"
                  f"  {' '.join(sorted(ov))}")
        allsets = [set(v) for v in picks.values()]
        union = set().union(*allsets)
        print(f"  全部 {len(days)} 天合计只涉及 {len(union)} 只不同股票")

    # ── 2. 历史分布校准 ──
    tf = VARIANTS[a.variant].get("train_file", TRAIN_FILE)
    rows = []
    for win in WINDOWS:
        cols = {}
        for c in CFGS:
            for s in DEFAULT_SEEDS[:20]:
                fp = out_path_cap(ev_tag(c, win, s, a.variant), win, 50000.0)
                if not fp.exists():
                    continue
                dd = json.load(open(fp))["daily"]
                cols[f"{c}_{s}"] = pd.Series(
                    {pd.Timestamp(x["date"]): x["daily_ret"] for x in dd})
        m = pd.DataFrame(cols).sort_index().median(axis=1)
        b = benchmark_series(m.index, train_file=tf,
                             pit_universe=PIT_UNIVERSE, skip_boards=())
        rows.append(pd.DataFrame({"ex": m.values - b}, index=m.index))
    d = pd.concat(rows).sort_index()

    # 滚动 hold 日累计超额(线性累加, 与短窗口下的复利几乎无差)
    roll = d["ex"].rolling(a.hold).sum() * 100
    roll = roll.dropna()
    pct = (roll < a.excess).mean() * 100
    print(f"\n{'=' * 70}")
    print(f"历史上任意 {a.hold} 个交易日的累计超额分布 "
          f"(n={len(roll)}, 2020-07~2026-07)")
    print(f"{'=' * 70}")
    print(f"{'分位':<10}{'累计超额%':>12}")
    for q in [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"{f'{int(q * 100)}%':<10}{roll.quantile(q):>12.2f}")
    print(f"\n最近这段 {a.excess:+.2f}% 落在第 {pct:.0f} 百分位")
    print(f"历史上有 {100 - pct:.0f}% 的 {a.hold} 日窗口超额不低于它")
    print(f"(该策略同期长期日均超额 t 值仅 {d['ex'].mean() / d['ex'].std() * np.sqrt(len(d)):.2f}"
          f" —— 即长期无显著 alpha, 却仍频繁出现这种量级的短期跑赢)")


if __name__ == "__main__":
    main()
