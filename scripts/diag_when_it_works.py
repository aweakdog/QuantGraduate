"""策略到底在什么市况下赚钱 —— 按年份和按市场状态分桶。

要回答的问题
────────────
"窗口B 赚钱, 是不是说明市场不特别差的时候我们能赚?"

这个问法有个陷阱: 窗口B(2022-09~2026-07)是历史上【反复用于选参数】的窗口,
窗口A(2020-07~2022-08)才是没调过参的。所以不能拿 B 的结果当能力证明。
更要命的是 A 的基准是 +3.1% —— 市场是走平的, 不是"特别差", 而策略在亏。

所以这里不按窗口看, 按【市场当期表现】看: 把每个交易日按基准的近20日收益分
成5档, 统计策略在每一档里的超额。如果"市场不差就能赚"成立, 那么中高档位上
的超额应该显著为正, 且 A/B 两窗口一致。

口径
────
基准 = 池内等权 fwd_1d_ret 平移一日, 与回测引擎第1220行完全一致。
注意 eval 阶段带 --load-preds 时引擎【不】对 df 做板块过滤(见 wf_v35 第644行),
所以产物里的基准是全市场等权而非主板等权。这里默认复刻该口径以便对齐; 用
--main-board 可切到主板等权做稳健性对照。

策略日收益取 20 个种子的【逐日中位】—— 不是先算各种子年化再取中位, 因为后者
会把不同种子在同一天的分散当成收益。

用法
────
    python scripts/diag_when_it_works.py --variant mb_dmw \\
        --configs g5_rg_slip05,g5_rg_slip05_ph1,g5_rg_slip05_ph2
"""
import argparse
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

CAP = 50000.0


def strat_daily(variant, cfgs, win, seeds):
    """(日期 -> 各种子日收益中位) 的 Series"""
    cols = {}
    for cname in cfgs:
        for s in seeds:
            p = out_path_cap(ev_tag(cname, win, s, variant), win, CAP)
            if not p.exists():
                continue
            d = json.load(open(p))["daily"]
            cols[f"{cname}_s{s}"] = pd.Series(
                {pd.Timestamp(x["date"]): x["daily_ret"] for x in d})
    if not cols:
        raise SystemExit(f"窗口{win}: 一个产物都没找到, 先跑 eval 阶段")
    m = pd.DataFrame(cols).sort_index()
    return m.median(axis=1), m.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mb_dmw", choices=list(VARIANTS))
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--main-board", action="store_true",
                    help="基准改用主板等权(与策略可买范围一致)")
    a = ap.parse_args()

    seeds = DEFAULT_SEEDS[:a.seeds]
    cfgs = a.configs.split(",")
    tf = VARIANTS[a.variant].get("train_file", TRAIN_FILE)
    skip = ("30", "688") if a.main_board else ()
    bench_desc = "主板等权" if a.main_board else "全市场等权"
    print(f"变体={a.variant}  配置={len(cfgs)}个  种子={len(seeds)}个  "
          f"基准={bench_desc}\n")

    allr = []
    for win in WINDOWS:
        sr, ncol = strat_daily(a.variant, cfgs, win, seeds)
        bench = benchmark_series(sr.index, train_file=tf,
                                 pit_universe=PIT_UNIVERSE, skip_boards=skip)
        d = pd.DataFrame({"strat": sr.values, "bench": bench}, index=sr.index)
        d["win"] = win
        allr.append(d)
        print(f"窗口 {win} ({WINDOWS[win]['desc']}): "
              f"{len(d)} 个交易日, {ncol} 条曲线取逐日中位")
    d = pd.concat(allr).sort_index()

    # ── 按年 ──
    print(f"\n{'=' * 78}\n按自然年 (策略 vs 基准, 单位%)\n{'=' * 78}")
    print(f"{'年份':<8}{'窗口':<6}{'交易日':>6}{'策略':>10}{'基准':>10}"
          f"{'超额':>10}{'胜天占比':>10}")
    for (yr, win), g in d.groupby([d.index.year, "win"]):
        s = (1 + g["strat"]).prod() - 1
        b = (1 + g["bench"]).prod() - 1
        wr = (g["strat"] > g["bench"]).mean() * 100
        print(f"{yr:<8}{win:<6}{len(g):>6}{s * 100:>10.1f}{b * 100:>10.1f}"
              f"{(s - b) * 100:>10.1f}{wr:>9.0f}%")

    # ── 按市场状态分桶 ──
    # 用【当日】基准收益分桶回答不了问题(那是事后诸葛), 要用可观测的近20日
    # 趋势 —— 这也贴近"市况好不好"的直觉。
    print(f"\n{'=' * 78}\n按市场状态分档 (基准近20日收益的5等分)\n{'=' * 78}")
    for win in list(WINDOWS) + ["合并"]:
        g = d if win == "合并" else d[d["win"] == win]
        m20 = (1 + g["bench"]).rolling(20).apply(np.prod, raw=True) - 1
        q = pd.qcut(m20.dropna(), 5,
                    labels=["最差20%", "偏差", "中性", "偏好", "最好20%"])
        gg = g.loc[q.index]
        m20 = m20.loc[q.index]
        print(f"\n-- {win} --")
        print(f"{'市场档位':<10}{'天数':>6}{'基准近20日':>12}{'策略日均':>11}"
              f"{'基准日均':>11}{'日均超额':>11}{'超额t值':>9}")
        for lab in ["最差20%", "偏差", "中性", "偏好", "最好20%"]:
            k = gg[q == lab]
            if k.empty:
                continue
            ex = k["strat"] - k["bench"]
            t = ex.mean() / ex.std() * np.sqrt(len(ex)) if ex.std() > 0 else 0
            print(f"{lab:<10}{len(k):>6}{m20[q == lab].mean() * 100:>11.1f}%"
                  f"{k['strat'].mean() * 100:>10.3f}%"
                  f"{k['bench'].mean() * 100:>10.3f}%"
                  f"{ex.mean() * 100:>10.3f}%{t:>9.2f}")


if __name__ == "__main__":
    main()
