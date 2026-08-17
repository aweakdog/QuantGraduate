# -*- coding: utf-8 -*-
"""回测交付物: 多种子权益曲线图 (含择时空仓阴影) + 每种子指标表

用法:
    python scripts/plot_backtest.py V24B5WF [--out docs/fig_V24B5WF.png]

图: 各种子权益曲线(细线) + 中位种子(粗线), 空仓日(in_cash)灰色阴影(中位种子口径),
    副轴画持仓数量。表: stdout 输出 markdown, 直接可贴文档。
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                   "PingFang SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

COLS = [("total_return_pct", "总收益%"), ("annualized_return_pct", "年化%"),
        ("sharpe", "夏普"), ("max_dd_pct", "回撤%"),
        ("excess_annual_pct", "超额年化%"), ("information_ratio", "IR"),
        ("total_cost_pct", "成本%"), ("cash_days_pct", "空仓日%")]


def load_tag(tag):
    files = sorted(glob.glob(str(PROC / f"wf_daily_{tag}_s*_ts*.json")))
    seeds = {}
    for f in files:
        d = json.load(open(f))
        s = int(Path(f).name.split("_s")[-1].split("_")[0])
        seeds[s] = d
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = load_tag(a.tag)
    if not seeds:
        raise SystemExit(f"找不到 {a.tag} 的结果文件")
    out = Path(a.out) if a.out else ROOT / "docs" / f"fig_{a.tag}.png"

    # ---- 表 ----
    rows = []
    for s in sorted(seeds):
        sm = seeds[s]["summary"]
        rows.append({"种子": s, **{cn: round(sm[k], 2) for k, cn in COLS}})
    df = pd.DataFrame(rows).set_index("种子")
    df.loc["均值"] = df.mean().round(2)
    d0, d1 = seeds[next(iter(seeds))]["daily"][0]["date"], \
        seeds[next(iter(seeds))]["daily"][-1]["date"]
    bench = seeds[next(iter(seeds))]["summary"].get("benchmark_total_pct")
    print(f"\n## {a.tag} 回测 ({d0} ~ {d1}, {len(seeds)} 种子, 基准同期 {bench}%)\n")
    print(df.to_markdown())

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    med_seed = df.drop(index="均值")["总收益%"].astype(float).sort_values().index[len(seeds) // 2]
    for s, d in sorted(seeds.items()):
        dd = pd.DataFrame(d["daily"])
        dd["date"] = pd.to_datetime(dd["date"])
        v = dd["portfolio_value"] / dd["portfolio_value"].iloc[0]
        if s == med_seed:
            ax.plot(dd["date"], v, lw=2.2, color="#c0392b", zorder=5,
                    label=f"中位种子 s{s} ({df.loc[s, '总收益%']:+.1f}%)")
            cash = dd["in_cash"].astype(bool).values
            ax.fill_between(dd["date"], 0, 1, where=cash, transform=ax.get_xaxis_transform(),
                            color="#95a5a6", alpha=0.18, label="择时空仓")
        else:
            ax.plot(dd["date"], v, lw=0.8, alpha=0.55)
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_title(f"{a.tag}  {d0} ~ {d1}  ({len(seeds)}种子, 空仓阴影=breadth择时)")
    ax.set_ylabel("净值 (期初=1)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
