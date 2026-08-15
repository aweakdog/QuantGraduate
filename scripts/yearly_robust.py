# -*- coding: utf-8 -*-
"""逐年稳健性 + 留一年检验 —— 判断一个配置是不是靠单一年份撑起来的

为什么必须做这个
────────────────
多种子中位数只能证明"不是抽样运气", 不能证明"不是单年运气"。
FBTR2 就栽在这一条: 3.9 年 +49.5% 看着不错, 剔掉 2025 单年就变 -3.9%。
一个只在某一年赚钱的配置, 向前看没有任何理由继续赚。

判据 (三条都要过才算稳):
  1. 逐年中位收益 —— 正收益年数应占多数
  2. 留一年检验 —— 去掉任何单独一年后, 剩余年份累计仍为正
  3. 多数种子同向 —— 每年为负的种子数不应接近全部

用法
────
    python scripts/yearly_robust.py V24A V24B
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed"


def load_tag(tag):
    """返回 (逐年收益 DataFrame[seed x year], summary DataFrame)"""
    rows, sums = [], []
    for f in sorted(glob.glob(str(BASE / f"wf_daily_{tag}_s*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "daily" not in d:
            continue
        dd = pd.DataFrame(d["daily"])
        if dd.empty or "portfolio_value" not in dd:
            continue
        dd["date"] = pd.to_datetime(dd["date"])
        dd["year"] = dd.date.dt.year
        r = {}
        for y, g in dd.groupby("year"):
            v0, v1 = g["portfolio_value"].iloc[0], g["portfolio_value"].iloc[-1]
            if v0 > 0:
                r[y] = v1 / v0 - 1
        rows.append(r)
        s = d.get("summary", {})
        sums.append({k: s.get(k) for k in
                     ["total_return_pct", "excess_annual_pct", "information_ratio",
                      "ic_tstat", "beta", "alpha_annual_pct", "benchmark_annual_pct"]})
    if not rows:
        return None, None
    return pd.DataFrame(rows).sort_index(axis=1), pd.DataFrame(sums)


def main():
    tags = sys.argv[1:] or ["V24A", "V24B"]
    store = {}
    for t in tags:
        y, s = load_tag(t)
        if y is None:
            print(f"{t}: 没有结果")
            continue
        store[t] = (y, s)
        print(f"{t}: {len(y)} 个种子, 年份 {list(y.columns)}")

    if not store:
        return
    years = sorted({y for _, (df, _) in store.items() for y in df.columns})

    print("\n===== 1. 逐年收益中位数 % (括号内 = 为负的种子数/总数) =====")
    hdr = "".join(f"{y:>16}" for y in years)
    print(f"{'配置':<8}{hdr}")
    for t, (df, _) in store.items():
        cells = []
        for y in years:
            if y not in df.columns:
                cells.append(f"{'-':>16}")
                continue
            s = df[y].dropna()
            cells.append(f"{s.median() * 100:>10.1f}({(s < 0).sum():>2}/{len(s)})")
        print(f"{t:<8}" + "".join(cells))

    print("\n===== 2. 留一年检验: 剔掉该年后, 其余年份累计收益中位数 % =====")
    print("     (任何一列为负 => 该配置靠那一年撑着)")
    print(f"{'配置':<8}{'全部年':>10}" + "".join(f"{'剔' + str(y):>10}" for y in years))
    for t, (df, _) in store.items():
        full = df.apply(lambda r: np.prod([1 + v for v in r.dropna()]) - 1, axis=1)
        cells = [f"{full.median() * 100:>10.1f}"]
        for y in years:
            if y not in df.columns:
                cells.append(f"{'-':>10}")
                continue
            sub = df.drop(columns=[y])
            loo = sub.apply(lambda r: np.prod([1 + v for v in r.dropna()]) - 1, axis=1)
            cells.append(f"{loo.median() * 100:>10.1f}")
        print(f"{t:<8}" + "".join(cells))

    print("\n===== 3. 超额与显著性 (中位数) =====")
    keys = ["benchmark_annual_pct", "excess_annual_pct", "information_ratio",
            "alpha_annual_pct", "beta", "ic_tstat"]
    print(f"{'配置':<8}" + "".join(f"{k[:16]:>18}" for k in keys))
    for t, (_, s) in store.items():
        cells = []
        for k in keys:
            v = pd.to_numeric(s[k], errors="coerce").dropna() if k in s else pd.Series(dtype=float)
            cells.append(f"{v.median():>18.3f}" if len(v) else f"{'-':>18}")
        print(f"{t:<8}" + "".join(cells))


if __name__ == "__main__":
    main()
