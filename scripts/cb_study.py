"""可转债双低轮动回测 (CB1 预注册, 见 docs/experiment_board.md)。

  python scripts/cb_study.py --panel /home/yliog/quant-research-20260912-n2/data/cb/cb_panel.parquet
每 R 个交易日按 双低 = 价格 + 转股溢价率(%) 取最低 N 只等权; 信号日收盘算、次日收盘成交 (第 D+2 天起计收益);
换仓按换掉的比例扣两边成本。对照 = 前一日全部可交易转债等权 (不扣费)。2018-21 选参, 2022 至今只检验选中的 1 组。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_GRID, R_GRID = (10, 20, 30), (5, 10, 20)
SEL, TEST = ("2018-01-01", "2021-12-31"), ("2022-01-01", "2099-12-31")
COST = 0.0002 + 0.001
MAX_PRICE, MIN_AGE = 150.0, 5


def prepare(panel):
    close = panel.pivot_table(index="date", columns="code", values="close").sort_index()
    prem = panel.pivot_table(index="date", columns="code", values="premium").reindex_like(close)
    ret = close.pct_change(fill_method=None)
    age = close.notna().cumsum()
    elig = close.notna() & prem.notna() & (close <= MAX_PRICE) & (age >= MIN_AGE)
    return ret, (close + prem).where(elig), elig


def backtest(ret, score, n, r, start, end):
    """返回 (逐日收益, 换手次数)。持仓在信号日后第 2 天起生效; 停牌/退市当天收益记 0。"""
    days = ret.index[(ret.index >= start) & (ret.index <= end)]
    idx = {d: i for i, d in enumerate(ret.index)}
    out, hold, pending, k = [], [], None, 0
    for t, d in enumerate(days):
        i = idx[d]
        cost = 0.0
        if pending is not None and pending[0] == i:          # 信号后第 2 天开始拿新组合 (次日收盘成交)
            new = pending[1]
            changed = len(set(new) - set(hold)) / max(1, len(new)) if hold else 1.0
            cost = 2 * COST * changed if hold else COST
            hold, pending = new, None
        rr = ret.iloc[i][hold].fillna(0.0).mean() if hold else 0.0
        out.append(rr - cost)
        if t % r == 0 and i + 2 < len(ret.index):
            s = score.iloc[i].dropna().nsmallest(n)
            if len(s):
                pending = (i + 2, list(s.index))
                k += 1
    return pd.Series(out, index=days), k


def benchmark(ret, elig, start, end):
    w = elig.shift(1, fill_value=False)
    b = ret.where(w).mean(axis=1).fillna(0.0)
    return b[(b.index >= start) & (b.index <= end)]


def metrics(r):
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    yrs = {str(y): round(float((1 + g).prod() - 1), 4) for y, g in r.groupby(r.index.year)}
    return {"ann": round(float(eq.iloc[-1] ** (244 / len(r)) - 1), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(244)), 3) if r.std() > 0 else 0.0,
            "mdd": round(float((eq / eq.cummax() - 1).min()), 4), "years": yrs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    a = ap.parse_args(argv)
    ret, score, elig = prepare(pd.read_parquet(a.panel))
    res = {"bench_sel": metrics(benchmark(ret, elig, *SEL)), "bench_test": metrics(benchmark(ret, elig, *TEST)), "grid": {}}
    for n in N_GRID:
        for r in R_GRID:
            sel, _ = backtest(ret, score, n, r, *SEL)
            test, _ = backtest(ret, score, n, r, *TEST)
            res["grid"][f"N{n}_R{r}"] = {"sel": metrics(sel), "test": metrics(test)}
    best = max(res["grid"], key=lambda k: res["grid"][k]["sel"]["sharpe"])
    bt, tt = res["bench_test"], res["grid"][best]["test"]
    yrs = [y for y in tt["years"] if y >= "2022"]
    excess_years = sum(tt["years"][y] > bt["years"].get(y, 0) for y in yrs)
    neigh = sum(g["test"]["ann"] > bt["ann"] for g in res["grid"].values())
    res["chosen"] = best
    res["gates"] = {"excess_ann": round(tt["ann"] - bt["ann"], 4), "sharpe_gap": round(tt["sharpe"] - bt["sharpe"], 3),
                    "mdd_gap": round(tt["mdd"] - bt["mdd"], 4), "excess_years": f"{excess_years}/{len(yrs)}",
                    "neighbors_positive": f"{neigh}/9"}
    g = res["gates"]
    res["pass"] = bool(g["excess_ann"] >= 0.03 and g["sharpe_gap"] >= 0.2 and g["mdd_gap"] >= -0.03
                       and excess_years >= 3 and neigh >= 6)
    Path(a.panel).with_name("summary_cb1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "grid"} | {"grid_test_ann": {k: v["test"]["ann"] for k, v in res["grid"].items()},
                      "grid_sel_sharpe": {k: v["sel"]["sharpe"] for k, v in res["grid"].items()}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
