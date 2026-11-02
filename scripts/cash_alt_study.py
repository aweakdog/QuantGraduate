"""空仓期替代资产 / 做空股指研究 (11-02 预注册 C1/C2, 见 docs/experiment_board.md)。

  python scripts/cash_alt_study.py --grid-dir /home/yliog/quant-research-20260912-n2/data/grid \\
      --kline-dir /home/yliog/quant-strategy/data/raw/kline --fetch
策略空仓的日子 (regime 的 cash_day, 信号滞后 2 天 = 真实持仓口径) 里改持某个 ETF, 或做空股指 (ETF 反向收益,
按年化扣期货贴水), 对照同日子持有货币 ETF 511880。空仓段第一天进、最后一天出, 每次进出各扣 COST。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

FEASIBLE = {"511010": "5年国债", "511260": "10年国债", "518880": "黄金"}      # C1 判
DESCRIBE = {"511090": "30年国债", "510880": "红利"}                          # 只描述
SHORT = {"510300": ("沪深300", 0.02), "510500": ("中证500", 0.06), "512100": ("中证1000", 0.08)}   # C2
BASE = "511880"
COST = 0.0005
SPLITS = (("S1", "2017-01-01", "2021-12-31"), ("S2", "2022-01-01", "2099-12-31"))


def spells(cash):
    """连续空仓段 [(第一天, 最后一天)] (索引标签)。"""
    out, start, prev = [], None, None
    for d, c in cash.items():
        if c and start is None:
            start = d
        if not c and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def held_returns(ret, cash, cost=COST, sign=1, carry=0.0):
    """空仓日持有该资产的日收益 (非空仓日为 NaN); 每段首日、末日各扣一次 cost; carry 为年化成本 (按 252 天摊)。"""
    r = (sign * ret - carry / 252).where(cash)
    for a, b in spells(cash):
        r.loc[a] -= cost
        r.loc[b] -= cost
    return r


def evaluate(asset, base, cash):
    """asset/base: 空仓日收益 (其余 NaN)。返回超额统计 (相对 base, 只在两者都有数据的空仓日上算)。"""
    ok = cash & asset.notna() & base.notna()
    a, b = asset.where(ok), base.where(ok)
    res = {"cash_days": int(ok.sum())}
    if not ok.any():
        return res
    n = int(ok.sum())
    ga, gb = (1 + a.dropna()).prod(), (1 + b.dropna()).prod()
    res["ann_excess"] = round((ga / gb) ** (252 / n) - 1, 4)
    res["ann_asset"] = round(ga ** (252 / n) - 1, 4)
    for name, lo, hi in SPLITS:
        m = ok & (ok.index >= lo) & (ok.index <= hi)
        res[f"{name}_cum_excess"] = (round((1 + a[m]).prod() / (1 + b[m]).prod() - 1, 4) if m.any() else None)
    ex = []
    for s, e in spells(ok):
        seg = ok.loc[s:e]
        ex.append((1 + a.loc[s:e][seg]).prod() / (1 + b.loc[s:e][seg]).prod() - 1)
    res.update(spells=len(ex), worst_spell=round(min(ex), 4), win_rate=round(sum(x > 0 for x in ex) / len(ex), 3))
    return res


def gate_c1(r):
    return bool(r.get("S1_cum_excess") is not None and r["S1_cum_excess"] > 0 and r.get("S2_cum_excess", -1) > 0
                and r.get("ann_excess", -1) >= 0.015 and r.get("worst_spell", -1) >= -0.03 and r.get("win_rate", 0) >= 0.5)


def gate_c2(r):
    return bool(r.get("S1_cum_excess") is not None and r["S1_cum_excess"] > 0 and r.get("S2_cum_excess", -1) > 0
                and r.get("worst_spell", -1) >= -0.05)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--kline-dir", default=None, help="用生产 K 线刷新空仓序列到最新 (只读)")
    ap.add_argument("--fetch", action="store_true", help="补拉缺的 ETF 日线")
    ap.add_argument("--start", default="2017-01-01")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid_data
    g = Path(a.grid_dir)
    codes = [BASE, *FEASIBLE, *DESCRIBE, *SHORT]
    if a.fetch:
        end = pd.Timestamp.today().strftime("%Y%m%d")
        for c in codes:
            p = g / f"etf_{c}.parquet"
            if not p.exists() or pd.read_parquet(p).index.max() < pd.Timestamp.today() - pd.Timedelta(days=7):
                grid_data.fetch_etf(c, end).to_parquet(p)
    reg = grid_data.regime_series(a.kline_dir) if a.kline_dir else pd.read_parquet(g / "regime.parquet")
    reg = reg[reg.index >= a.start]
    cash = reg["cash_day"].astype(bool)
    rets = {c: pd.read_parquet(g / f"etf_{c}.parquet")["close"].pct_change().reindex(cash.index) for c in codes}
    base = held_returns(rets[BASE], cash, cost=0.0)
    out = {"window": [str(cash.index[0].date()), str(cash.index[-1].date())], "cash_share": round(float(cash.mean()), 3),
           "base": {"cash_days": int(cash.sum()),
                    "ann_money": round((1 + base.dropna()).prod() ** (252 / max(1, base.notna().sum())) - 1, 4)},
           "c1": {}, "describe": {}, "c2": {}}
    for c, name in FEASIBLE.items():
        r = evaluate(held_returns(rets[c], cash), base, cash)
        out["c1"][c] = {"name": name, **r, "pass": gate_c1(r)}
    for c, name in DESCRIBE.items():
        out["describe"][c] = {"name": name, **evaluate(held_returns(rets[c], cash), base, cash)}
    zero = pd.Series(0.0, index=cash.index).where(cash)
    for c, (name, carry) in SHORT.items():
        gross = evaluate(held_returns(rets[c], cash, sign=-1), zero, cash)
        net = evaluate(held_returns(rets[c], cash, sign=-1, carry=carry), zero, cash)
        out["c2"][c] = {"name": name, "carry": carry, "gross": gross, "net": net, "pass": gate_c2(net)}
    (g / "summary_cash_alt.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
