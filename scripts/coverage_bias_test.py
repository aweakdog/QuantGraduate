# -*- coding: utf-8 -*-
"""覆盖偏差检验 —— 证明 FBTR +27.1% 来自事后选股而非资金流信息

背景
────
2x2 消融(FBTR3/4/5/6)显示: 锁定同一套 75 特征, 只把资金流源从 iFinD 换成 tushare,
20 种子中位总收益从 +27.1% 塌到 -14.5% (-23.7pp, 仅 2/20 种子改善)。
但 tushare 的 ts_lg_net 与旧源 dde_net 相关 0.911、同号 90.7% —— 信息本身没变。
那 -23.7pp 从哪来?

假设
────
旧源(iFinD)只覆盖 data/universe/watchlist_216.json —— 一份 2026-07-18 手工按题材
挑的 216 只概念股。于是老矩阵里 fund_flow_* 只有 243 只有值, 其余全是 NaN。
LightGBM 可以用"资金流是否有值"这个切分, 隐式地把选股范围锁进这 216 只。
若这批票本身就跑赢全 universe, 那这个 NaN 掩码本身就是 alpha —— 而它来自
名单挑选(用 2026 年的后见之明挑 2022 年的票), 不是来自资金流信息, 不可复制。

检验
────
直接比"有资金流值"与"全 universe"/"无值"三组的等权收益。不需要跑回测, 不需要逐笔。
若"有值"组显著跑赢, 假设成立。

用法: python scripts/coverage_bias_test.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "2022-09-01", "2026-08-07"


def load_codes(fn):
    p = ROOT / "data/universe" / fn
    if not p.exists():
        return None
    wl = json.load(open(p, encoding="utf-8"))
    items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
    return sorted({str(x["code"] if isinstance(x, dict) else x).split(".")[0].zfill(6)
                   for x in items})


def eq_weight_curve(codes):
    """等权买入持有的日收益序列(个股日收益的简单截面平均)"""
    rets = []
    for c in codes:
        f = ROOT / f"data/raw/kline/{c}.parquet"
        if not f.exists():
            continue
        k = pd.read_parquet(f, columns=["date", "close"])
        k["date"] = pd.to_datetime(k["date"])
        k = k[(k["date"] >= START) & (k["date"] <= END)].sort_values("date")
        if len(k) < 50:
            continue
        rets.append(k.set_index("date")["close"].pct_change().rename(c))
    return pd.concat(rets, axis=1).mean(axis=1) if rets else None


def main():
    wl = load_codes("watchlist_216.json") or load_codes("watchlist.json")
    print(f"旧源覆盖名单 watchlist_216: {len(wl)} 只")

    ff = pd.read_parquet(ROOT / "data/raw/fund_flow_full/fundflow_history.parquet",
                         columns=["code", "date", "dde_net"])
    ff["code"] = ff["code"].astype(str).str.zfill(6)
    have = sorted(ff.loc[ff["dde_net"].notna(), "code"].unique())
    print(f"实际 dde_net 有值: {len(have)} 只")

    uni = pd.read_parquet(ROOT / "data/universe/universe_pit_2019.parquet")
    uni_codes = sorted(uni["code"].astype(str).str.zfill(6).unique())
    overlap = sorted(set(have) & set(uni_codes))
    rest = sorted(set(uni_codes) - set(overlap))
    print(f"PIT universe {len(uni_codes)} 只, 其中有资金流值 {len(overlap)} 只 "
          f"({len(overlap)/len(uni_codes):.1%})\n计算等权收益...")

    groups = [(f"有资金流值 ({len(overlap)}只)", eq_weight_curve(overlap)),
              (f"全 universe ({len(uni_codes)}只)", eq_weight_curve(uni_codes)),
              (f"无资金流值 ({len(rest)}只)", eq_weight_curve(rest))]
    curves = []
    for name, r in groups:
        if r is None:
            continue
        cum = (1 + r.fillna(0)).prod() - 1
        ann = (1 + cum) ** (252 / len(r)) - 1
        print(f"  {name:22s} 总收益 {cum:+7.1%}  年化 {ann:+6.1%}  "
              f"夏普 {r.mean()/r.std()*np.sqrt(252):5.2f}")
        curves.append((name, r))

    if len(curves) >= 2:
        ex = curves[0][1] - curves[1][1]
        print(f"\n  [有值] 相对 [全universe] 年化超额 {ex.mean()*252:+.1%}  "
              f"IR {ex.mean()/ex.std()*np.sqrt(252):.2f}")
        print("\n逐年:")
        print(f"{'年':>6} {'有值':>9} {'全universe':>11} {'无值':>9} {'有值-全':>9}")
        for y in range(pd.Timestamp(START).year, pd.Timestamp(END).year + 1):
            v = [(1 + r[r.index.year == y].fillna(0)).prod() - 1 for _, r in curves]
            print(f"{y:>6} {v[0]:>+8.1%} {v[1]:>+10.1%} {v[2]:>+8.1%} {v[0]-v[1]:>+8.1%}")
        print("\n若'有值'组显著跑赢 -> NaN 掩码本身即 alpha, FBTR +27.1% 不可信。")


if __name__ == "__main__":
    main()
