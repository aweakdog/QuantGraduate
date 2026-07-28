"""诊断空仓择时的对错: 逐月拆解 + 每一段空仓/满仓区间的基准表现

回答三个问题:
  1. 超额收益是哪几个月亏掉的
  2. 每次切换到空仓, 后面这段基准是跌(躲对了)还是涨(踏空了)
  3. 空仓期 vs 满仓期, 基准平均日收益哪个高 —— 判断开关方向是否反了

用法:
  python scripts/analyze_regime_timing.py                      # 默认 breadth
  python scripts/analyze_regime_timing.py --config off --year 2024
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def benchmark_series(train_file, pit_universe, dates):
    df = pd.read_parquet(ROOT / "data/processed" / train_file,
                         columns=["date", "code", "fwd_1d_ret"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_1d_ret"])
    if pit_universe:
        u = pd.read_parquet(ROOT / "data/universe" / pit_universe)
        u["effective_date"] = pd.to_datetime(u["effective_date"])
        eff = np.array(sorted(u["effective_date"].unique()))
        members = {d: set(g["code"].astype(str).str.zfill(6))
                   for d, g in u.groupby("effective_date")}
        c6 = df["code"].astype(str).str[:6]
        per = np.searchsorted(eff, df["date"].values, side="right") - 1
        keep = np.zeros(len(df), bool)
        for i, d in enumerate(eff):
            m = per == i
            if m.any():
                keep[m] = c6[m].isin(members[pd.Timestamp(d)]).values
        df = df[keep]
    b = df.groupby("date")["fwd_1d_ret"].mean().shift(1)
    return b.reindex(pd.DatetimeIndex(dates)).fillna(0.0)


def load(config):
    hits = glob.glob(f"data/processed/wf_daily_pit_{config}_*_cap100000.json")
    if not hits:
        raise SystemExit(f"没有 {config} 的结果")
    d = json.load(open(hits[0], encoding="utf-8"))
    dd = pd.DataFrame(d["daily"])
    dd["date"] = pd.to_datetime(dd["date"])
    dd["bench"] = benchmark_series(d["train_file"], d.get("pit_universe"), dd["date"]).values
    dd["excess"] = dd["daily_ret"] - dd["bench"]
    if "in_cash" not in dd.columns:
        dd["in_cash"] = False
    dd["in_cash"] = dd["in_cash"].astype(bool)
    return d, dd


def monthly(dd, year=None):
    g = dd if year is None else dd[dd["date"].dt.year == year]
    rows = []
    for m, x in g.groupby(g["date"].dt.to_period("M")):
        s = (1 + x["daily_ret"]).prod() - 1
        b = (1 + x["bench"]).prod() - 1
        rows.append({"月份": str(m), "交易日": len(x),
                     "策略%": round(s * 100, 1), "基准%": round(b * 100, 1),
                     "超额%": round((s - b) * 100, 1),
                     "空仓天": int(x["in_cash"].sum()),
                     "空仓占比%": round(x["in_cash"].mean() * 100),
                     "IC": round(x["ic"].mean(), 3) if "ic" in x else None})
    return pd.DataFrame(rows)


def segments(dd):
    """把逐日 in_cash 切成连续区间, 统计每段基准收益"""
    dd = dd.reset_index(drop=True)
    grp = (dd["in_cash"] != dd["in_cash"].shift()).cumsum()
    rows = []
    for _, x in dd.groupby(grp):
        b = (1 + x["bench"]).prod() - 1
        s = (1 + x["daily_ret"]).prod() - 1
        state = "空仓" if x["in_cash"].iloc[0] else "满仓"
        if state == "空仓":
            verdict = "躲对了" if b < 0 else "踏空"
        else:
            verdict = "抓住了" if s > b else "跑输"
        rows.append({"状态": state, "起": x["date"].iloc[0].strftime("%Y-%m-%d"),
                     "止": x["date"].iloc[-1].strftime("%Y-%m-%d"), "天数": len(x),
                     "基准%": round(b * 100, 1), "策略%": round(s * 100, 1),
                     "超额%": round((s - b) * 100, 1), "判定": verdict})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breadth")
    ap.add_argument("--year", type=int, default=None)
    a = ap.parse_args()

    d, dd = load(a.config)
    print(f"配置 {a.config} | {d['period']} | 空仓 {dd['in_cash'].sum()} 天")

    print("\n=== 逐月拆解 ===")
    print(monthly(dd, a.year).to_string(index=False))

    seg = segments(dd)
    print(f"\n=== 空仓/满仓区间 (共 {len(seg)} 段) ===")
    cash = seg[seg["状态"] == "空仓"]
    if len(cash):
        right = (cash["基准%"] < 0).sum()
        print(f"空仓 {len(cash)} 段: 躲对 {right} 段, 踏空 {len(cash)-right} 段 "
              f"(胜率 {right/len(cash)*100:.0f}%)")
        print(f"  空仓期基准累计: {(cash['基准%']/100 + 1).prod()*100-100:+.1f}%  "
              f"<- 为负说明整体躲对")
        print("  踏空最惨的 5 段:")
        print(cash.nlargest(5, "基准%").to_string(index=False))
        print("  躲对最多的 5 段:")
        print(cash.nsmallest(5, "基准%").to_string(index=False))

    full = seg[seg["状态"] == "满仓"]
    if len(full):
        w = (full["超额%"] > 0).sum()
        print(f"\n满仓 {len(full)} 段: 跑赢 {w} 段, 跑输 {len(full)-w} 段 "
              f"(胜率 {w/len(full)*100:.0f}%)")

    print("\n=== 开关方向检验 ===")
    c, f = dd[dd["in_cash"]], dd[~dd["in_cash"]]
    print(f"空仓日 {len(c)} 天: 基准平均日收益 {c['bench'].mean()*100:+.3f}%")
    print(f"满仓日 {len(f)} 天: 基准平均日收益 {f['bench'].mean()*100:+.3f}%  "
          f"策略平均 {f['daily_ret'].mean()*100:+.3f}%")
    verdict = "开关方向正确(空仓日基准更弱)" if c["bench"].mean() < f["bench"].mean() \
        else "开关方向反了(空仓日基准反而更强)"
    print(f"-> {verdict}")

    if a.year:
        y = dd[dd["date"].dt.year == a.year]
        seg_y = segments(y)
        print(f"\n=== {a.year} 年区间明细 ===")
        print(seg_y.to_string(index=False))


if __name__ == "__main__":
    main()
