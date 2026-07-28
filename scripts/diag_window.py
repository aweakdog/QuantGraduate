"""钻取某个时间窗内的逐笔操作, 定位跑输原因

用法:
  python scripts/diag_window.py --config breadth --start 2024-09-26 --end 2024-11-25
"""
import argparse
import glob
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_names():
    names = {}
    pm = ROOT / "data/universe/pit_metadata.parquet"
    if pm.exists():
        m = pd.read_parquet(pm)
        names = dict(zip(m["code"].astype(str).str.zfill(6), m["name"]))
    return names


def pair_trades(trades):
    pending = defaultdict(deque)
    rows = []
    for t in sorted(trades, key=lambda x: (x["date"], x["action"] != "sell")):
        code = str(t["code"])[:6]
        if t["action"] == "buy":
            pending[code].append(t)
            continue
        buy = pending[code].popleft() if pending[code] else None
        if buy is None:
            continue
        rows.append({"买入日": buy["date"], "卖出日": t["date"], "代码": code,
                     "买入价": buy["price"], "卖出价": t["price"],
                     "金额": round(buy["gross"]), "净收益": round(t["net"] + buy["net"], 2),
                     "收益率%": round((t["net"] + buy["net"]) / buy["gross"] * 100, 2)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="breadth")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()

    f = glob.glob(f"data/processed/wf_daily_pit_{a.config}_*_cap100000.json")[0]
    d = json.load(open(f, encoding="utf-8"))
    names = load_names()

    ops = pair_trades(d["trades"])
    ops["买入日"] = pd.to_datetime(ops["买入日"])
    w = ops[(ops["买入日"] >= a.start) & (ops["买入日"] <= a.end)].copy()
    w["名称"] = w["代码"].map(lambda c: names.get(c, ""))
    w = w[["买入日", "卖出日", "代码", "名称", "金额", "买入价", "卖出价", "净收益", "收益率%"]]
    w["买入日"] = w["买入日"].dt.strftime("%Y-%m-%d")

    print(f"窗口 {a.start} ~ {a.end} | 配置 {a.config} | {len(w)} 笔操作")
    print(w.to_string(index=False))
    print(f"\n净收益合计 {w['净收益'].sum():+,.0f} | 胜率 {(w['净收益']>0).mean()*100:.0f}% "
          f"| 平均收益率 {w['收益率%'].mean():+.2f}% | 中位 {w['收益率%'].median():+.2f}%")

    # 同期该股在池内的表现分位: 用训练集算窗口内每只票的累计收益
    df = pd.read_parquet(ROOT / "data/processed" / d["train_file"],
                         columns=["date", "code", "fwd_1d_ret"])
    df["date"] = pd.to_datetime(df["date"])
    m = df[(df["date"] >= a.start) & (df["date"] <= a.end)].dropna(subset=["fwd_1d_ret"])
    cum = m.groupby(m["code"].astype(str).str[:6])["fwd_1d_ret"].apply(lambda x: (1 + x).prod() - 1)
    picked = w["代码"].unique()
    print(f"\n窗口内池内个股累计收益分布: 中位 {cum.median()*100:+.1f}% "
          f"均值 {cum.mean()*100:+.1f}%")
    rank = cum.rank(pct=True)
    pr = [rank.get(c) for c in picked if c in rank.index]
    if pr:
        print(f"被选中 {len(pr)} 只股票的同期分位: 平均 {sum(pr)/len(pr)*100:.0f}% "
              f"(50% 即与池内中位一致)")
        tb = pd.DataFrame({"代码": picked,
                           "名称": [names.get(c, "") for c in picked],
                           "窗口累计%": [round(cum.get(c, float('nan')) * 100, 1) for c in picked],
                           "池内分位%": [round(rank.get(c, float('nan')) * 100) for c in picked]})
        print(tb.sort_values("窗口累计%").to_string(index=False))


if __name__ == "__main__":
    main()
