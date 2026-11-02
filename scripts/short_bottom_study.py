"""做空模型排名末位 (SB, 11-02 预注册, 见 docs/experiment_board.md)。复用 N7_Q5 的 40 个预测缓存。

  python scripts/short_bottom_study.py --root /home/yliog/quant-research-20260912-n2 \\
      --kline-dir /home/yliog/quant-strategy/data/raw/kline
每个信号日取排名最后 N 只 (做空) / 最前 N 只 (做多对照), 次日收盘进、持 HOLD 天、HOLD 档轮换:
第 t 天的组合 = 信号日 D 满足 D+2 <= t <= D+HOLD+1 的各档 (按交易日序号算) 的等权平均。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_overnight_research as ro  # noqa: E402

N, HOLD = 5, 5
SIDE_COST = 0.0006 + 0.002          # 每边手续费 + 滑点 (与回测同)
BORROW = 0.08                       # 融券年费率
SPLIT = "2025-03-17"


def book_returns(picks, rets):
    """picks: {信号日: [代码...]}; rets: 日收益宽表 (行=交易日, 列=代码)。返回每天组合等权收益 (无持仓 NaN)。"""
    days = list(rets.index)
    pos = {d: i for i, d in enumerate(days)}
    acc = [[] for _ in days]
    for d, codes in picks.items():
        if d not in pos:
            continue
        i = pos[d]
        for j in range(i + 2, min(i + HOLD + 2, len(days))):
            r = rets.iloc[j].reindex(codes).dropna()
            if len(r):
                acc[j].append(float(r.mean()))
    return pd.Series([np.mean(a) if a else np.nan for a in acc], index=rets.index)


def net(book, sign):
    """sign=-1 做空: 扣每天一档进出的成本 (2 边 / HOLD 档) 和融券费; sign=+1 做多只扣交易成本。"""
    cost = 2 * SIDE_COST / HOLD + (BORROW / 244 if sign < 0 else 0.0)
    return (sign * book - cost).where(book.notna())


def load_preds(path):
    with open(path, "rb") as f:
        return pickle.load(f)["preds"]


def ann(r):
    r = r.dropna()
    return float((1 + r).prod() ** (244 / len(r)) - 1) if len(r) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kline-dir", required=True)
    a = ap.parse_args(argv)
    proc = Path(a.root) / "data/processed"
    first = load_preds(proc / ro.n6_caches(ro.N6_SEEDS[0])[1])
    codes = sorted({c for p in first for c in p["ranked"]})
    closes = {}
    for c in codes:
        f = Path(a.kline_dir) / f"{c[:6]}.parquet"
        if f.exists():
            k = pd.read_parquet(f, columns=["date", "close"])
            closes[c] = pd.Series(k["close"].astype(float).values, index=pd.to_datetime(k["date"]))
    px = pd.DataFrame(closes).sort_index()
    px = px[px.index >= pd.Timestamp(first[0]["date"]) - pd.Timedelta(days=10)]
    rets = px.pct_change()
    base = json.loads((proc / f"wf_daily_N7_Q5_s{ro.N6_SEEDS[0]}_ts2023-09-20_te2026-09-11_cap100000.json").read_text())["daily"]
    cash = pd.Series({pd.Timestamp(d["date"]): bool(d["in_cash"]) for d in base}).reindex(rets.index).fillna(False).astype(bool)
    rows, spell_rows = [], []
    spells, start, prev = [], None, None
    for d, c in cash.items():
        if c and start is None:
            start = d
        if not c and start is not None:
            spells.append((start, prev))
            start = None
        prev = d
    for s in ro.N6_SEEDS:
        preds = load_preds(proc / ro.n6_caches(s)[1])
        bottom = {pd.Timestamp(p["date"]): p["ranked"][-N:] for p in preds}
        top = {pd.Timestamp(p["date"]): p["ranked"][:N] for p in preds}
        b, t = book_returns(bottom, rets), book_returns(top, rets)
        sb, lt = net(b, -1), net(t, 1)
        row = {"seed": s, "short_gross_cash": ann(-b[cash]), "short_net_cash": ann(sb[cash]), "short_net_all": ann(sb),
               "long_net_cash": ann(lt[cash]), "long_net_invested": ann(lt[~cash]),
               "short_net_cash_h1": ann(sb[cash & (cash.index < SPLIT)]), "short_net_cash_h2": ann(sb[cash & (cash.index >= SPLIT)])}
        rows.append(row)
        spell_rows.append([float((1 + sb.loc[x:y][cash.loc[x:y]].dropna()).prod() - 1) for x, y in spells])
    t = pd.DataFrame(rows)
    sp = np.median(np.array(spell_rows), axis=0)
    med = {k: round(float(v) * 100, 2) for k, v in t.drop(columns="seed").median().items()}
    res = {"n_seeds": len(t), "cash_share": round(float(cash.mean()), 3), "median_ann_pct": med,
           "seeds_short_net_cash_pos": int((t["short_net_cash"] > 0).sum()), "n_spells": len(spells),
           "spell_pos_share": round(float((sp > 0).mean()), 3),
           "spells": [{"start": str(x.date()), "end": str(y.date()), "median_pct": round(float(m) * 100, 2)} for (x, y), m in zip(spells, sp, strict=True)]}
    res["pass"] = bool(med["short_net_cash_h1"] > 0 and med["short_net_cash_h2"] > 0 and res["seeds_short_net_cash_pos"] >= 26
                       and res["spell_pos_share"] >= 0.6)
    (Path(a.root) / "summary_short_bottom.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "spells"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
