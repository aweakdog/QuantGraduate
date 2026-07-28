import os, json, pandas as pd, numpy as np
from pathlib import Path
from backtest import execution as ex

DATA_DIR = Path(os.environ["QUANT_DATA_DIR"])
TRADE_COST = 0.0006


def _load_klines_fast(data_dir, codes):
    cache = {}
    print(f"[klines] loading {len(codes)} codes...")
    for code in codes:
        code6 = str(code)[:6]
        path = data_dir / "raw" / "kline" / f"{code6}.parquet"
        if not path.exists():
            continue
        kline = pd.read_parquet(path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        kline.attrs["date_to_pos"] = {
            pd.Timestamp(d): j for j, d in enumerate(kline["date"].values)
        }
        cache[code] = kline
    print(f"[klines] loaded {len(cache)} klines")
    return cache


def _next_bar_fast(kline, signal_date):
    pos = kline.attrs["date_to_pos"].get(signal_date)
    if pos is None or pos + 1 >= len(kline):
        return None
    current = kline.iloc[pos + 1]
    previous = kline.iloc[pos]
    return {
        "date": current["date"],
        "index": pos + 1,
        "previous_close": float(previous["close"]),
        "open": float(current["open"]),
        "close": float(current["close"]),
    }


# Monkey-patch: skip limit up/down checks
ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast
ex._at_limit_up = lambda *a, **kw: False
ex._at_limit_down = lambda *a, **kw: False

from backtest.execution import simulate_portfolio

IN = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json"

d = json.load(open(IN))
rdf = pd.DataFrame(d["daily"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf["holdings"] = rdf["holdings"].apply(lambda x: [str(c) for c in x])
rdf["rand_holdings"] = rdf["rand_holdings"].apply(lambda x: [str(c) for c in x])


def run_cap(cap):
    print(f"\n=== nolimit cap={cap} ===")
    pf, exs = simulate_portfolio(rdf, DATA_DIR, cap, TRADE_COST, "holdings")
    fv = float(pf["portfolio_value"].iloc[-1])
    print(f"  final={round(fv,2)} return={round((fv/cap-1)*100,2)}%")
    print(f"  rejected_buy={exs['rejected_buy']} rejected_sell={exs['rejected_sell']}")
    print(f"  total_cost={exs['total_cost_paid']}")
    return pf, exs, fv


def build_portfolio_dict(pf, exs, fv, cap):
    return {
        "initial_capital": cap,
        "final_value": round(fv, 2),
        "total_return_pct": round((fv / cap - 1) * 100, 2),
        "avg_turnover_sold": round(float(pf["turnover_sold"].mean()), 2),
        "avg_turnover_bought": round(float(pf["turnover_bought"].mean()), 2),
        "daily_values": [
            {"date": r["date"], "value": r["portfolio_value"], "daily_ret": r["daily_ret"],
             "rolling_100d_ret_pct": r["rolling_100d_ret_pct"],
             "turnover_bought": r["turnover_bought"], "turnover_sold": r["turnover_sold"],
             "buy_cost": r["buy_cost"], "sell_cost": r["sell_cost"], "total_cost": r["total_cost"]}
            for _, r in pf.iterrows()
        ],
    }


for cap in [10000.0, 100000.0, 1000000.0]:
    pf, exs, fv = run_cap(cap)
    OUT = f"data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap{int(cap)}_nolimit.json"
    d2 = json.load(open(IN))
    d2["initial_capital"] = cap
    d2["portfolio"] = build_portfolio_dict(pf, exs, fv, cap)
    d2["execution"] = {"stats": {k: v for k, v in exs.items() if k != "trade_log"}, "trades": exs["trade_log"]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d2, f, indent=2, default=str, ensure_ascii=False)
    print(f"  saved {OUT}")
