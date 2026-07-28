"""
Close-price execution with ALL realistic constraints:
- Integer lot (100 shares)
- Limit up/down checks
- Minimum 5 RMB commission
- T+1 close buy, T+2 close sell
"""
import os, json, pandas as pd, numpy as np
from pathlib import Path
from backtest import execution as ex

DATA_DIR = Path(os.environ["QUANT_DATA_DIR"])
TRADE_COST = 0.0006


def _load_klines_fast(data_dir, codes):
    cache = {}
    for code in codes:
        code6 = str(code)[:6]
        path = data_dir / "raw" / "kline" / f"{code6}.parquet"
        if not path.exists():
            continue
        kline = pd.read_parquet(path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        kline.attrs["date_to_pos"] = {pd.Timestamp(d): j for j, d in enumerate(kline["date"].values)}
        cache[code] = kline
    return cache


def _next_bar_fast(kline, signal_date):
    pos = kline.attrs["date_to_pos"].get(signal_date)
    if pos is None or pos + 1 >= len(kline):
        return None
    c = kline.iloc[pos + 1]
    p = kline.iloc[pos]
    return {"date": c["date"], "index": pos + 1, "previous_close": float(p["close"]),
            "open": float(c["open"]), "close": float(c["close"])}


ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast

# Patch simulate_portfolio: use bar["close"] instead of bar["open"] everywhere
# Keep integer lot, limit up/down, min 5 RMB fee
import inspect

src = inspect.getsource(ex.simulate_portfolio)
# Replace all bar["open"] with bar["close"]
src_close = src.replace('bar["open"]', 'bar["close"]')
# Fix the limit_up check: now uses close vs previous_close
# _at_limit_up checks price >= previous_close * (1 + limit - 0.001)
# With close price this means the stock closed at limit up (can't buy)
# _at_limit_down checks price <= previous_close * (1 - limit + 0.001)
# With close price this means the stock closed at limit down (can't sell)
# This is correct behavior for close execution
exec(src_close, ex.__dict__)
simulate_portfolio = ex.simulate_portfolio

IN = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json"
d = json.load(open(IN))
rdf = pd.DataFrame(d["daily"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf["holdings"] = rdf["holdings"].apply(lambda x: [str(c) for c in x])
rdf["rand_holdings"] = rdf["rand_holdings"].apply(lambda x: [str(c) for c in x])


def build_dict(pf, exs, fv, cap):
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


for cap in [1000000.0]:
    print(f"\n=== close exec, full constraints, cap={cap} ===")
    pf, exs = simulate_portfolio(rdf, DATA_DIR, cap, TRADE_COST, "holdings")
    fv = float(pf["portfolio_value"].iloc[-1])
    print(f"  final={round(fv,2)} return={round((fv/cap-1)*100,2)}%")
    print(f"  rejected_buy={exs['rejected_buy']} rejected_sell={exs['rejected_sell']}")
    print(f"  total_cost={exs['total_cost_paid']}")

    OUT = f"data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap{int(cap)}_close.json"
    d2 = json.load(open(IN))
    d2["initial_capital"] = cap
    d2["portfolio"] = build_dict(pf, exs, fv, cap)
    d2["execution"] = {"stats": {k: v for k, v in exs.items() if k != "trade_log"}, "trades": exs["trade_log"]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d2, f, indent=2, default=str, ensure_ascii=False)
    print(f"  saved {OUT}")
