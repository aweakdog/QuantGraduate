import os, json, pandas as pd, numpy as np
from pathlib import Path
from backtest import execution as ex

DATA_DIR = Path(os.environ["QUANT_DATA_DIR"])
TRADE_COST = 0.0006
INIT_CAP = 1000000.0


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


ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast

from backtest.execution import simulate_portfolio

IN = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json"
OUT = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap1000000.json"

print("[load] reading interval 4 predictions")
d = json.load(open(IN))
rdf = pd.DataFrame(d["daily"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf["holdings"] = rdf["holdings"].apply(lambda x: [str(c) for c in x])
rdf["rand_holdings"] = rdf["rand_holdings"].apply(lambda x: [str(c) for c in x])

print("[sim] holdings 100000")
pf_portfolio, execution_stats = simulate_portfolio(rdf, DATA_DIR, INIT_CAP, TRADE_COST, "holdings")
final_val = float(pf_portfolio["portfolio_value"].iloc[-1])
print("[sim] final", round(final_val, 2), "return", round((final_val / INIT_CAP - 1) * 100, 2), "%")

print("[sim] random baseline 100000")
pf_rand, random_execution_stats = simulate_portfolio(rdf, DATA_DIR, INIT_CAP, TRADE_COST, "rand_holdings")
rand_final = float(pf_rand["portfolio_value"].iloc[-1])

rand_sharpe = float(pf_rand["daily_ret"].mean() / pf_rand["daily_ret"].std() * np.sqrt(252)) if pf_rand["daily_ret"].std() > 0 else 0
rand_win = float((pf_rand["daily_ret"] > 0).mean())
rand_cum = float((1 + pf_rand["daily_ret"]).prod() - 1)
rand_ann = float((1 + pf_rand["daily_ret"]).prod() ** (252 / len(pf_rand)) - 1) if len(pf_rand) > 0 else 0
rand_cum_series = (1 + pf_rand["daily_ret"]).cumprod()
rand_dd = float((rand_cum_series / rand_cum_series.expanding().max() - 1).min())

d["initial_capital"] = INIT_CAP
d["portfolio"] = {
    "initial_capital": INIT_CAP,
    "final_value": round(final_val, 2),
    "total_return_pct": round((final_val / INIT_CAP - 1) * 100, 2),
    "avg_turnover_sold": round(float(pf_portfolio["turnover_sold"].mean()), 2),
    "avg_turnover_bought": round(float(pf_portfolio["turnover_bought"].mean()), 2),
    "daily_values": [
        {"date": r["date"], "value": r["portfolio_value"], "daily_ret": r["daily_ret"],
         "rolling_100d_ret_pct": r["rolling_100d_ret_pct"],
         "turnover_bought": r["turnover_bought"], "turnover_sold": r["turnover_sold"],
         "buy_cost": r["buy_cost"], "sell_cost": r["sell_cost"], "total_cost": r["total_cost"]}
        for _, r in pf_portfolio.iterrows()
    ],
}
d["execution"] = {
    "stats": {k: v for k, v in execution_stats.items() if k != "trade_log"},
    "trades": execution_stats["trade_log"],
}
d["random_baseline"] = {
    "final_value": round(rand_final, 2),
    "total_return_pct": round((rand_final / INIT_CAP - 1) * 100, 2),
    "sharpe": round(rand_sharpe, 2),
    "max_dd_pct": round(rand_dd * 100, 1),
    "win_rate_pct": round(rand_win * 100, 1),
    "cum_return_pct": round(rand_cum * 100, 1),
    "annualized_return_pct": round(rand_ann * 100, 1),
    "daily_values": [
        {"date": r["date"], "value": r["portfolio_value"], "daily_ret": r["daily_ret"],
         "rolling_100d_ret_pct": r["rolling_100d_ret_pct"]}
        for _, r in pf_rand.iterrows()
    ],
}
d["random_execution"] = {
    "stats": {k: v for k, v in random_execution_stats.items() if k != "trade_log"},
    "trades": random_execution_stats["trade_log"],
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, default=str, ensure_ascii=False)
print("[save]", OUT)
