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
    return {"date": c["date"], "index": pos + 1, "previous_close": float(p["close"]), "open": float(c["open"]), "close": float(c["close"])}


ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast
ex._at_limit_up = lambda *a, **kw: False
ex._at_limit_down = lambda *a, **kw: False

# Patch simulate_portfolio to allow fractional shares (no integer lot)
import types, inspect

src = inspect.getsource(ex.simulate_portfolio)
# Replace the integer-lot buying block with fractional allocation
src_nolot = src.replace(
    """        bought_today = 0
        if not positions and eligible:
            allocation = cash / len(eligible)
            for code, bar in eligible:
                one_lot = bar["open"] * 100
                lot_fee = max(one_lot * trade_cost, 5.0)
                lot_cost = one_lot + lot_fee
                shares = int(allocation / lot_cost) * 100
                if shares <= 0:
                    rejected_buy += 1
                    trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY_REJECTED", "timing": "open", "code": code, "shares": 0, "price": bar["open"], "reason": "insufficient_one_lot"})
                    continue
                gross = shares * bar["open"]
                buy_fee = max(gross * trade_cost, 5.0)
                buy_cost_paid += buy_fee
                buy_cost_today += buy_fee
                cash -= gross + buy_fee
                positions[code] = shares
                bought_today += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "gross_amount": round(gross, 2), "reason": "integer_lot"})""",
    """        bought_today = 0
        if not positions and eligible:
            allocation = cash / len(eligible)
            for code, bar in eligible:
                gross = allocation
                buy_fee = max(gross * trade_cost, 5.0)
                if gross + buy_fee > cash:
                    buy_fee = gross * trade_cost
                if gross <= 0:
                    rejected_buy += 1
                    trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY_REJECTED", "timing": "open", "code": code, "shares": 0, "price": bar["open"], "reason": "insufficient_funds"})
                    continue
                shares = gross / bar["open"]
                buy_cost_paid += buy_fee
                buy_cost_today += buy_fee
                cash -= gross + buy_fee
                positions[code] = shares
                bought_today += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "open", "code": code, "shares": round(shares, 4), "price": bar["open"], "gross_amount": round(gross, 2), "reason": "fractional"})""",
)

# Also patch the final sell block to not check limit_down
src_nolot = src_nolot.replace(
    """            if _at_limit_down(code, bar["index"], bar["previous_close"], bar["open"]):
                rejected_sell += 1
                trade_log.append({"signal_date": str(last_signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL_REJECTED", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "reason": "limit_down_open"})
                final_value += shares * bar["open"]
                continue""",
    """            # no limit_down check""",
)

exec(src_nolot, ex.__dict__)
simulate_portfolio = ex.simulate_portfolio

IN = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json"
d = json.load(open(IN))
rdf = pd.DataFrame(d["daily"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf["holdings"] = rdf["holdings"].apply(lambda x: [str(c) for c in x])
rdf["rand_holdings"] = rdf["rand_holdings"].apply(lambda x: [str(c) for c in x])


def run_cap(cap, tag):
    print(f"\n=== nolot_nolimit cap={cap} ===")
    pf, exs = simulate_portfolio(rdf, DATA_DIR, cap, TRADE_COST, "holdings")
    fv = float(pf["portfolio_value"].iloc[-1])
    print(f"  final={round(fv,2)} return={round((fv/cap-1)*100,2)}%")
    print(f"  rejected_buy={exs['rejected_buy']} rejected_sell={exs['rejected_sell']}")
    print(f"  total_cost={exs['total_cost_paid']}")
    return pf, exs, fv


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


for cap in [10000.0, 100000.0, 1000000.0]:
    pf, exs, fv = run_cap(cap, "")
    OUT = f"data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap{int(cap)}_nolot.json"
    d2 = json.load(open(IN))
    d2["initial_capital"] = cap
    d2["portfolio"] = build_dict(pf, exs, fv, cap)
    d2["execution"] = {"stats": {k: v for k, v in exs.items() if k != "trade_log"}, "trades": exs["trade_log"]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(d2, f, indent=2, default=str, ensure_ascii=False)
    print(f"  saved {OUT}")
