"""
Compare open vs close execution timing.
Patches execution.py to use close price instead of open for buy/sell/marking.
Also tests "same-day close buy, next-day close sell" (option 2).
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
ex._at_limit_up = lambda *a, **kw: False
ex._at_limit_down = lambda *a, **kw: False

# Patch: use fractional shares (no integer lot) + close price execution
import inspect

src = inspect.getsource(ex.simulate_portfolio)

# Replace all bar["open"] with bar["close"] for execution and marking
src_close = src.replace('bar["open"]', 'bar["close"]')

# Replace integer-lot with fractional
src_close = src_close.replace(
    """        bought_today = 0
        if not positions and eligible:
            allocation = cash / len(eligible)
            for code, bar in eligible:
                one_lot = bar["close"] * 100
                lot_fee = max(one_lot * trade_cost, 5.0)
                lot_cost = one_lot + lot_fee
                shares = int(allocation / lot_cost) * 100
                if shares <= 0:
                    rejected_buy += 1
                    trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY_REJECTED", "timing": "open", "code": code, "shares": 0, "price": bar["close"], "reason": "insufficient_one_lot"})
                    continue
                gross = shares * bar["close"]
                buy_fee = max(gross * trade_cost, 5.0)
                buy_cost_paid += buy_fee
                buy_cost_today += buy_fee
                cash -= gross + buy_fee
                positions[code] = shares
                bought_today += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "open", "code": code, "shares": shares, "price": bar["close"], "gross_amount": round(gross, 2), "reason": "integer_lot"})""",
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
                    trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY_REJECTED", "timing": "close", "code": code, "shares": 0, "price": bar["close"], "reason": "insufficient_funds"})
                    continue
                shares = gross / bar["close"]
                buy_cost_paid += buy_fee
                buy_cost_today += buy_fee
                cash -= gross + buy_fee
                positions[code] = shares
                bought_today += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "close", "code": code, "shares": round(shares, 4), "price": bar["close"], "gross_amount": round(gross, 2), "reason": "fractional_close"})""",
)

# Remove limit_down check in final sell block
src_close = src_close.replace(
    """            if _at_limit_down(code, bar["index"], bar["previous_close"], bar["close"]):
                rejected_sell += 1
                trade_log.append({"signal_date": str(last_signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL_REJECTED", "timing": "open", "code": code, "shares": shares, "price": bar["close"], "reason": "limit_down_open"})
                final_value += shares * bar["close"]
                continue""",
    """            # no limit_down check""",
)

exec(src_close, ex.__dict__)
simulate_portfolio_close = ex.simulate_portfolio

# Also get the original (open) fractional version for comparison
src_open = src.replace(
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
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "open", "code": code, "shares": round(shares, 4), "price": bar["open"], "gross_amount": round(gross, 2), "reason": "fractional_open"})""",
)
src_open = src_open.replace(
    """            if _at_limit_down(code, bar["index"], bar["previous_close"], bar["open"]):
                rejected_sell += 1
                trade_log.append({"signal_date": str(last_signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL_REJECTED", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "reason": "limit_down_open"})
                final_value += shares * bar["open"]
                continue""",
    """            # no limit_down check""",
)
exec(src_open, ex.__dict__)
simulate_portfolio_open = ex.simulate_portfolio


IN = "data/processed/wf_daily_v25_w100_ts2023-01-01_te2026-07-16_cap10000.json"
d = json.load(open(IN))
rdf = pd.DataFrame(d["daily"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf["holdings"] = rdf["holdings"].apply(lambda x: [str(c) for c in x])
rdf["rand_holdings"] = rdf["rand_holdings"].apply(lambda x: [str(c) for c in x])

# Also compute theoretical returns for close-to-close
# fwd_1d_t1_open_ret = open_{t+2}/open_{t+1} - 1  (model label)
# We need close_{t+2}/close_{t+1} - 1 for close execution
# And close_{t+1}/close_t - 1 for same-day-close option

print("=== Computing theoretical close-to-close returns ===")
all_codes = set()
for h in rdf["holdings"]:
    all_codes.update(h)
klines = _load_klines_fast(DATA_DIR, all_codes)

# For each day, compute actual close-to-close return of holdings
close_t1_ret = []  # close_{t+2}/close_{t+1} - 1 (T+1 close buy, T+2 close sell)
close_t0_ret = []  # close_{t+1}/close_t - 1 (T close buy, T+1 close sell)
open_ret = []      # open_{t+2}/open_{t+1} - 1 (current label)

for _, row in rdf.iterrows():
    signal_date = pd.Timestamp(row["date"])
    holdings = row["holdings"]
    rets_t1 = []  # close_{t+2}/close_{t+1}
    rets_t0 = []  # close_{t+1}/close_t
    rets_open = [] # open_{t+2}/open_{t+1}
    for code in holdings:
        if code not in klines:
            continue
        kl = klines[code]
        pos = kl.attrs["date_to_pos"].get(signal_date)
        if pos is None or pos + 2 >= len(kl):
            continue
        c_t = float(kl.iloc[pos]["close"])
        c_t1 = float(kl.iloc[pos + 1]["close"])
        c_t2 = float(kl.iloc[pos + 2]["close"])
        o_t1 = float(kl.iloc[pos + 1]["open"])
        o_t2 = float(kl.iloc[pos + 2]["open"])
        rets_t1.append(c_t2 / c_t1 - 1)
        rets_t0.append(c_t1 / c_t - 1)
        rets_open.append(o_t2 / o_t1 - 1)
    if rets_t1:
        close_t1_ret.append(np.mean(rets_t1))
        close_t0_ret.append(np.mean(rets_t0))
        open_ret.append(np.mean(rets_open))
    else:
        close_t1_ret.append(np.nan)
        close_t0_ret.append(np.nan)
        open_ret.append(np.nan)

rdf["close_t1_ret"] = close_t1_ret
rdf["close_t0_ret"] = close_t0_ret
rdf["open_ret_actual"] = open_ret

# Theoretical cumulative (with 0.12% round-trip cost)
for label, col in [("open-to-open (T+1 buy, T+2 sell)", "open_ret_actual"),
                   ("close-to-close (T+1 buy, T+2 sell)", "close_t1_ret"),
                   ("close-to-close (T close buy, T+1 sell)", "close_t0_ret")]:
    r = rdf[col].dropna()
    cum = (1 + r - 0.0012).prod() - 1
    ann = (1 + r - 0.0012).prod() ** (252 / len(r)) - 1
    print(f"  {label}: cum={cum*100:.1f}% ann={ann*100:.1f}% mean_bp={(r.mean()-0.0012)*10000:.1f}")

# Run execution simulations
print("\n=== Execution simulation (fractional, no limit) ===")
for cap in [100000.0]:
    pf_open, exs_open = simulate_portfolio_open(rdf, DATA_DIR, cap, TRADE_COST, "holdings")
    fv_open = float(pf_open["portfolio_value"].iloc[-1])
    print(f"  OPEN  cap={cap}: final={round(fv_open,2)} return={round((fv_open/cap-1)*100,2)}% cost={exs_open['total_cost_paid']}")

    pf_close, exs_close = simulate_portfolio_close(rdf, DATA_DIR, cap, TRADE_COST, "holdings")
    fv_close = float(pf_close["portfolio_value"].iloc[-1])
    print(f"  CLOSE cap={cap}: final={round(fv_close,2)} return={round((fv_close/cap-1)*100,2)}% cost={exs_close['total_cost_paid']}")
