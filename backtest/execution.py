from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


BOARD_LIMITS = {
    "300": 0.20,
    "301": 0.20,
    "688": 0.20,
    "689": 0.20,
    "8": 0.30,
    "4": 0.30,
}


def _limit_pct(code: str, row_index: int) -> float | None:
    if row_index < 5:
        return None
    code6 = str(code)[:6]
    for prefix, limit in BOARD_LIMITS.items():
        if code6.startswith(prefix):
            return limit
    return 0.10


def _at_limit_up(code: str, row_index: int, previous_close: float, price: float) -> bool:
    limit = _limit_pct(code, row_index)
    return limit is not None and price >= previous_close * (1 + limit - 0.001)


def _at_limit_down(code: str, row_index: int, previous_close: float, price: float) -> bool:
    limit = _limit_pct(code, row_index)
    return limit is not None and price <= previous_close * (1 - limit + 0.001)


def _load_klines(data_dir: Path, codes: set[str]):
    cache = {}
    for code in codes:
        code6 = str(code)[:6]
        path = data_dir / "raw" / "kline" / f"{code6}.parquet"
        if not path.exists():
            continue
        kline = pd.read_parquet(path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        cache[code] = kline
    return cache


def _next_bar(kline: pd.DataFrame, signal_date: pd.Timestamp):
    matches = kline.index[kline["date"] == signal_date]
    if len(matches) == 0 or matches[0] + 1 >= len(kline):
        return None
    index = int(matches[0] + 1)
    current = kline.iloc[index]
    previous = kline.iloc[index - 1]
    return {
        "date": current["date"],
        "index": index,
        "previous_close": float(previous["close"]),
        "open": float(current["open"]),
        "close": float(current["close"]),
    }


def simulate_portfolio(
    daily: pd.DataFrame,
    data_dir: Path,
    initial_capital: float,
    trade_cost: float,
    holdings_column: str,
):
    codes = {
        str(code)
        for values in daily[holdings_column]
        for code in values
    }
    klines = _load_klines(data_dir, codes)
    cash = float(initial_capital)
    positions = {}
    values = []
    rejected_buy = 0
    rejected_sell = 0
    missing_bars = 0
    buy_cost_paid = 0.0
    sell_cost_paid = 0.0
    trade_log = []
    previous_value = cash

    for _, signal in daily.sort_values("date").iterrows():
        signal_date = pd.Timestamp(signal["date"])
        bars = {}
        for code in set(positions) | set(map(str, signal[holdings_column])):
            if code not in klines:
                missing_bars += 1
                continue
            bar = _next_bar(klines[code], signal_date)
            if bar is not None:
                bars[code] = bar

        sold_today = 0
        buy_cost_today = 0.0
        sell_cost_today = 0.0
        for code, shares in list(positions.items()):
            bar = bars.get(code)
            if bar is None:
                continue
            if _at_limit_down(code, bar["index"], bar["previous_close"], bar["open"]):
                rejected_sell += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL_REJECTED", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "reason": "limit_down_open"})
                continue
            gross = shares * bar["open"]
            sell_fee = max(gross * trade_cost, 5.0)
            sell_cost_paid += sell_fee
            sell_cost_today += sell_fee
            cash += gross - sell_fee
            trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "gross_amount": round(gross, 2), "reason": "t1_exit"})
            sold_today += 1
            del positions[code]

        targets = [str(code) for code in signal[holdings_column]]
        eligible = []
        for code in targets:
            bar = bars.get(code)
            if bar is None:
                rejected_buy += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": None, "action": "BUY_REJECTED", "timing": "open", "code": code, "shares": 0, "reason": "missing_bar"})
                continue
            if _at_limit_up(code, bar["index"], bar["previous_close"], bar["open"]):
                rejected_buy += 1
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY_REJECTED", "timing": "open", "code": code, "shares": 0, "price": bar["open"], "reason": "limit_up_open"})
                continue
            eligible.append((code, bar))

        bought_today = 0
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
                trade_log.append({"signal_date": str(signal_date.date()), "trade_date": str(bar["date"].date()), "action": "BUY", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "gross_amount": round(gross, 2), "reason": "integer_lot"})

        marked_value = cash
        for code, shares in positions.items():
            bar = bars.get(code)
            if bar is not None:
                marked_value += shares * bar["open"]
        daily_return = marked_value / previous_value - 1 if previous_value else 0.0
        values.append({
            "date": str(signal_date.date()),
            "portfolio_value": round(marked_value, 2),
            "daily_ret": daily_return,
            "turnover_bought": bought_today,
            "turnover_sold": sold_today,
            "buy_cost": round(buy_cost_today, 2),
            "sell_cost": round(sell_cost_today, 2),
            "total_cost": round(buy_cost_today + sell_cost_today, 2),
        })
        previous_value = marked_value

    if positions and values:
        last_signal_date = pd.Timestamp(daily.sort_values("date").iloc[-1]["date"])
        last_entry_date = pd.Timestamp(values[-1]["date"])
        final_bars = {}
        for code in positions:
            entry_bar = _next_bar(klines[code], last_entry_date)
            if entry_bar is None:
                continue
            exit_bar = _next_bar(klines[code], entry_bar["date"])
            if exit_bar is not None:
                final_bars[code] = exit_bar
        final_value = cash
        final_sold = 0
        final_sell_cost = 0.0
        for code, shares in list(positions.items()):
            bar = final_bars.get(code)
            if bar is None:
                continue
            if _at_limit_down(code, bar["index"], bar["previous_close"], bar["open"]):
                rejected_sell += 1
                trade_log.append({"signal_date": str(last_signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL_REJECTED", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "reason": "limit_down_open"})
                final_value += shares * bar["open"]
                continue
            gross = shares * bar["open"]
            sell_fee = max(gross * trade_cost, 5.0)
            sell_cost_paid += sell_fee
            final_sell_cost += sell_fee
            cash += gross - sell_fee
            final_sold += 1
            trade_log.append({"signal_date": str(last_signal_date.date()), "trade_date": str(bar["date"].date()), "action": "SELL", "timing": "open", "code": code, "shares": shares, "price": bar["open"], "gross_amount": round(gross, 2), "reason": "final_t1_exit"})
            del positions[code]
        final_value = cash
        previous_value = values[-1]["portfolio_value"]
        values[-1]["portfolio_value"] = round(final_value, 2)
        values[-1]["daily_ret"] = final_value / previous_value - 1 if previous_value else 0.0
        values[-1]["turnover_sold"] += final_sold
        values[-1]["sell_cost"] += round(final_sell_cost, 2)
        values[-1]["total_cost"] += round(final_sell_cost, 2)

    result = pd.DataFrame(values)
    result["rolling_100d_ret_pct"] = (
        result["portfolio_value"].pct_change(100).fillna(0) * 100
    )
    stats = {
        "rejected_buy": rejected_buy,
        "rejected_sell": rejected_sell,
        "missing_bars": missing_bars,
        "buy_cost_paid": round(buy_cost_paid, 2),
        "sell_cost_paid": round(sell_cost_paid, 2),
        "total_cost_paid": round(buy_cost_paid + sell_cost_paid, 2),
        "trade_log": trade_log,
    }
    return result, stats
