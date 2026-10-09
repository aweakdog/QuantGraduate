"""执行器: 影子不委托; 实盘 14:55 卖、没成交完转竞价、14:57 竞价买且不超钱; 对不上/重复/太晚都不下。"""
from datetime import datetime, timedelta
from types import SimpleNamespace

from qmt_bridge.executor import plan_buys, plan_sells, run
from qmt_bridge.readonly_probe import ReadOnlyTrader
from qmt_bridge.trading import TradingTrader

Q = {"600000": {"last": 10.0, "bid1": 9.99, "up": 11.0, "down": 9.0},
     "000001": {"last": 20.0, "bid1": 19.98, "up": 22.0, "down": 18.0},
     "600111": {"last": 11.0, "bid1": 10.99, "up": 11.0, "down": 9.0}}     # 涨停


def ticket(sell=(), buy=(), lots=(), cash=100000.0, cap=26000.0):
    return {"today": "2026-10-09", "profile": "qmt10w", "signal_date": "2026-10-08", "line_cash": cash,
            "slot_cap": cap, "trading_enabled": True, "lots": [{"code": c, "shares": s} for c, s in lots],
            "sell": [{"code": c, "shares": s} for c, s in sell], "buy": [{"code": c, "shares": s} for c, s in buy]}


def test_plan_sells_prices_and_skips():
    pos = {"000001": {"volume": 300, "can_use": 300}, "600000": {"volume": 100, "can_use": 0}}
    orders, skips = plan_sells(ticket(sell=[("000001", 300), ("600000", 100)]), pos, Q)
    assert orders == [{"code": "000001", "side": "sell", "volume": 300, "phase": "continuous", "price": 19.78,
                       "up": 22.0, "down": 18.0}]
    assert skips[0]["reason"].startswith("没有可卖数量")
    _, skips = plan_sells(ticket(sell=[("000001", 100)]), {"000001": {"volume": 100, "can_use": 100}},
                          {"000001": {**Q["000001"], "last": 18.0}})
    assert skips[0]["reason"] == "跌停卖不出"


def test_plan_buys_limit_price_cash_and_caps():
    orders, skips = plan_buys(ticket(buy=[("600000", 2000), ("600111", 100), ("000001", 1000)]), Q, 30000.0)
    assert orders[0]["volume"] == 2000 and orders[0]["price"] == 10.2       # 现价×1.02, 不超涨停
    assert [s["reason"] for s in skips][0] == "涨停买不进"
    # 剩余钱: 30000*0.995 - 2000*10.2*1.001 - 5 = 9429.6 -> 000001 限价 20.4, 只够 400 股
    assert orders[1]["code"] == "000001" and orders[1]["volume"] == 400
    assert any(s["reason"] == "减手 1000→400" for s in skips)
    orders, skips = plan_buys(ticket(buy=[("000001", 2000)], cap=10000.0), Q, 100000.0)
    assert orders[0]["volume"] == 400                                          # 单笔上限 10000 / 20.4
    orders, skips = plan_buys(ticket(buy=[("000001", 100)]), Q, 1000.0)
    assert orders == [] and skips[0]["reason"] == "现金不足一手"


class FakeClock:
    def __init__(self, start="14:53:30"):
        self.t = datetime.fromisoformat(f"2026-10-09T{start}")

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)

    def wait_until(self, hms):
        target = datetime.fromisoformat(f"2026-10-09T{hms}")
        if self.t < target:
            self.t = target


class FakeRaw:
    def __init__(self, positions=(), cash=100000.0, existing_remark=None, fill_sell=1.0):
        self.positions = [SimpleNamespace(stock_code=c, volume=v, can_use_volume=v) for c, v in positions]
        self.cash, self.calls, self.orders, self.fill_sell = cash, [], {}, fill_sell
        self.existing = [SimpleNamespace(order_remark=existing_remark)] if existing_remark else []

    def query_stock_positions(self, acc):
        return self.positions

    def query_stock_orders(self, acc, cancelable):
        return self.existing

    def query_stock_asset(self, acc):
        return SimpleNamespace(cash=self.cash)

    def order_stock(self, acc, code, side, vol, ptype, price, strat, remark):
        oid = 1000 + len(self.orders)
        filled = int(vol * self.fill_sell) // 100 * 100 if side == 24 and remark.endswith("-C") else 0
        self.orders[oid] = SimpleNamespace(order_status=56 if filled == vol else 50, traded_volume=filled,
                                           traded_price=price, code=code, side=side, vol=vol, price=price, remark=remark)
        if side == 24:
            self.cash += filled * price
        self.calls.append(("order", code, side, vol, price, remark))
        return oid

    def query_stock_order(self, acc, oid):
        return self.orders.get(oid)

    def cancel_order_stock(self, acc, oid):
        self.calls.append(("cancel", oid))
        self.orders[oid].order_status = 54
        return 0


def test_shadow_mode_never_orders_but_reports():
    raw = FakeRaw(positions=[("000001.SZ", 300)])
    t = ticket(sell=[("000001", 300)], buy=[("600000", 1000)], lots=[("000001", 300)], cash=20000.0)
    rep = run(t, False, ReadOnlyTrader(raw), "acc", lambda codes: Q, FakeClock(), "q10w-1009")
    assert rep["mode"] == "shadow" and rep["status"] == "done" and raw.calls == []
    assert [(o["side"], o["code"], o["volume"]) for o in rep["orders"]] == [("sell", "000001", 300), ("buy", "600000", 1000)]


def test_live_sells_then_auction_buys_within_cash():
    raw = FakeRaw(positions=[("000001.SZ", 300)], cash=4000.0)
    t = ticket(sell=[("000001", 300)], buy=[("600000", 1000)], lots=[("000001", 300)], cash=4000.0)
    clock = FakeClock()
    rep = run(t, True, TradingTrader(raw), "acc", lambda codes: Q, clock, "q10w-1009")
    assert rep["status"] == "done" and rep["errors"] == []
    sell, buy = raw.calls[0], raw.calls[1]
    assert sell[:4] == ("order", "000001.SZ", 24, 300) and sell[5] == "q10w-1009-C"
    assert buy[:3] == ("order", "600000.SH", 23) and buy[5] == "q10w-1009-A"
    # 买入可用钱 = min(QMT 现金 4000+5934, 线现金 4000 + 回款×0.999) -> 不超过两者
    assert rep["buy_cash"] <= 4000 + 300 * 19.78 and buy[3] * buy[4] <= rep["buy_cash"]
    assert clock.now().strftime("%H:%M:%S") >= "15:00:40"


def test_unfilled_sell_is_cancelled_and_moved_to_auction():
    raw = FakeRaw(positions=[("000001.SZ", 300)], fill_sell=0.34)   # 只成交 100 股
    t = ticket(sell=[("000001", 300)], lots=[("000001", 300)])
    rep = run(t, True, TradingTrader(raw), "acc", lambda codes: Q, FakeClock(), "q10w-1009")
    kinds = [c[0] for c in raw.calls]
    assert kinds == ["order", "cancel", "order"]
    assert raw.calls[2][3] == 200 and raw.calls[2][4] == 18.0 and raw.calls[2][5] == "q10w-1009-A"   # 剩余按跌停价进竞价
    assert rep["status"] == "done"


def test_guards_mismatch_duplicate_late():
    t = ticket(buy=[("600000", 100)], lots=[("000001", 300)])
    rep = run(t, True, TradingTrader(FakeRaw(positions=[])), "acc", lambda c: Q, FakeClock(), "q10w-1009")
    assert rep["status"] == "aborted" and "对账不一致" in rep["errors"][0]
    raw = FakeRaw(existing_remark="q10w-1009-A")
    rep = run(ticket(buy=[("600000", 100)]), True, TradingTrader(raw), "acc", lambda c: Q, FakeClock(), "q10w-1009")
    assert rep["status"] == "aborted" and "不重复下单" in rep["errors"][0] and raw.calls == []
    raw = FakeRaw()
    rep = run(ticket(buy=[("600000", 100)]), True, TradingTrader(raw), "acc", lambda c: Q, FakeClock("14:59:45"), "q10w-1009")
    assert rep["status"] == "aborted" and "太晚" in rep["errors"][0] and raw.calls == []
