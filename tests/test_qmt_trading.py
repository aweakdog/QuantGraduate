"""下单封装: 白名单 + 委托前硬检查 (主板/整百/限价在涨跌停内/金额上限), 不通过一律不下。"""
import pytest

from qmt_bridge.trading import (
    STOCK_BUY,
    STOCK_SELL,
    OrderRejectedError,
    TradingTrader,
    check_order,
    is_main_board,
    place_limit,
    round_tick,
    to_xt_code,
)


class Fake:
    def __init__(self, ret=123):
        self.calls, self.ret = [], ret

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a))
            return self.ret if name == "order_stock" else 0
        return f


def test_whitelist_allows_orders_but_not_transfers():
    t = TradingTrader(Fake())
    assert callable(t.order_stock) and callable(t.cancel_order_stock) and callable(t.query_stock_orders)
    for bad in ("fund_transfer", "smart_algo_order_async", "order_stock_async", "credit_order"):
        with pytest.raises(PermissionError):
            getattr(t, bad)


def test_board_and_code_helpers():
    assert is_main_board("600000.SH") and is_main_board("002595.SZ") and is_main_board("001979.SZ")
    for c in ("300750.SZ", "688981.SH", "830799.BJ", "600000", "689009.SH"):
        assert not is_main_board(c)
    assert to_xt_code("600000") == "600000.SH" and to_xt_code("000001") == "000001.SZ"
    assert round_tick(10.234) == 10.23 and round_tick(10.231, up=True) == 10.24 and round_tick(10.20, up=True) == 10.2


def test_check_order_rejects_everything_suspicious():
    ok = dict(stock_code="600000.SH", side=STOCK_BUY, volume=200, price=10.0, up_stop=11.0, down_stop=9.0, max_value=5000)
    check_order(**ok)
    for bad in (dict(stock_code="300750.SZ"), dict(volume=150), dict(volume=0), dict(price=11.5), dict(price=8.9),
                dict(up_stop=None), dict(max_value=1000), dict(side=99)):
        with pytest.raises(OrderRejectedError):
            check_order(**{**ok, **bad})
    check_order(**{**ok, "side": STOCK_SELL, "volume": 150})          # 卖出零股允许(部分成交剩余)


def test_place_limit_requires_trading_wrapper_and_checks_first():
    raw = Fake()
    with pytest.raises(PermissionError):
        place_limit(raw, "acc", "600000.SH", STOCK_BUY, 100, 10.0, 11.0, 9.0, 5000, "r")
    t = TradingTrader(raw)
    with pytest.raises(OrderRejectedError):
        place_limit(t, "acc", "688981.SH", STOCK_BUY, 100, 10.0, 11.0, 9.0, 5000, "r")
    assert raw.calls == []                                   # 检查不过就一笔都没发
    assert place_limit(t, "acc", "600000.SH", STOCK_BUY, 100, 10.0, 11.0, 9.0, 5000, "r") == 123
    name, a = raw.calls[0]
    assert name == "order_stock" and a[:6] == ("acc", "600000.SH", STOCK_BUY, 100, 11, 10.0)
    with pytest.raises(OrderRejectedError):
        place_limit(TradingTrader(Fake(ret=-1)), "acc", "600000.SH", STOCK_BUY, 100, 10.0, 11.0, 9.0, 5000, "r")
