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


def test_cash_equivalents_allowed_with_their_own_rules():
    # 货币 ETF: 按普通 ETF 规则 (买入整百)
    check_order("511880.SH", STOCK_BUY, 100, 100.903, 110.982, 90.804, 11000)
    with pytest.raises(OrderRejectedError):
        check_order("511880.SH", STOCK_BUY, 100, 100.903, 110.982, 90.804, 5000)      # 超单笔上限
    # 逆回购: 只能卖出, 10 张(1000 元)整数倍, 金额按面值算上限; 价格是年化利率
    check_order("204001.SH", STOCK_SELL, 10, 1.09, 100000.0, 0.0, 1000)
    for bad in (dict(side=STOCK_BUY), dict(volume=15), dict(max_value=500)):
        kw = dict(stock_code="204001.SH", side=STOCK_SELL, volume=10, price=1.09, up_stop=100000.0, down_stop=0.0, max_value=1000)
        with pytest.raises(OrderRejectedError):
            check_order(**{**kw, **bad})
    with pytest.raises(OrderRejectedError):
        check_order("510300.SH", STOCK_BUY, 100, 4.0, 4.4, 3.6, 5000)                 # 普通 ETF 仍不在白名单


def test_executor_positions_ignore_cash_equivalents():
    from types import SimpleNamespace

    from qmt_bridge.executor import positions_of
    raw = [SimpleNamespace(stock_code="511880.SH", volume=100, can_use_volume=100),
           SimpleNamespace(stock_code="204001.SH", volume=10, can_use_volume=0),
           SimpleNamespace(stock_code="600000.SH", volume=300, can_use_volume=300)]
    assert positions_of(raw) == {"600000": {"volume": 300, "can_use": 300}}


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
