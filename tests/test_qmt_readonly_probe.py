"""QMT 只读探针: 不需要 xtquant, 用假交易对象验证只读保证与快照格式。"""
from types import SimpleNamespace

import pytest

from qmt_bridge.readonly_probe import ReadOnlyTrader, collect, mask, pick_stock_account


class FakeTrader:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append(name)
            if name == "query_stock_asset":
                return SimpleNamespace(account_type=2, account_id="1234567890", cash=1000.5,
                                       frozen_cash=0, market_value=2000, total_asset=3000.5)
            if name == "query_stock_positions":
                return [SimpleNamespace(stock_code="600000.SH", volume=200, can_use_volume=200,
                                        open_price=10.0, market_value=2000, account_id="1234567890")]
            if name in ("query_stock_orders", "query_stock_trades"):
                return []
            return 0
        return f


def test_read_only_proxy_blocks_everything_but_queries_and_connection():
    raw = FakeTrader()
    t = ReadOnlyTrader(raw)
    t.start()
    t.connect()
    t.query_stock_asset("acc")
    for name in ("order_stock", "order_stock_async", "cancel_order_stock", "cancel_order_stock_sysid",
                 "fund_transfer", "smart_algo_order_async", "_trader_raw"):
        with pytest.raises(PermissionError):
            getattr(t, name)
    with pytest.raises(PermissionError):
        t.order_stock = lambda *a: None
    assert raw.calls == ["start", "connect", "query_stock_asset"]


def test_collect_masks_account_and_only_queries():
    raw = FakeTrader()
    snap = collect(ReadOnlyTrader(raw), "acc")
    assert snap["asset"]["account_id"] == "******7890" and snap["asset"]["total_asset"] == 3000.5
    assert snap["positions"][0]["stock_code"] == "600000.SH" and snap["positions"][0]["account_id"] == "******7890"
    assert snap["orders"] == [] and snap["trades"] == []
    assert all(c == "subscribe" or c.startswith("query_") for c in raw.calls)
    assert collect(ReadOnlyTrader(FakeTrader()), "acc", masked=False)["asset"]["account_id"] == "1234567890"


def test_pick_stock_account_and_mask():
    infos = [SimpleNamespace(account_type=3, account_id="999"), SimpleNamespace(account_type=2, account_id="12345678")]
    assert pick_stock_account(infos).account_id == "12345678"
    assert pick_stock_account(infos, "12345678").account_id == "12345678"
    with pytest.raises(SystemExit):
        pick_stock_account(infos, "000")
    with pytest.raises(SystemExit):
        pick_stock_account([SimpleNamespace(account_type=3, account_id="1")])
    assert mask("12345678") == "****5678" and mask("") == "" and mask(None) == ""
