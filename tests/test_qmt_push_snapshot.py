"""Windows 推送脚本: 签名与 041 一致; 连不上/没账户也要推 offline 快照; 只读。"""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from qmt_bridge import push_snapshot as ps


class _FakeRaw:
    def __init__(self, rc=0, accounts=()):
        self.rc, self.accounts, self.calls = rc, list(accounts), []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append(name)
            if name == "connect":
                return self.rc
            if name == "query_account_infos":
                return self.accounts
            if name == "query_stock_asset":
                return SimpleNamespace(account_type=2, account_id="1234566735", cash=100000.0,
                                       frozen_cash=0.0, market_value=0.0, total_asset=100000.0)
            if name in ("query_stock_positions", "query_stock_orders", "query_stock_trades"):
                return []
            return 0
        return f


@pytest.fixture
def fake_xt(monkeypatch):
    holder = {}

    def install(raw):
        xt = types.ModuleType("xtquant")
        trader_mod = types.ModuleType("xtquant.xttrader")
        type_mod = types.ModuleType("xtquant.xttype")
        trader_mod.XtQuantTrader = lambda path, sid: raw
        type_mod.StockAccount = lambda aid, kind: SimpleNamespace(account_id=aid, kind=kind)
        monkeypatch.setitem(sys.modules, "xtquant", xt)
        monkeypatch.setitem(sys.modules, "xtquant.xttrader", trader_mod)
        monkeypatch.setitem(sys.modules, "xtquant.xttype", type_mod)
        holder["raw"] = raw
    return install


def test_online_snapshot_masks_account_and_only_reads(fake_xt):
    raw = _FakeRaw(accounts=[SimpleNamespace(account_type=2, account_id="1234566735")])
    fake_xt(raw)
    snap, _ = ps.take_snapshot("close", timeout=2)
    assert snap["account_online"] is True and snap["kind"] == "close"
    assert snap["account"] == "******6735" and snap["asset"]["account_id"] == "******6735"
    assert all(c in ("start", "connect", "subscribe") or c.startswith("query_") for c in raw.calls)


def test_offline_paths_still_produce_snapshot(fake_xt):
    fake_xt(_FakeRaw(rc=-1))
    snap, _ = ps.take_snapshot("morning", timeout=2)
    assert snap["account_online"] is False and "rc=-1" in snap["error"]
    fake_xt(_FakeRaw(rc=0, accounts=[]))
    snap, _ = ps.take_snapshot("morning", timeout=2)
    assert snap["account_online"] is False and "未挂上" in snap["error"]


def test_push_signs_like_the_server():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from qmt_sync import verify
    seen = {}

    class Resp:
        status = 200

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        seen["req"] = req
        return Resp()

    status, _ = ps.push({"probe_at": "2026-10-09T15:10:00", "kind": "close"}, "http://x/api/qmt/snapshot", b"k" * 32, opener=opener)
    req = seen["req"]
    assert status == 200 and req.get_method() == "POST"
    hdr = {k.lower(): v for k, v in req.header_items()}
    assert verify(req.data, hdr["x-qmt-ts"], hdr["x-qmt-sig"], b"k" * 32)[0]
    assert json.loads(req.data)["kind"] == "close"
