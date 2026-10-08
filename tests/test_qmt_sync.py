"""QMT 快照/对账/自动确认的纯逻辑: 宁可停下等人工, 不可错记。"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qmt_sync import (  # noqa: E402
    aggregate_trades,
    build_autoconfirm,
    final_snapshot_for,
    reconcile,
    sign,
    store_snapshot,
    validate_snapshot,
    verify,
)

KEY = b"k" * 64
D = "2026-10-09"


def _ts(hms):
    return int(datetime.fromisoformat(f"{D}T{hms}").timestamp())


def snap(positions=(), trades=(), cash=100000.0, at="15:10:00", online=True, kind="close"):
    return {"probe_at": f"{D}T{at}", "kind": kind, "account_online": online,
            "asset": {"cash": cash, "total_asset": cash}, "orders": [],
            "positions": [{"stock_code": c, "volume": v} for c, v in positions],
            "trades": [{"stock_code": c, "order_type": ot, "traded_volume": v, "traded_price": p,
                        "traded_amount": v * p, "traded_time": _ts("14:58:00")} for c, ot, v, p in trades]}


def state(lots=(), cash=100000.0):
    return {"cash": cash, "lots": [{"code": c, "shares": s, "buy_price": 10.0} for c, s in lots]}


def test_signature_window_and_tamper():
    body = b'{"a": 1}'
    now = 1_800_000_000
    good = sign(body, str(now), KEY)
    assert verify(body, str(now), good, KEY, now=now) == (True, "")
    assert not verify(body, str(now), good, KEY, now=now + 301)[0]          # 过期
    assert not verify(b'{"a": 2}', str(now), good, KEY, now=now)[0]        # 篡改
    assert not verify(body, str(now), sign(body, str(now), b"x" * 64), KEY, now=now)[0]  # 错钥匙
    assert not verify(body, "abc", good, KEY, now=now)[0]
    assert verify(body, str(now), good, b"", now=now) == (False, "服务器未配置 QMT 桥密钥")


def test_validate_store_and_pick_final_close_snapshot(tmp_path):
    assert validate_snapshot(snap()) == []
    assert validate_snapshot({"probe_at": "x", "kind": "noon"})
    assert validate_snapshot({"probe_at": f"{D}T09:35:00", "kind": "morning", "account_online": False}) == []
    store_snapshot(snap(at="09:35:00", kind="morning"), tmp_path)
    store_snapshot(snap(at="15:10:00", cash=1.0), tmp_path)
    store_snapshot(snap(at="15:40:00", cash=2.0), tmp_path)
    store_snapshot(snap(at="16:00:00", cash=3.0, online=False), tmp_path)   # 离线的不算
    assert final_snapshot_for(tmp_path, D)["asset"]["cash"] == 2.0
    assert json.loads((tmp_path / "qmt" / "latest.json").read_text())["asset"]["cash"] == 3.0
    assert final_snapshot_for(tmp_path, "2026-10-10") is None


def test_aggregate_trades_vwap_and_rejects():
    fills, probs = aggregate_trades([
        {"stock_code": "600000.SH", "order_type": 23, "traded_volume": 100, "traded_price": 10.0, "traded_amount": 1000.0},
        {"stock_code": "600000.SH", "order_type": 23, "traded_volume": 300, "traded_price": 10.4, "traded_amount": 3120.0},
        {"stock_code": "000001.SZ", "order_type": 24, "traded_volume": 200, "traded_price": 12.0},
        {"stock_code": "000002.SZ", "order_type": 99, "traded_volume": 100, "traded_price": 5.0},
    ])
    assert fills == [{"code": "000001", "action": "sell", "shares": 200, "price": 12.0},
                     {"code": "600000", "action": "buy", "shares": 400, "price": 10.3}]
    assert len(probs) == 1 and "order_type=99" in probs[0]
    _, probs = aggregate_trades([{"stock_code": "600000.SH", "order_type": 23, "traded_volume": 100,
                                  "traded_price": 10.0, "traded_time": _ts("10:00:00") - 86400}], D)
    assert probs and "不是执行日" in probs[0]


def test_no_trades_and_flat_account_confirms_empty():
    fills, probs, notes = build_autoconfirm(snap(), state(), {"buy": []}, D)
    assert (fills, probs, notes) == ([], [], [])


def test_planned_buys_and_held_sells_confirm():
    s = snap(positions=[("600000.SH", 400), ("601111.SH", 100)],
             trades=[("000001.SZ", 24, 200, 12.0), ("600000.SH", 23, 400, 10.3)])
    st = state(lots=[("000001", 200), ("601111", 100)])
    plan = {"buy": [{"code": "600000"}], "alternates": []}
    fills, probs, notes = build_autoconfirm(s, st, plan, D)
    assert probs == [] and [f["code"] for f in fills] == ["000001", "600000"]


def test_alternate_buy_is_allowed_with_note():
    s = snap(positions=[("600111.SH", 100)], trades=[("600111.SH", 23, 100, 20.0)])
    fills, probs, notes = build_autoconfirm(s, state(), {"buy": [{"code": "600000"}], "alternates": [{"code": "600111"}]}, D)
    assert probs == [] and notes == ["买入 600111 来自候补名单"]


def test_anything_unexplained_blocks_autoconfirm():
    plan = {"buy": [{"code": "600000"}]}
    # 计划外买入
    _, probs, _ = build_autoconfirm(snap(positions=[("688001.SH", 200)], trades=[("688001.SH", 23, 200, 30.0)]), state(), plan, D)
    assert any("计划外" in p for p in probs)
    # 卖出线外代码
    _, probs, _ = build_autoconfirm(snap(trades=[("000002.SZ", 24, 100, 9.0)]), state(), plan, D)
    assert any("线外代码" in p for p in probs)
    # 卖超
    _, probs, _ = build_autoconfirm(snap(trades=[("000001.SZ", 24, 300, 9.0)]), state(lots=[("000001", 200)]), plan, D)
    assert any("只有 200 股" in p for p in probs)
    # 无成交但 QMT 多出线外持仓
    _, probs, _ = build_autoconfirm(snap(positions=[("600519.SH", 100)]), state(), plan, D)
    assert any("不一致" in p for p in probs)
    # 线账本有持仓但 QMT 没有 (漏报卖出)
    _, probs, _ = build_autoconfirm(snap(), state(lots=[("000001", 200)]), plan, D)
    assert any("不一致" in p for p in probs)
    # 盘中快照 / 别的日子的快照 / 账户离线 / 没有快照
    assert build_autoconfirm(snap(at="14:30:00"), state(), plan, D)[1]
    assert build_autoconfirm(snap(), state(), plan, "2026-10-10")[1]
    assert build_autoconfirm(snap(online=False), state(), plan, D)[1]
    assert build_autoconfirm(None, state(), plan, D)[1] == [f"没有 {D} 收盘后的 QMT 快照"]


def test_reconcile_reports_position_diffs_and_outside_cash():
    r = reconcile(snap(positions=[("600000.SH", 400)], cash=503121.8), state(lots=[("600000", 300)], cash=100000))
    assert r["positions_match"] is False and r["diffs"] == [{"code": "600000", "qmt": 400, "line": 300}]
    assert r["outside_cash"] == 403121.8
    assert reconcile(snap(), state())["positions_match"] is True


def test_sign_is_stable_across_processes():
    # Windows 侧用同一算法: HMAC-SHA256(key, ts + "\n" + body)
    import hashlib
    import hmac
    ts = str(int(time.time()))
    assert sign(b"x", ts, KEY) == hmac.new(KEY, ts.encode() + b"\nx", hashlib.sha256).hexdigest()
