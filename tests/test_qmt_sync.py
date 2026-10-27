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


def _plan(sig="2026-10-08", buys=(), sells=(), in_cash=False):
    return {"signal_date": sig, "in_cash": in_cash, "equity": 100000.0, "generated_at": "2026-10-08T18:51:00",
            "buy": [{"code": c, "shares": s, "ref_close": 10.0, "budget": 20000} for c, s in buys],
            "sell": [{"code": c, "shares": s, "ref_close": 10.0} for c, s in sells]}


def test_ticket_valid_only_for_today_and_confirmed_ledger(tmp_path):
    from qmt_sync import build_ticket, load_switch, save_switch
    nxt = {"2026-10-08": "2026-10-09"}.get
    st = {"cash": 80000.0, "pending": {"signal_date": "2026-10-08"}, "awaiting_confirm": None,
          "lots": [{"code": "000001", "shares": 200, "buy_price": 10.0}]}
    plan = _plan(buys=[("600000", 300)], sells=[("000001", 200)])
    t, probs = build_ticket("qmt10w", st, plan, "2026-10-09", nxt, {"enabled": False}, 5)
    assert probs == [] and t["exec_date"] == "2026-10-09" and t["trading_enabled"] is False
    assert t["sell"] == [{"code": "000001", "shares": 200, "ref_close": 10.0}] and t["buy"][0]["code"] == "600000"
    assert t["slot_cap"] == 26000.0 and t["lots"] == [{"code": "000001", "shares": 200}]
    # 不是今天 / 待确认 / 信号日对不上 / 卖超 / 空仓却买 / 没计划
    assert build_ticket("qmt10w", st, plan, "2026-10-10", nxt, {}, 5)[1]
    assert build_ticket("qmt10w", {**st, "awaiting_confirm": {"exec_date": "2026-10-08"}}, plan, "2026-10-09", nxt, {}, 5)[1]
    assert build_ticket("qmt10w", {**st, "pending": {"signal_date": "2026-09-30"}}, plan, "2026-10-09", nxt, {}, 5)[1]
    assert build_ticket("qmt10w", st, _plan(sells=[("000001", 300)]), "2026-10-09", nxt, {}, 5)[1]
    assert build_ticket("qmt10w", st, _plan(buys=[("600000", 100)], in_cash=True), "2026-10-09", nxt, {}, 5)[1]
    assert build_ticket("qmt10w", st, None, "2026-10-09", nxt, {}, 5)[1]
    # 开关: 默认关, 存取往返, 坏文件当关
    assert load_switch(tmp_path) == {"enabled": False}
    save_switch(tmp_path, True, "llx")
    assert load_switch(tmp_path)["enabled"] is True and load_switch(tmp_path)["by"] == "llx"
    (tmp_path / "qmt" / "trading_switch.json").write_text("{bad")
    assert load_switch(tmp_path) == {"enabled": False}
    assert len((tmp_path / "qmt" / "switch_log.jsonl").read_text().splitlines()) == 1


def test_manual_confirm_gate_only_after_autoconfirm_had_its_chance(tmp_path):
    from qmt_sync import manual_confirm_gate
    aw = {"awaiting_confirm": {"signal_date": "2026-10-08", "exec_date": "2026-10-09"}}
    at = lambda hms, day="2026-10-09": datetime.fromisoformat(f"{day}T{hms}")  # noqa: E731
    # 普通线 / 没在待确认: 不管
    assert manual_confirm_gate(tmp_path, "steady5w", aw, at("18:34:00")) == (True, "")
    assert manual_confirm_gate(tmp_path, "qmt10w", {"awaiting_confirm": None}, at("18:34:00")) == (True, "")
    # 执行日当晚 21:00 之前: 只认自动确认 (10-09 18:34 那次手点就该被拦)
    ok, note = manual_confirm_gate(tmp_path, "qmt10w", aw, at("18:34:00"))
    assert ok is False and "18:50" in note
    assert manual_confirm_gate(tmp_path, "qmt10w", aw, at("20:59:59"))[0] is False
    # 21:00 之后 / 执行日已过: 放人工
    assert manual_confirm_gate(tmp_path, "qmt10w", aw, at("21:00:00"))[0] is True
    assert manual_confirm_gate(tmp_path, "qmt10w", aw, at("09:30:00", "2026-10-12"))[0] is True
    # 自动确认被拦下/失败: 立刻放人工, 并带上原因; 别的日子/别的线的记录不算
    log = tmp_path / "qmt" / "autoconfirm.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"pid": "qmt10w", "exec_date": "2026-10-08", "status": "blocked", "problems": ["旧"]}) + "\n"
                   + json.dumps({"pid": "qmt10w", "exec_date": "2026-10-09", "status": "confirmed"}) + "\n")
    assert manual_confirm_gate(tmp_path, "qmt10w", aw, at("19:00:00"))[0] is False
    with log.open("a") as fh:
        fh.write("garbage\n" + json.dumps({"pid": "qmt10w", "exec_date": "2026-10-09", "status": "blocked",
                                           "problems": ["买入了计划外代码 688001"]}) + "\n")
    ok, note = manual_confirm_gate(tmp_path, "qmt10w", aw, at("19:00:00"))
    assert ok is True and "计划外代码 688001" in note


class _FakeSMTP:
    sent, tls_capable = [], True

    def __init__(self, host, port, timeout):
        self.addr, self.tls = (host, port), False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self):
        return 250, b"ok"

    def has_extn(self, name):
        return name == "starttls" and _FakeSMTP.tls_capable

    def starttls(self, context=None):
        self.tls = True

    def send_message(self, msg):
        _FakeSMTP.sent.append((self.addr, msg, self.tls))
        return {}


def test_email_channel_requires_tls_and_sets_reply_to():
    from notify_channels import EmailChannel
    _FakeSMTP.sent.clear()
    _FakeSMTP.tls_capable = True
    ok = EmailChannel("yliog@connect.ust.hk", smtp=_FakeSMTP).send("【QMT】10-14 开盘检查: 资金账号离线\n第二行 **加粗**")
    assert ok is True
    addr, msg, tls = _FakeSMTP.sent[0]
    assert tls is True and addr == ("smtp.ece.ust.hk", 25) and msg["To"] == "yliog@connect.ust.hk"
    assert msg["Reply-To"] == "yliog@connect.ust.hk" and "EEZ041@ece.ust.hk" in msg["From"]
    assert msg["Subject"] == "【QMT】10-14 开盘检查: 资金账号离线" and msg["Message-ID"] and "**" not in msg.get_content()
    _FakeSMTP.tls_capable = False                                   # 中继不支持加密 -> 不发明文
    assert EmailChannel("yliog@connect.ust.hk", smtp=_FakeSMTP).send("x") is False and len(_FakeSMTP.sent) == 1
    _FakeSMTP.tls_capable = True


def test_email_summary_never_leaks_codes_amounts_or_line_name():
    from qmt_sync import email_summary
    detailed = "【QMT】华泰QMT 10万 2026-10-14 收盘对账: 持仓与券商不一致 [{'code': '600000', 'qmt': 400, 'line': 300}] ¥103,121.80"
    for key in ("2026-10-14:morning:offline", "2026-10-14:close:mismatch:qmt10w", "2026-10-14:autoconfirm_ok:qmt10w",
                "2026-10-14:autoconfirm_blocked:qmt10w", "2026-10-14:autoconfirm_failed:qmt10w",
                "2026-10-14:exec:live", "2026-10-14:exec:shadow", "weird-key"):
        s = email_summary(key, detailed)
        assert s.startswith("【QMT】") and "eez041.ece.ust.hk:8737" in s
        for leak in ("600000", "400", "300", "103", "10万", "华泰"):
            assert leak not in s.split("\n")[0], (key, s)
    assert "开盘检查" in email_summary("2026-10-14:morning:offline") and "10-14" in email_summary("2026-10-14:morning:offline")
    assert "自动下单" in email_summary("2026-10-14:exec:live")


def test_alert_main_gets_details_email_gets_summary_each_recorded(tmp_path, monkeypatch):
    import qmt_sync
    from qmt_sync import alert_once
    got = []
    monkeypatch.setattr(qmt_sync, "_default_senders",
                        lambda: [("main", lambda t: got.append(("main", t)) or True),
                                 ("email", lambda t: got.append(("email", t)) or True)])
    assert alert_once(tmp_path, "2026-10-14:close:mismatch:qmt10w", "不一致 600000 400 vs 300") is True
    assert got[0] == ("main", "不一致 600000 400 vs 300") and "600000" not in got[1][1] and got[1][0] == "email"
    monkeypatch.setattr(qmt_sync, "_default_senders",
                        lambda: [("main", lambda t: got.append(("main", t)) or True),
                                 ("email", lambda t: (_ for _ in ()).throw(OSError("relay down")))])
    assert alert_once(tmp_path, "2026-10-14:morning:offline", "离线") is True     # 邮件挂了不影响主通道
    rec = json.loads((tmp_path / "qmt" / "alerts_sent.json").read_text())["2026-10-14:morning:offline"]
    assert rec["ok"] is True and rec["channels"] == {"main": True, "email": False}
    assert alert_once(tmp_path, "2026-10-14:morning:offline", "离线") is False    # 同 key 不重发


def test_cash_equivalents_are_cash_not_strategy_positions():
    from qmt_sync import aggregate_trades, build_autoconfirm, is_cash_equiv, reconcile
    assert is_cash_equiv("511880.SH") and is_cash_equiv("204001.SH") and is_cash_equiv("131810.SZ")
    assert not is_cash_equiv("600000.SH") and not is_cash_equiv("510300.SH")
    s = snap(positions=[("511880.SH", 100)], cash=93000.0,
             trades=[("511880.SH", 23, 100, 100.903), ("511880.SH", 24, 100, 100.902), ("204001.SH", 24, 10, 1.09)])
    s["positions"][0]["market_value"] = 10090.3
    r = reconcile(s, state())
    assert r["positions_match"] is True and r["cash_equiv_value"] == 10090.3
    assert r["outside_cash"] == round(93000.0 + 10090.3 - 100000.0, 2)
    assert aggregate_trades(s["trades"]) == ([], [])
    fills, probs, notes = build_autoconfirm(s, state(), {"buy": []}, D)
    assert (fills, probs) == ([], [])                      # 当晚自动确认不会因现金管理被拦


def test_sign_is_stable_across_processes():
    # Windows 侧用同一算法: HMAC-SHA256(key, ts + "\n" + body)
    import hashlib
    import hmac
    ts = str(int(time.time()))
    assert sign(b"x", ts, KEY) == hmac.new(KEY, ts.encode() + b"\nx", hashlib.sha256).hexdigest()


def test_email_summary_explains_screen_failures_without_trade_details():
    from qmt_sync import email_summary
    s = email_summary("2026-10-26:close:offline:layout")
    assert "收盘检查" in s and "界面可能变了" in s and "极简模式" not in s
    assert "DeepSeek" in email_summary("2026-10-26:morning:offline:exception")
    assert "人工登录" in email_summary("2026-10-26:morning:offline:login")
    assert "资金账号离线" in email_summary("2026-10-26:morning:offline")          # 旧格式 (无类别) 仍可用


def test_sanity_checks_fill_price_and_total_asset(tmp_path):
    import json as _j

    import pandas as pd
    from qmt_sync import sanity_checks
    kd, ld = tmp_path / "kline", tmp_path / "live"
    kd.mkdir()
    (ld / "qmt").mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-10-27"]), "low": [9.45], "high": [9.68]}).to_parquet(kd / "600000.parquet")
    snap = {"asset": {"total_asset": 503121.79}}
    ok = [{"code": "600000", "action": "buy", "shares": 1000, "price": 9.62}]
    assert sanity_checks(snap, ok, "2026-10-27", kd, ld) == ([], [])
    p, _ = sanity_checks(snap, [{**ok[0], "price": 6.92}], "2026-10-27", kd, ld)           # 读错的价格
    assert p and "不在当天行情" in p[0]
    p, _ = sanity_checks(snap, [{**ok[0], "shares": 1050}], "2026-10-27", kd, ld)
    assert p and "整百" in p[0]
    _, n = sanity_checks(snap, [{**ok[0], "code": "000001"}], "2026-10-27", kd, ld)
    assert n and "未核对" in n[0]
    (ld / "qmt" / "snap_20261027_093637_morning.json").write_text(_j.dumps(
        {"account_online": True, "asset": {"total_asset": 503121.80, "market_value": 0}}))
    assert sanity_checks(snap, [], "2026-10-27", kd, ld)[0] == []
    p, _ = sanity_checks({"asset": {"total_asset": 50312.18}}, [], "2026-10-27", kd, ld)   # 少读一位
    assert p and "超出合理范围" in p[0]
