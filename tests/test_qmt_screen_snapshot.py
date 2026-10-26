"""读屏快照: 多次读取一致性、算术交叉核对、输出格式与 041 现有对账/自动确认兼容。"""
import sys
from pathlib import Path

import qmt_bridge.screen_snapshot as ss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qmt_sync import aggregate_trades, qmt_positions, validate_snapshot  # noqa: E402

FUNDS = {"total_asset": 503121.80, "available": 493585.80, "frozen": 0.0, "market_value": 9536.0}
POS = [{"code": "600000", "name": "浦发银行", "volume": 1000, "can_use": 0, "cost": 9.52, "market_value": 9536.0}]
TRADES = [{"time": "14:57:10", "code": "600000", "side": "买入", "price": 9.52, "volume": 1000, "amount": 9520.0,
           "order_id": "15731", "trade_id": "T1"}]
ORDERS = [{"time": "14:57:09", "code": "600000", "side": "买入", "price": 9.70, "volume": 1000, "traded_volume": 1000,
           "status": "已成", "order_id": "15731"}]


def test_agree_needs_two_matching_reads():
    canon = ss.canon_funds
    a = {"total_asset": "503,121.80", "available": 503121.8, "frozen": 0, "market_value": 0}
    b = {"total_asset": 503121.8, "available": "503121.80", "frozen": "0.00", "market_value": 0.0}
    assert ss.agree([a, b], canon)[0] == {"total_asset": 503121.8, "available": 503121.8, "frozen": 0.0, "market_value": 0.0}
    c = {**b, "available": 502258.74}
    assert ss.agree([a, c], canon)[0] is None                       # 两次不一致 -> 不采用
    assert ss.agree([a, c, b], canon)[0]["available"] == 503121.8    # 三次取多数


def test_canon_rows_ignores_order_and_formatting():
    r1 = [{"code": "600000.SH", "volume": "1,000股", "can_use": "0", "cost": "9.520", "market_value": "9536"}]
    r2 = [{"code": "600000", "volume": 1000, "can_use": 0, "cost": 9.52, "market_value": 9536.0}]
    assert ss.canon_rows("positions", r1) == ss.canon_rows("positions", r2)


def test_checks_catch_inconsistent_numbers():
    assert ss.checks(FUNDS, POS, TRADES, ORDERS) == []
    assert ss.checks({**FUNDS, "available": 503121.80}, POS, TRADES, ORDERS)          # 可用+市值≠总资产
    assert ss.checks(FUNDS, [{**POS[0], "market_value": 5000}], TRADES, ORDERS)        # 持仓市值≠总市值
    assert ss.checks(FUNDS, POS, [{**TRADES[0], "volume": 100}], ORDERS)               # 价×量≠金额 (读错列)
    assert ss.checks(FUNDS, POS, TRADES, [{**ORDERS[0], "traded_volume": 2000}])        # 成交量>委托量
    assert ss.checks({"total_asset": None}, [], [], [])                                # 读不全


def test_snapshot_format_works_with_041_reconcile_and_autoconfirm():
    snap = ss.build_snapshot("close", FUNDS, POS, TRADES, ORDERS, "6735", [], {"tabs": {}})
    assert validate_snapshot(snap) == [] and snap["account_online"] is True and snap["source"] == "screen"
    assert qmt_positions(snap) == {"600000": 1000}
    fills, probs = aggregate_trades(snap["trades"])
    assert probs == [] and fills == [{"code": "600000", "action": "buy", "shares": 1000, "price": 9.52}]
    assert snap["account"].endswith("6735") and snap["asset"]["account_id"][-4:] == "6735"


def test_any_problem_means_offline_so_ledger_is_never_written_from_bad_reads():
    snap = ss.build_snapshot("close", FUNDS, POS, TRADES, ORDERS, "6735", ["「成交」表多次读数不一致"], {})
    assert snap["account_online"] is False and "读屏核对不一致" in snap["error"]
    assert validate_snapshot(snap) == []


def test_codes_and_sides():
    assert ss.xt_code("600000") == "600000.SH" and ss.xt_code("000001") == "000001.SZ"
    assert ss.xt_code("300750") == "300750.SZ" and ss.xt_code("511880") == "511880.SH"
    assert ss.side_type("证券买入") == 23 and ss.side_type("卖出") == 24 and ss.side_type("撤单") is None


def _patch_ui(monkeypatch, fg_results, login_ok=True):
    import qmt_bridge.ui_agent as ua
    import qmt_bridge.ui_channel as uc
    calls = {"login": 0, "fg": 0}
    seq = iter(fg_results)

    def fg(log=print):
        calls["fg"] += 1
        return next(seq)

    def login(log=print):
        calls["login"] += 1
        return login_ok
    monkeypatch.setattr(ua, "ensure_qmt_foreground", fg)
    monkeypatch.setattr(ua, "login_normal_mode", login)
    monkeypatch.setattr(uc, "read", lambda kind, step=None: ({"logged_in": False, "trade_online": False}, {}))
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    return calls


def test_minimized_window_is_restored_not_relogged(monkeypatch):
    calls = _patch_ui(monkeypatch, [[]])                     # 主窗口在 (哪怕刚被最小化, 恢复后没问题)
    snap = ss.take("close", log=lambda m: None)
    assert calls["login"] == 0 and snap["account_online"] is False   # 后面 read 说未登录 -> 离线, 但不重启登录


def test_missing_window_triggers_login_once(monkeypatch):
    calls = _patch_ui(monkeypatch, [["找不到 QMT 主窗口 (未登录或已退出)"], ["QMT 主窗口没能切到最前"]])
    snap = ss.take("close", log=lambda m: None)
    assert calls["login"] == 1 and snap["account_online"] is False and "切到最前" in snap["error"]


def test_wrong_resolution_goes_offline_without_clicking(monkeypatch):
    calls = _patch_ui(monkeypatch, [["屏幕分辨率是 1280x720, 读屏坐标按 1920x1080 量的"]])
    snap = ss.take("close", log=lambda m: None)
    assert calls["login"] == 0 and snap["account_online"] is False and "分辨率" in snap["error"]
