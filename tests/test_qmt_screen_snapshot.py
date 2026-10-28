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
    assert ss.xt_code("204001") == "204001.SH" and ss.xt_code("131810") == "131810.SZ"     # 沪/深逆回购
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


def test_classify_picks_a_reason_for_the_email():
    assert ss.classify([]) is None
    assert ss.classify(["切换「持仓」页签失败 (复核: ...)"]) == "layout"
    assert ss.classify(["「成交」表多次读数不一致"]) == "read"
    assert ss.classify(["屏幕分辨率是 1280x720, 读屏坐标按 1920x1080 量的"]) == "window"
    assert ss.classify(["QMT 普通方式登录失败"]) == "login"
    assert ss.classify(["读屏程序异常: HTTPError: 401"]) == "exception"


def test_version_note_once_per_change(tmp_path):
    f = tmp_path / "seen.txt"
    assert ss.version_note("2.1.31.3", f) is None                        # 第一次只记下, 不报
    assert ss.version_note("2.1.31.3", f) is None
    assert ss.version_note("2.1.32.0", f) == "QMT 客户端版本 2.1.31.3 -> 2.1.32.0"
    assert ss.version_note("2.1.32.0", f) is None


def test_crash_still_pushes_offline_snapshot(monkeypatch, tmp_path):
    pushed = []
    monkeypatch.setattr(ss, "take", lambda kind, log=print: (_ for _ in ()).throw(RuntimeError("DeepSeek 429")))
    monkeypatch.setattr(ss, "push", lambda snap, url, key: pushed.append(snap) or (200, "{}"))
    monkeypatch.setattr(ss, "qmt_version", lambda root=None: None)
    key = tmp_path / "k"
    key.write_text("x")
    monkeypatch.setattr(ss.Path, "resolve", lambda self: tmp_path / "a" / "b" / "c.py")
    assert ss.main(["--kind", "close", "--key-file", str(key)]) == 0
    snap = pushed[0]
    assert snap["account_online"] is False and snap["error_kind"] == "exception" and "DeepSeek 429" in snap["error"]


def test_wrong_account_or_page_is_layout_problem():
    assert ss.classify(["资金表账号尾号 None 不是预期的 6735 (可能停在别的页签)"]) == "layout"
    assert ss.classify(["切到「股票交易」页签失败 (复核: ...)"]) == "layout"


def test_reverse_repo_rows_pass_checks_and_are_cash_for_041():
    funds = {"total_asset": 503121.79, "available": 502121.79, "frozen": 0.0, "market_value": 1000.0}
    pos = [{"code": "888880", "name": "新标准券", "volume": 10, "can_use": 0, "cost": 0.0, "market_value": 1000.0}]
    trades = [{"time": "11:08:41", "code": "204001", "side": "通用回购逆回购", "price": 1.145, "volume": 10, "amount": 1000.0,
               "order_id": "30354"}]
    orders = [{"time": "11:08:41", "code": "204001", "side": "通用回购逆回购", "price": 1.1, "volume": 10, "traded_volume": 10,
               "status": "已成", "order_id": "30354"}]
    assert ss.checks(funds, pos, trades, orders) == []                     # 10-27 实测这里报过「成交行不全」
    assert ss.checks(funds, pos, [{**trades[0], "amount": 10000.0}], orders)   # 张数×100 对不上仍要拦
    snap = ss.build_snapshot("close", funds, pos, trades, orders, "6735", [], {})
    assert snap["positions"][0]["stock_code"] == "888880.SH" and snap["trades"][0]["order_type"] == 24
    from qmt_sync import aggregate_trades, qmt_positions, reconcile
    assert qmt_positions(snap) == {} and aggregate_trades(snap["trades"]) == ([], [])
    r = reconcile(snap, {"cash": 100000.0, "lots": []})
    assert r["positions_match"] is True and r["outside_cash"] == round(502121.79 + 1000.0 - 100000.0, 2)


def test_cross_model_wording_differences_still_agree():
    a = [{"time": "11:08:41", "code": "204001", "side": "卖出", "price": 1.145, "volume": "10张", "amount": "1,000.00",
          "order_id": "30354"}]
    b = [{"time": "11:08:41", "code": "204001", "side": "通用回购逆回购", "price": "1.145", "volume": 10, "amount": 1000,
          "order_id": "666656116735"}]
    assert ss.canon_rows("trades", a) == ss.canon_rows("trades", b)
    assert ss.canon_rows("trades", a) != ss.canon_rows("trades", [{**b[0], "volume": 100}])   # 张数读错仍算不一致 (利率不比)


def test_matured_repo_or_sold_out_zero_rows_are_ignored():
    funds = {"total_asset": 503135.05, "available": 503135.05, "frozen": 0.0, "market_value": 0.0}
    zero = [{"code": "888880", "name": "新标准券", "volume": 0, "can_use": 0, "cost": 0.0, "market_value": 0.0}]
    assert ss.checks(funds, zero, [], []) == []                                  # 10-28 开盘这里报过「持仓行异常」
    assert ss.build_snapshot("morning", funds, zero, [], [], "6735", [], {})["positions"] == []
    assert ss.checks(funds, [{**zero[0], "volume": None}], [], [])                # 股数读不出仍要拦
    assert ss.checks(funds, [{**zero[0], "market_value": 1000.0}], [], [])       # 0 股却有市值仍要拦


def test_repo_rows_compare_only_code_volume_amount():
    a = [{"time": "10:53:41", "code": "204001", "side": "卖出", "price": 1.42, "volume": 5020, "amount": 502000}]
    b = [{"time": "10:53:47", "code": "204001", "side": "通用回购逆回购", "price": 1.4, "volume": "5020张",
          "amount": "502,000.00"}]
    assert ss.canon_rows("trades", a) == ss.canon_rows("trades", b)
    assert ss.canon_rows("trades", a) != ss.canon_rows("trades", [{**b[0], "volume": 5200}])       # 张数读错仍算不一致
    pa = [{"code": "888880", "volume": 5020, "can_use": 0, "cost": 100.0, "market_value": 502000}]
    pb = [{"code": "888880", "volume": 5020, "can_use": None, "cost": 0, "market_value": 502000.0}]
    assert ss.canon_rows("positions", pa) == ss.canon_rows("positions", pb)
    s1 = [{"time": "14:55:01", "code": "600000", "side": "买入", "price": 9.62, "volume": 1000, "amount": 9620}]
    s2 = [{**s1[0], "time": "14:55:07"}]
    assert ss.canon_rows("trades", s1) == ss.canon_rows("trades", s2)                               # 秒数不同不算
    assert ss.canon_rows("trades", s1) != ss.canon_rows("trades", [{**s1[0], "price": 9.26}])      # 股票价格仍要一致


def test_read_agreed_ocr_plus_one_model(monkeypatch):
    import qmt_bridge.ocr_read as ocr
    import qmt_bridge.ui_agent as ua
    monkeypatch.setattr(ua, "_has_key", lambda name: True)
    monkeypatch.setattr(ocr, "available", lambda: True)
    good = {"total_asset": 503121.8, "available": 503121.8, "frozen": 0, "market_value": 0, "account_last4": "6735"}
    monkeypatch.setattr(ocr, "read_table", lambda png, box, kind: dict(good))
    calls = []
    monkeypatch.setattr(ss, "read_box", lambda png, box, prompt, scale, provider=None: calls.append(provider) or dict(good))
    val, meta, _ = ss.read_agreed(b"", ss.FUNDS_BOX, "", ss.canon_funds, ocr_kind="funds")
    assert val["available"] == 503121.8 and meta["sources"] == ["ocr", "minimax"] and calls == ["minimax"]
    bad = {**good, "total_asset": 50312.18, "available": 50312.18}                         # MiniMax 少读一位
    seq = [bad, good]
    calls.clear()
    monkeypatch.setattr(ss, "read_box", lambda png, box, prompt, scale, provider=None: calls.append(provider) or seq.pop(0))
    val, meta, _ = ss.read_agreed(b"", ss.FUNDS_BOX, "", ss.canon_funds, ocr_kind="funds")
    assert val["available"] == 503121.8 and meta["sources"] == ["ocr", "minimax", "deepseek"]   # 第三家与 OCR 一致
    monkeypatch.setattr(ocr, "read_table", lambda png, box, kind: None)                     # OCR 认不出表头
    calls.clear()
    monkeypatch.setattr(ss, "read_box", lambda png, box, prompt, scale, provider=None: calls.append(provider) or dict(good))
    val, meta, _ = ss.read_agreed(b"", ss.FUNDS_BOX, "", ss.canon_funds, ocr_kind="funds")
    assert meta["sources"] == ["deepseek", "minimax"] and val is not None                   # 退回两家模型
