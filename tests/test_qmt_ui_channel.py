"""通道测试的硬限制: 只在交易时段、只允许买入 100 股跌停价、读回必须逐项一致。"""
from datetime import datetime

import pytest

import qmt_bridge.ui_channel as ch


def test_session_window():
    assert ch.in_session(datetime(2026, 10, 26, 9, 45))
    assert ch.in_session(datetime(2026, 10, 26, 13, 30))
    for t in ((9, 29), (11, 29), (12, 0), (14, 51), (14, 58), (15, 5), (20, 0)):
        assert not ch.in_session(datetime(2026, 10, 26, *t))      # 午休/收盘竞价/收盘后/夜市都不行


def test_guard_only_allows_limit_down_buy_100():
    exp = ch.parse_expect("600000:BUY:8.58:100")
    assert exp == {"code": "600000", "side": "买入", "price": 8.58, "volume": 100}
    assert ch.guard_channel_order(exp, 8.58)[0]
    assert not ch.guard_channel_order(exp, 8.59)[0]                      # 不是跌停价
    assert not ch.guard_channel_order(exp, None)[0]
    assert not ch.guard_channel_order(ch.parse_expect("600000:BUY:8.58:200"), 8.58)[0]
    assert not ch.guard_channel_order(ch.parse_expect("600000:SELL:8.58:100"), 8.58)[0]
    with pytest.raises(ValueError):
        ch.parse_expect("600000:HOLD:8.58:100")


def test_matches_field_by_field():
    exp = ch.parse_expect("600000:BUY:8.58:100")
    assert ch.matches({"code": "600000", "side": "买入", "price": 8.58, "volume": 100}, exp)[0]
    assert ch.matches({"code": "600000", "side": "证券买入", "price": "8.580", "volume": "100"}, exp)[0]
    ok, bad = ch.matches({"code": "600000", "side": "买入", "price": 8.58, "volume": 1000}, exp)
    assert not ok and "volume" in bad[0]
    assert not ch.matches({"code": "600001", "side": "买入", "price": 8.58, "volume": 100}, exp)[0]
    assert not ch.matches({"code": "600000", "side": "卖出", "price": 8.58, "volume": 100}, exp)[0]
    assert not ch.matches({"code": "600000", "side": "买入", "price": None, "volume": 100}, exp)[0]


def test_submit_refuses_outside_session_or_bad_guard(monkeypatch):
    exp = ch.parse_expect("600000:BUY:8.58:100")
    monkeypatch.setattr(ch, "in_session", lambda now=None: False)
    assert ch.submit_buy(exp, 8.58)["ok"] is False
    monkeypatch.setattr(ch, "in_session", lambda now=None: True)
    clicked = []
    monkeypatch.setattr(ch, "click", lambda *a, **k: clicked.append(a) or {"ok": True})
    assert ch.submit_buy(ch.parse_expect("600000:BUY:9.55:100"), 8.58)["ok"] is False   # 市价附近: 拒
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"form": {"code": "600000", "side_tab": "买入", "price": 8.58,
                                                                        "volume": 1000}, "popups": []}, {}))
    assert ch.submit_buy(exp, 8.58)["ok"] is False                                         # 表单数量不符: 拒
    assert clicked == []


def test_confirm_requires_exactly_one_matching_dialog(monkeypatch):
    exp = ch.parse_expect("600000:BUY:8.58:100")
    monkeypatch.setattr(ch, "in_session", lambda now=None: True)
    clicked = []
    monkeypatch.setattr(ch, "click", lambda *a, **k: clicked.append(a) or {"ok": True})
    dlg = {"title": "委托确认", "code": "600000", "side": "买入", "price": 8.58, "volume": 100, "buttons": ["确定", "取消"]}
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"dialogs": [{**dlg, "volume": 1000}]}, {}))
    assert ch.confirm(exp)["ok"] is False and clicked == []
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"dialogs": [dlg, dlg]}, {}))
    assert ch.confirm(exp)["ok"] is False and clicked == []
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"dialogs": [dlg]}, {}))
    assert ch.confirm(exp)["ok"] is True and "确定" in clicked[0][0]
    # 信息类通知 (没有证券代码, 标题还可能被读成别的字) 不算确认框
    notice = {"title": "系统预警信息", "code": None, "buttons": ["×"]}
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"dialogs": [notice, {**dlg, "title": "下达任务"}]}, {}))
    assert ch.confirm(exp)["ok"] is True


def test_tick_row_only_inside_select_column(monkeypatch):
    clicked = []
    monkeypatch.setattr(ch, "click_xy_checked", lambda x, y, step: clicked.append((x, y)) or {"ok": True})
    assert ch.tick_row(1096, 499)["ok"] is False and clicked == []          # 那是工具栏按钮, 不是勾选列


def test_fixed_point_click_still_needs_verification(monkeypatch):
    clicked = []
    monkeypatch.setattr(ch, "snap", lambda step: (b"png", "p"))
    monkeypatch.setattr(ch, "click_windows", lambda x, y: clicked.append((x, y)))
    monkeypatch.setattr(ch, "verify_point", lambda *a, **k: (False, {"under_cross": "反选", "is_target": False}))
    assert ch.click_xy_verified(1096, 499, "选中撤单", "撤单", "t")["ok"] is False and clicked == []
