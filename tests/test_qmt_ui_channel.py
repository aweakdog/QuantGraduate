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


def test_repo_guard_only_gc001_10_lots_at_or_below_bid():
    ok = ch.parse_expect("204001:SELL:1.105:10")
    assert ch.guard_repo_order(ok, 1.105)[0]
    assert ch.guard_repo_order(ch.parse_expect("204001:SELL:1.0:10"), 1.105)[0]          # 低于买一也能立即成交
    assert not ch.guard_repo_order(ch.parse_expect("204001:SELL:1.11:10"), 1.105)[0]     # 高于买一: 挂着不成交
    assert not ch.guard_repo_order(ch.parse_expect("204001:SELL:0.5:10"), 1.105)[0]      # 偏离太远: 多半填错
    assert not ch.guard_repo_order(ch.parse_expect("204001:SELL:1.105:100"), 1.105)[0]   # 不是 10 张
    assert not ch.guard_repo_order(ch.parse_expect("204001:BUY:1.105:10"), 1.105)[0]     # 买入 = 借钱 (正回购)
    assert not ch.guard_repo_order(ch.parse_expect("600000:SELL:1.105:10"), 1.105)[0]
    assert not ch.guard_repo_order(ok, None)[0]


def test_submit_sell_repo_refuses_before_clicking(monkeypatch):
    clicked = []
    monkeypatch.setattr(ch, "side_by_button", lambda: ("买入", {}))
    monkeypatch.setattr(ch, "in_session", lambda now=None: True)
    assert ch.submit_sell_repo(ch.parse_expect("204001:SELL:1.105:10"), 1.105)["ok"] is False   # 按钮是橙色 (买入)
    monkeypatch.setattr(ch, "side_by_button", lambda: ("卖出", {}))
    monkeypatch.setattr(ch, "click_xy_verified", lambda *a, **k: clicked.append(a) or {"ok": True})
    exp = ch.parse_expect("204001:SELL:1.105:10")
    monkeypatch.setattr(ch, "read_form_zoomed", lambda: ({"code": "204001", "price": 1.105, "volume": 100}, []))
    assert ch.submit_sell_repo(exp, 1.105)["ok"] is False                                      # 表单读回 100 张
    monkeypatch.setattr(ch, "read_form_zoomed", lambda: (None, [{"code": "60001"}, {"code": "204001"}]))
    assert ch.submit_sell_repo(exp, 1.105)["ok"] is False                                      # 两次读数不一致
    monkeypatch.setattr(ch, "read_form_zoomed", lambda: ({"code": "204001", "price": 1.105, "volume": 10,
                                                          "est_amount": 10000.0}, []))
    assert ch.submit_sell_repo(exp, 1.105)["ok"] is False                                      # 预估金额不对
    assert clicked == []
    monkeypatch.setattr(ch, "read_form_zoomed", lambda: ({"code": "204001", "price": 1.105, "volume": 10,
                                                          "est_amount": 1000.0}, []))
    monkeypatch.setattr(ch, "read", lambda kind, step=None: ({"dialogs": []}, {}))
    assert ch.submit_sell_repo(exp, 1.105)["ok"] is True and len(clicked) == 1


def test_click_xy_verified_uses_ocr_first_then_model(monkeypatch):
    import qmt_bridge.ocr_read as ocr
    clicks, model = [], []
    monkeypatch.setattr(ch, "snap", lambda step: (b"png", step))
    monkeypatch.setattr(ch, "click_windows", lambda x, y: clicks.append((x, y)))
    monkeypatch.setattr(ch.time, "sleep", lambda s: None)
    monkeypatch.setattr(ch, "verify_point", lambda *a, **k: model.append(a) or (True, {"is_target": True}))
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "locate_text", lambda png, text, box, scale: (112, 52))
    out = ch.click_xy_verified(110, 51, "股票交易页签", "股票交易", "t")
    assert out["ok"] and out["verify"]["by"] == "ocr" and clicks == [(110, 51)] and model == []   # OCR 确认: 不调模型
    monkeypatch.setattr(ocr, "locate_text", lambda png, text, box, scale: (180, 52))             # 偏 70px: 不算
    out = ch.click_xy_verified(110, 51, "股票交易页签", "股票交易", "t")
    assert out["verify"]["by"] == "model" and len(model) == 1                                     # 退回模型复核
    monkeypatch.setattr(ch, "verify_point", lambda *a, **k: (False, {"is_target": False}))
    monkeypatch.setattr(ocr, "locate_text", lambda png, text, box, scale: None)
    n = len(clicks)
    assert ch.click_xy_verified(110, 51, "股票交易页签", "股票交易", "t")["ok"] is False and len(clicks) == n   # 都不认: 不点
    seen = []
    monkeypatch.setattr(ocr, "locate_text", lambda png, text, box, scale: seen.append(text) or (298, 752))
    assert ch.click_xy_verified(367, 752, "委托价输入框", "", "t", anchor=(298, 752, "委托价"))["verify"]["by"] == "ocr"
    assert seen == ["委托价"]                                                                       # 输入框核对旁边的标签


def test_tab_side_checks_button_color(monkeypatch):
    clicks = []
    monkeypatch.setattr(ch, "click_xy_checked", lambda x, y, step: clicks.append((x, y)) or {"ok": True})
    monkeypatch.setattr(ch, "side_by_button", lambda: ("买入", {}))
    assert ch.tab_side("买入")["ok"] and clicks[-1] == ch.BUY_TAB
    assert ch.tab_side("卖出")["ok"] is False and clicks[-1] == ch.SELL_TAB      # 点了但按钮还是橙色: 不算
