"""界面填单演练: 键盘只允许数字和小数点 (整串先校验), 绝不提交, 读回核对不一致就停。"""
import pytest

import qmt_bridge.ui_order as uo


def test_type_digits_validates_whole_string_before_pressing(monkeypatch):
    pressed = []
    monkeypatch.setattr(uo, "_key", lambda vk, up=False: pressed.append((vk, up)))
    for bad in ("8.6a", "", "100\n", "-1", "8,6"):
        with pytest.raises(ValueError):
            uo.type_digits(bad)
    assert pressed == []                                   # 一个键都没按
    uo.type_digits("8.73")
    assert [vk for vk, up in pressed if not up] == [0x38, 0xBE, 0x37, 0x33]


def test_no_submit_or_enter_paths_in_module():
    from pathlib import Path
    src = Path(uo.__file__).read_text(encoding="utf-8")
    assert "0x0D" not in src and "VK_RETURN" not in src     # 不按回车
    for forbidden in ("「买入」按钮", "下单按钮", "确认下单"):
        assert forbidden not in src                           # 不定位提交按钮


def test_dry_fill_stops_when_wrong_security_selected(monkeypatch):
    clicks = []
    monkeypatch.setattr(uo, "dismiss_notice", lambda log: True)
    monkeypatch.setattr(uo, "read_form", lambda: {"code": "000001", "side_tab": "买入"})
    monkeypatch.setattr(uo, "click_target", lambda *a, **k: clicks.append(a) or True)
    ok, _ = uo.dry_fill("600000", "8.73", 100, log=lambda m: None)
    assert not ok and clicks == []


def test_dry_fill_verifies_readback(monkeypatch):
    forms = iter([{"code": "600000", "side_tab": "买入", "price": 9.55, "volume": 0},
                  {"code": "600000", "side_tab": "买入", "price": 8.73, "volume": 100}])
    monkeypatch.setattr(uo, "dismiss_notice", lambda log: True)
    monkeypatch.setattr(uo, "read_form", lambda: next(forms))
    monkeypatch.setattr(uo, "set_field", lambda *a, **k: True)
    ok, form = uo.dry_fill("600000", "8.73", 100, log=lambda m: None)
    assert ok and form["volume"] == 100
    forms2 = iter([{"code": "600000", "side_tab": "买入"}, {"code": "600000", "side_tab": "买入", "price": 8.73, "volume": 1000}])
    monkeypatch.setattr(uo, "read_form", lambda: next(forms2))
    assert uo.dry_fill("600000", "8.73", 100, log=lambda m: None)[0] is False   # 数量读回不符
