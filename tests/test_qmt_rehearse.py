"""填单演练: 计划怎么变成委托、OCR 文字怎么读回; 并确保演练程序里根本没有提交路径。"""
from pathlib import Path

import qmt_bridge.rehearse as rh

Q = {"600703": ("三安光电", 10.00, 11.00, 9.00), "000960": ("锡业股份", 20.00, 22.00, 18.00)}


def test_cash_day_uses_ranked_as_buys():
    t = {"buy": [], "sell": [], "line_cash": 100000.0, "ranked": ["600703", "000960", "600000"]}
    o = rh.plan_orders(t, Q, n=5)                      # 600000 没行情: 跳过
    assert [(x["side"], x["code"], x["price"], x["shares"]) for x in o] == [
        ("买入", "600703", 10.2, 1900), ("买入", "000960", 20.4, 900)]   # 10 万/5 = 2 万一笔, 取整百


def test_real_plan_used_when_present_and_limits_respected():
    t = {"buy": [{"code": "600703", "shares": 500}], "sell": [{"code": "000960", "shares": 300}], "line_cash": 1e5,
         "ranked": ["600000"]}
    o = rh.plan_orders(t, {"600703": ("x", 10.9, 11.0, 9.0), "000960": ("y", 18.1, 22.0, 18.0)})
    assert o == [{"side": "卖出", "code": "000960", "name": "y", "price": 18.0, "shares": 300},     # 不低于跌停
                 {"side": "买入", "code": "600703", "name": "x", "price": 11.0, "shares": 500}]    # 不高于涨停


def test_parse_panel_from_ocr_text():
    r = rh.parse_panel("三安光电 10.00 600703 +0.10 +1.0%",
                       "委托价 10.20 最大可买 9800 股 全部 可用1/21/31/41 总量 1900 +股 预估金额[CNY]1 9380.00元")
    assert r == {"code": "600703", "price": 10.2, "volume": 1900, "est_amount": 19380.0}
    assert rh.same(r, {"code": "600703", "price": 10.2, "shares": 1900})
    assert not rh.same({**r, "volume": 190}, {"code": "600703", "price": 10.2, "shares": 1900})


def test_rehearse_never_submits():
    src = Path(rh.__file__).read_text(encoding="utf-8")
    for bad in ("SELL_BTN", "submit", "confirm(", "0x0D", "repo_submit"):
        assert bad not in src, bad
