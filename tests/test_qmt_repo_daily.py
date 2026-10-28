"""每日逆回购: 借多少 (策略当天要用的钱不借)、利率取法、硬限制。"""
import qmt_bridge.repo_daily as rd


def test_plan_amount_keeps_strategy_cash_on_buy_days():
    assert rd.plan_amount(503121.79, {"buy": [], "line_cash": 100000})[0] == 502000.0     # 空仓日: 留 1000, 其余借出
    amt, reserve, why = rd.plan_amount(503121.79, {"buy": [{"code": "600000"}], "line_cash": 100000})
    assert amt == 402000.0 and reserve == 101000.0 and "买入" in why                       # 有买入: 线内 10 万不借
    assert rd.plan_amount(503121.79, None)[0] == 402000.0                                  # 取不到单据: 保守
    assert rd.plan_amount(1500.0, {"buy": []})[0] == 0.0                                  # 不足 1000
    assert rd.plan_amount(9e9, {"buy": []})[0] == rd.MAX_AMOUNT


def test_repo_rate_below_bid_on_tick():
    assert rd.repo_rate(1.145) == 1.095 and rd.repo_rate(1.15) == 1.1 and rd.repo_rate(2.003) == 1.95


def test_guard():
    ok = {"code": "204001", "side": "卖出", "price": 1.095, "volume": 5020}
    assert rd.guard_repo_daily(ok, 1.145, 502121.79)[0]
    assert not rd.guard_repo_daily({**ok, "volume": 5025}, 1.145, 502121.79)[0]           # 不是 10 的倍数
    assert not rd.guard_repo_daily({**ok, "volume": 6000}, 1.145, 502121.79)[0]           # 超过可借
    assert not rd.guard_repo_daily({**ok, "side": "买入"}, 1.145, 502121.79)[0]           # 买入 = 借钱
    assert not rd.guard_repo_daily({**ok, "code": "600000"}, 1.145, 502121.79)[0]
    assert not rd.guard_repo_daily({**ok, "price": 1.2}, 1.145, 502121.79)[0]             # 高于买一, 不会成交
    assert not rd.guard_repo_daily(ok, None, 502121.79)[0]


def test_morning_only_on_confirmed_cash_days():
    assert rd.morning_ok({"buy": [], "sell": [], "in_cash": True})
    assert not rd.morning_ok({"buy": [{"code": "600000"}], "sell": []})
    assert not rd.morning_ok({"buy": [], "sell": [{"code": "600000"}]})
    assert not rd.morning_ok(None)                              # 取不到计划: 留给下午


def test_failure_text_and_quiet_statuses():
    t = rd.failure_text({"status": "exception", "error": "HTTPError: HTTP Error 529"}, now="10:30")
    assert "529" in t and "14:40" in t
    assert "不再重试" in rd.failure_text({"status": "readback_mismatch", "bad": ["代码 60001"]}, now="14:41")
    assert {"lent", "defer_to_afternoon", "nothing_to_lend"} <= rd.QUIET and "exception" not in rd.QUIET
