"""界面下单执行器的纯逻辑: 对账/防重复、硬限制、先卖后买与资金、只在双钥匙都开时才提交、出错即停。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qmt_bridge.ui_executor as ux  # noqa: E402

Q = {"600105": {"name": "永鼎股份", "last": 23.76, "up": 26.14, "down": 21.38, "bid1": 23.75},
     "601872": {"name": "招商轮船", "last": 15.67, "up": 17.24, "down": 14.10, "bid1": 15.66}}


def _snap(positions=(), orders=(), cash=100500.0):
    return {"account_online": True, "asset": {"cash": cash},
            "positions": [{"stock_code": c, "volume": v, "can_use_volume": v} for c, v in positions],
            "orders": [{"stock_code": c} for c in orders]}


def _ticket(**kw):
    return {"buy": [{"code": "600105", "shares": 900}, {"code": "601872", "shares": 1200}], "sell": [], "lots": [],
            "line_cash": 100000.0, "trading_enabled": True, "problems": [], **kw}


def test_precheck_reconcile_and_no_duplicates():
    assert ux.precheck(_ticket(), _snap(positions=[("204001.SH", 5020)])) == []          # 逆回购是现金类, 不算持仓
    assert "对账不一致" in ux.precheck(_ticket(), _snap(positions=[("600000.SH", 100)]))[0]
    assert "不重复下单" in ux.precheck(_ticket(), _snap(orders=["600105.SH"]))[0]
    assert ux.precheck(_ticket(problems=["单据过期"]), _snap()) == ["单据过期"]


def test_guard_limits():
    q = Q["600105"]
    assert ux.guard({"side": "buy", "code": "600105", "price": 24.24, "volume": 900}, q, 100000)[0]
    assert not ux.guard({"side": "buy", "code": "600105", "price": 24.24, "volume": 950}, q, 100000)[0]     # 非整百
    assert not ux.guard({"side": "buy", "code": "600105", "price": 26.5, "volume": 100}, q, 100000)[0]      # 超涨停
    assert not ux.guard({"side": "buy", "code": "600105", "price": 25.0, "volume": 100}, q, 100000)[0]      # 偏离 >3%
    assert not ux.guard({"side": "buy", "code": "600105", "price": 24.0, "volume": 5000}, q, 100000)[0]     # 超线现金
    assert ux.guard({"side": "buy", "code": "600105", "price": q["down"], "volume": 100}, {**q, "last": q["down"]}, 1e5)[0]


def test_plan_buys_use_line_cash_not_outside_cash():
    orders, skips = ux.plan(_ticket(), _snap(cash=502000.0), Q)        # 账户里有线外现金, 也只用线内 10 万
    assert [o["code"] for o in orders] == ["600105", "601872"] and all(o["side"] == "buy" for o in orders)
    assert sum(o["price"] * o["volume"] for o in orders) <= 100000
    assert orders[0]["price"] == 24.24                                   # min(涨停, 现价×1.02)


def test_run_stops_on_first_failure_and_shadow_never_submits(monkeypatch, tmp_path):
    import qmt_bridge.rehearse as rh
    import qmt_bridge.screen_snapshot as ss
    monkeypatch.setattr(ux, "DONE_DIR", tmp_path)
    monkeypatch.setattr(ss, "take", lambda kind, log=print: _snap())
    monkeypatch.setattr(ux.Path, "read_text", lambda self, **k: "k")
    monkeypatch.setattr("qmt_bridge.executor.signed_post", lambda *a, **k: (200, {}))
    calls = []

    def fake_submit(o, live, log):
        calls.append((o["code"], live))
        return {"problems": ["读回不一致"] if o["code"] == "600105" else [], "submitted": live}
    monkeypatch.setattr(ux, "submit_one", fake_submit)
    monkeypatch.setattr(ux, "datetime", type("D", (), {"now": staticmethod(lambda: __import__("datetime").datetime(2026, 11, 13, 14, 45))}))
    fq = lambda c: (Q[c]["name"], Q[c]["last"], Q[c]["up"], Q[c]["down"])   # noqa: E731
    rep = ux.run(True, log=lambda m: None, ticket=_ticket(), fetch_quote=fq)
    assert calls == [("600105", True)] and rep["status"] == "stopped"      # 第一笔失败, 第二笔不下
    assert (tmp_path / "2026-11-13.json").exists()                          # 点提交前就落了「今天已下」标记
    assert ux.run(True, log=lambda m: None, ticket=_ticket(), fetch_quote=fq)["status"] == "already_done"
    calls.clear()
    ux.run(False, log=lambda m: None, ticket=_ticket(trading_enabled=True), fetch_quote=fq)
    assert calls and all(live is False for _, live in calls)               # 影子模式绝不提交
    (tmp_path / "2026-11-13.json").unlink()
    calls.clear()
    rep = ux.run(True, log=lambda m: None, ticket=_ticket(trading_enabled=False), fetch_quote=fq)
    assert rep["mode"] == "shadow" and calls and all(live is False for _, live in calls)   # 网页开关关 = 影子
    assert rh  # 模块可导入
