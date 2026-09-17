"""账户「历史操作」页 (ledger_history + /api/ledger 权限)。

守住三条:
  1. 回放口径: 现金按每笔 net 累加, 总资产按执行日收盘复算, 收益率用当时本金;
     存取现金改本金不改收益率, 校准现金改收益率不改本金。
  2. 回放结果必须与当前状态对账, 不一致要显式标出而不是静默展示。
  3. 权限: 账户口令只能看自己名下的线(?all=1 也不放开), 只读口令不能看,
     管理员/站长能看所有线。
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _kline(root, code, rows, cols=("date", "close")):
    kdir = root / "data" / "raw" / "kline"
    kdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(cols)).to_parquet(kdir / f"{code}.parquet")


def _state():
    return {
        "initial_capital": 21000.0,         # 20000 建户 + 1000 存入
        "cash": 9578.0,
        "lots": [{"code": "600000", "shares": 100, "buy_price": 10.0}],
        "history": [
            {"signal_date": "2026-08-01", "exec_date": "2026-08-04",
             "fills": [{"code": "600000", "action": "buy", "shares": 200, "price": 10.0,
                        "fee": 5.0, "net": -2005.0, "source": "manual"},
                       {"code": "000001", "action": "buy", "shares": 100, "price": 50.0,
                        "fee": 5.0, "net": -5005.0, "source": "manual"}],
             "rejected": [{"code": "300001", "action": "buy", "reason": "涨停买不进"}],
             "source": "manual"},
            {"type": "deposit", "at": "2026-08-05T20:00:00", "note": "加钱", "amount": 1000.0,
             "before": {"cash": 12990.0, "equity": 20200.0, "initial_capital": 20000.0},
             "after": {"cash": 13990.0, "equity": 21200.0, "initial_capital": 21000.0}},
            {"type": "set_cash", "at": "2026-08-06T20:00:00", "note": "网页校准", "delta": -10.0,
             "before": {"cash": 13990.0, "equity": 21190.0},
             "after": {"cash": 13980.0, "equity": 21180.0}},
            {"signal_date": "2026-08-10", "exec_date": "2026-08-11",
             "fills": [{"code": "000001", "action": "sell", "shares": 100, "price": 52.0,
                        "fee": 5.0, "net": 5195.0, "reason": "matured", "source": "auto"},
                       {"code": "600000", "action": "roll", "shares": 200, "price": 10.0,
                        "fee": 0.0, "net": 0.0, "reason": "still_ranked", "source": "auto"}],
             "rejected": [], "source": "auto"},
            {"type": "drop_sold", "at": "2026-08-12T21:00:00", "note": "手动卖了一半",
             "lot": {"code": "600000", "name": "浦发", "shares": 100, "buy_price": 10.0,
                     "held_before": 200}, "partial": True, "remaining": 100,
             "sold_at": 11.0, "fee": 5.0,
             "before": {"cash": 19175.0, "equity": 21375.0},
             "after": {"cash": 20270.0, "equity": 21370.0}},
            {"type": "withdraw", "at": "2026-08-13T20:00:00", "note": "取钱", "amount": -10692.0,
             "before": {"cash": 20270.0, "equity": 21370.0, "initial_capital": 21000.0},
             "after": {"cash": 9578.0, "equity": 10678.0, "initial_capital": 10308.0}},
        ],
        "strategy_epochs": [{"since": "2026-08-04", "config": {}},
                            {"since": "2026-08-10", "config": {}, "note": "改持仓数"}],
    }


def test_replay_equity_and_return(tmp_path):
    from ledger_history import build_ledger
    _kline(tmp_path, "600000", [("2026-08-04", 10.5), ("2026-08-11", 11.0)])
    _kline(tmp_path, "000001", [("2026-08-04", 51.0), ("2026-08-11", 52.0)], cols=("时间", "收盘价"))
    st = _state()
    st["initial_capital"] = 10308.0          # 20000 + 1000 - 10692
    st["cash"] = 9578.0
    st["lots"] = [{"code": "600000", "shares": 100, "buy_price": 10.0}]
    out = build_ledger(tmp_path, "demo", st, names={"600000": "浦发", "000001": "平安"})
    rows, s = out["rows"], out["summary"]
    assert s["initial_capital_at_start"] == 20000.0
    first = [r for r in rows if r["date"] == "2026-08-04"]
    assert [r["kind"] for r in first] == ["买入", "买入", "未成交"]
    # 现金 20000 - 2005 - 5005 = 12990; 市值 200*10.5 + 100*51 = 7200 -> 总资产 20190
    assert first[0]["cash_after"] == 12990.0
    assert first[0]["equity_after"] == 20190.0
    assert first[0]["return_pct_after"] == pytest.approx(0.95, abs=0.01)
    assert first[2]["note"].startswith("计划买入未成交")
    dep = next(r for r in rows if r["kind"] == "存入现金")
    assert dep["capital_after"] == 21000.0 and dep["cash_after"] == 13990.0
    assert dep["return_pct_after"] == pytest.approx(21200 / 21000 * 100 - 100, abs=0.01)
    cal = next(r for r in rows if r["kind"] == "校准现金")
    assert cal["capital_after"] == 21000.0 and cal["cash_delta"] == -10.0
    day2 = [r for r in rows if r["date"] == "2026-08-11"]
    assert [r["kind"] for r in day2] == ["卖出", "续持"]
    assert day2[0]["note"] == "持满到期"
    # 现金 13980 + 5195 = 19175; 持仓 200*11 = 2200 -> 21375
    assert day2[0]["cash_after"] == 19175.0 and day2[0]["equity_after"] == 21375.0
    drop = next(r for r in rows if r["kind"].startswith("删除持仓"))
    assert drop["code"] == "600000" and drop["shares"] == 100 and drop["cash_after"] == 20270.0
    wd = next(r for r in rows if r["kind"] == "取出现金")
    assert wd["capital_after"] == 10308.0 and wd["cash_after"] == 9578.0
    assert any(r["kind"] == "策略配置切换" and r["date"] == "2026-08-10" for r in rows)
    assert s["reconciled"] is True and s["positions_match_state"] is True
    assert s["n_trades"] == 3


def test_daily_curve_revalues_between_events(tmp_path):
    """两次变动之间只换收盘价; 存入现金当天收益率基本不变; 回撤按净值峰值算"""
    from ledger_history import build_ledger
    _kline(tmp_path, "600000", [("2026-08-04", 10.5), ("2026-08-05", 12.0), ("2026-08-06", 9.0),
                                ("2026-08-07", 9.5), ("2026-08-11", 11.0)])
    _kline(tmp_path, "000001", [("2026-08-04", 51.0), ("2026-08-05", 51.0), ("2026-08-06", 51.0),
                                ("2026-08-07", 51.0), ("2026-08-11", 52.0)])
    st = _state()
    st["initial_capital"] = 10308.0
    st["calendar"] = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
                      "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"]
    out = build_ledger(tmp_path, "demo", st, names={})
    curve = {c["date"]: c for c in out["curve"]}
    assert min(curve) == "2026-08-04" and max(curve) == "2026-08-13"
    # 08-04 收盘: 现金 12990, 市值 200*10.5 + 100*51 = 7200 -> 20190 (与明细表同值)
    assert curve["2026-08-04"]["equity"] == 20190.0
    # 08-05 盘后存入 1000: 当日快照取当天最后一次变动 -> 现金 13990 / 本金 21000;
    # 市值 200*12 + 100*51 = 7500 -> 21490; 存入不该明显改变收益率
    assert curve["2026-08-05"]["cash"] == 13990.0 and curve["2026-08-05"]["capital"] == 21000.0
    assert curve["2026-08-05"]["equity"] == 21490.0
    assert curve["2026-08-05"]["return_pct"] == pytest.approx(21490 / 21000 * 100 - 100, abs=0.01)
    d6 = curve["2026-08-06"]
    assert d6["cash"] == 13980.0 and d6["capital"] == 21000.0
    assert d6["equity"] == pytest.approx(13980 + 200 * 9.0 + 100 * 51.0)
    assert d6["drawdown_pct"] < 0                              # 从 08-05 高点回落
    # 08-10 停牌(无K线): 沿用 08-07 收盘并打标
    assert curve["2026-08-10"]["flag"] == "approx"
    assert curve["2026-08-13"]["n_positions"] == 1 and curve["2026-08-13"]["capital"] == 10308.0
    stats = out["summary"]["curve_stats"]
    assert stats["n_days"] == len(out["curve"]) and stats["annualized_pct"] is None
    assert stats["max_drawdown_pct"] == min(c["drawdown_pct"] for c in out["curve"])
    assert stats["invested_days"] == len(out["curve"])


def test_unreconciled_is_flagged(tmp_path):
    from ledger_history import build_ledger
    _kline(tmp_path, "600000", [("2026-08-04", 10.5), ("2026-08-11", 11.0)])
    _kline(tmp_path, "000001", [("2026-08-04", 51.0), ("2026-08-11", 52.0)])
    st = _state()
    st["cash"] = 9999.0                       # 与回放对不上
    out = build_ledger(tmp_path, "demo", st, names={})
    assert out["summary"]["reconciled"] is False
    assert out["summary"]["cash_diff_vs_state"] == pytest.approx(9999.0 - 9578.0)
    assert "不完全一致" in out["summary"]["note"]


def test_suspended_and_missing_quotes_are_marked(tmp_path):
    from ledger_history import build_ledger
    _kline(tmp_path, "600000", [("2026-08-03", 9.0)])   # 执行日停牌 -> 用前一日
    st = {"initial_capital": 1000.0, "cash": 100.0,
          "lots": [{"code": "600000", "shares": 100, "buy_price": 9.0}],
          "history": [{"signal_date": "2026-08-03", "exec_date": "2026-08-04",
                       "fills": [{"code": "600000", "action": "buy", "shares": 100, "price": 9.0,
                                  "fee": 0.0, "net": -900.0, "source": "auto"}],
                       "rejected": [], "source": "auto"}]}
    out = build_ledger(tmp_path, "demo", st, names={})
    row = out["rows"][-1]
    assert row["equity_after"] == 1000.0 and out["summary"]["approx_quote_dates"] == ["2026-08-04"]
    st["history"][0]["fills"][0]["code"] = "999999"
    st["lots"][0]["code"] = "999999"
    out = build_ledger(tmp_path, "demo", st, names={})
    assert out["summary"]["unknown_price_codes"] == ["999999"]
    assert out["rows"][-1]["equity_after"] == 1000.0     # 退回成本价, 不编 0


def test_xlsx_roundtrip(tmp_path):
    from ledger_history import build_ledger, ledger_xlsx
    from openpyxl import load_workbook
    _kline(tmp_path, "600000", [("2026-08-04", 10.5), ("2026-08-11", 11.0)])
    _kline(tmp_path, "000001", [("2026-08-04", 51.0), ("2026-08-11", 52.0)])
    st = _state()
    st["initial_capital"] = 10308.0
    out = build_ledger(tmp_path, "demo", st, names={})
    data = ledger_xlsx("demo", "示例账户", out)
    wb = load_workbook(io.BytesIO(data))
    assert wb.sheetnames == ["操作明细", "净值曲线", "说明"]
    ws = wb["操作明细"]
    assert ws["A1"].value == "日期" and ws.max_row == len(out["rows"]) + 1
    heads = [c.value for c in ws[1]]
    kinds = [ws.cell(row=i, column=heads.index("操作") + 1).value for i in range(2, ws.max_row + 1)]
    assert "买入" in kinds and "卖出" in kinds and "存入现金" in kinds
    curve = wb["净值曲线"]
    dates = [curve.cell(row=i, column=1).value for i in range(2, curve.max_row + 1)]
    assert dates == sorted(dates) and "2026-08-11" in dates


class _Req:
    def __init__(self, cookies=None, q=None):
        self.cookies = cookies or {}
        self.query_params = q or {}
        self.method = "GET"


def _token(ws, code):
    exp = int(time.time()) + 3600
    return f"{exp}.c.{ws._code_id(code)}.{ws._view_sign_code(exp, code)}"


def test_ledger_permission_matrix():
    """看: 所有登录身份看所有线; 导出: 只有账户本人/管理员 (2026-09-17 用户口径)"""
    import web_server as ws
    px = _Req({ws.VIEW_COOKIE: _token(ws, "px")}, q={"all": "1"})
    adm = _Req({ws.VIEW_COOKIE: _token(ws, "611611")})
    ro = _Req({ws.VIEW_COOKIE: _token(ws, "213213")})
    # 看
    for req in (px, adm, ro):
        assert ws._ledger_deny(req, "steady5w") is None
        assert ws._ledger_deny(req, "aggr5w") is None
    assert ws._ledger_deny(_Req(), "aggr5w").status_code == 401
    assert ws._ledger_deny(adm, "nope").status_code == 400
    # 导出
    assert ws._ledger_deny(px, "steady5w", export=True) is None
    assert ws._ledger_deny(px, "aggr2w_px2", export=True) is None
    deny = ws._ledger_deny(px, "aggr5w", export=True)       # 别人的线
    assert deny is not None and deny.status_code == 403
    assert ws._ledger_deny(adm, "aggr5w", export=True) is None
    deny = ws._ledger_deny(ro, "aggr5w", export=True)       # 只读口令
    assert deny is not None and deny.status_code == 403
    assert ws._ledger_deny(_Req(), "aggr5w", export=True).status_code == 401
    assert not any(p.startswith(b) for p in ("/api/ledger", "/api/ledger/xlsx")
                   for b in ws.ACCT_GET_BLOCK), "账户会话必须能访问历史操作接口"
