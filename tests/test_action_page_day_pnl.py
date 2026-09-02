# -*- coding: utf-8 -*-
"""今日盈亏 (_day_quotes): 页面"每股今日涨跌 + 持仓合计"的数据来源。

口径: 每股最近两根日线收盘的差 x 股数; 停牌股(K线停在更早日期)当天
没交易, 不得把它历史那天的涨跌算进"今日"。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _write_kline(root: Path, code: str, rows, cols=("date", "close")):
    kdir = root / "data" / "raw" / "kline"
    kdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(cols)).to_parquet(kdir / f"{code}.parquet")


def test_day_quotes_basic(tmp_path):
    from action_page import _day_quotes
    _write_kline(tmp_path, "600000", [("2026-08-15", 10.0), ("2026-08-17", 10.0),
                                      ("2026-08-18", 10.5)])
    # 乱序写入也必须取到最新两根 (库文件不保证有序)
    _write_kline(tmp_path, "000001", [("2026-08-18", 19.0), ("2026-08-17", 20.0)])
    q = _day_quotes(tmp_path, ["600000", "000001", "999999"])  # 999999 无文件
    assert q["600000"] == {"prev": 10.0, "last": 10.5, "date": "2026-08-18"}
    assert q["000001"] == {"prev": 20.0, "last": 19.0, "date": "2026-08-18"}
    assert "999999" not in q


def test_day_quotes_chinese_columns(tmp_path):
    """旧库文件用中文列名 (时间/收盘价), 也得能读"""
    from action_page import _day_quotes
    _write_kline(tmp_path, "600519", [("2026-08-17", 1500.0), ("2026-08-18", 1530.0)],
                 cols=("时间", "收盘价"))
    q = _day_quotes(tmp_path, ["600519"])
    assert q["600519"] == {"prev": 1500.0, "last": 1530.0, "date": "2026-08-18"}


def test_day_quotes_single_row_skipped(tmp_path):
    """只有一根K线算不出"今日" —— 宁缺毋滥, 不编数字"""
    from action_page import _day_quotes
    _write_kline(tmp_path, "300001", [("2026-08-18", 5.0)])
    assert _day_quotes(tmp_path, ["300001"]) == {}


def _mk_live(root: Path, pid, state, plan):
    live = root / "data" / "live"
    live.mkdir(parents=True, exist_ok=True)
    import json
    (live / f"state_{pid}.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")
    sig = plan["signal_date"]
    (live / f"plan_{pid}_{sig}.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (live / "pipeline_status.json").write_text(
        json.dumps({"ok": True, "finished_at": "2026-08-18T19:00:00",
                    "kline_max_date": "2026-08-18"}), encoding="utf-8")


def test_build_today_day_pnl_semantics(tmp_path):
    """三条口径一起验:
    1) 存量持仓 昨收->今收; 2) 今天刚买的 买入价->今收;
    3) 今天卖掉的保留一天 昨收->卖出价, 计入合计"""
    from action_page import build_today
    pid = "steady5w"
    # 存量股 10 -> 10.5 (+0.5x100=50); 今买股 昨收8 今收9, 但 20:30 买入价 8.8
    # -> 只算 (9-8.8)x100=20; 卖出股 昨收4 卖成 4.2 x 100 = +20
    _write_kline(tmp_path, "600000", [("2026-08-17", 10.0), ("2026-08-18", 10.5)])
    _write_kline(tmp_path, "600111", [("2026-08-17", 8.0), ("2026-08-18", 9.0)])
    _write_kline(tmp_path, "600222", [("2026-08-17", 4.0), ("2026-08-18", 4.4)])
    state = {
        "cash": 1000.0, "initial_capital": 10000.0,
        "calendar": ["2026-08-14", "2026-08-17", "2026-08-18"],
        "config": {"hold_days": 5, "tranche_n": 5},
        "last_signal_date": "2026-08-18",
        "lots": [
            {"code": "600000", "shares": 100, "buy_price": 9.0,
             "open_signal_date": "2026-08-14", "open_date": "2026-08-17"},
            {"code": "600111", "shares": 100, "buy_price": 8.8,
             "open_signal_date": "2026-08-17", "open_date": "2026-08-18"},
        ],
        "history": [
            {"signal_date": "2026-08-17", "exec_date": "2026-08-18",
             "fills": [{"code": "600222", "action": "sell", "shares": 100,
                        "price": 4.2, "fee": 5.0, "net": 415.0},
                       {"code": "600111", "action": "buy", "shares": 100,
                        "price": 8.8, "fee": 5.0, "net": -885.0}]},
        ],
    }
    plan = {"signal_date": "2026-08-18", "generated_at": "2026-08-18T19:00:00",
            "hold": [{"code": "600000", "name": "浦发银行", "ref_close": 10.5},
                     {"code": "600111", "name": "北方稀土", "ref_close": 9.0}],
            "sell": [], "buy": [], "config": {"hold_days": 5}}
    _mk_live(tmp_path, pid, state, plan)

    d = build_today(tmp_path, pid)
    by = {h["code"]: h for h in d["hold"]}
    assert by["600000"]["day_chg"] == 50.0 and not by["600000"]["bought_today"]
    assert by["600111"]["day_chg"] == 20.0 and by["600111"]["bought_today"]
    assert len(d["sold_today"]) == 1
    s = d["sold_today"][0]
    assert s["code"] == "600222" and s["day_chg"] == 20.0 and s["sell_price"] == 4.2
    assert d["account"]["day_pnl"] == 90.0
    assert d["account"]["day_date"] == "2026-08-18"


def test_sold_ghost_disappears_next_day(tmp_path):
    """第二天(基准日前进)昨天卖掉的股不再显示, 也不再计入今日盈亏"""
    from action_page import build_today
    pid = "steady5w"
    _write_kline(tmp_path, "600000", [("2026-08-18", 10.0), ("2026-08-19", 10.1)])
    _write_kline(tmp_path, "600222", [("2026-08-17", 4.0), ("2026-08-18", 4.4)])
    state = {
        "cash": 1000.0, "initial_capital": 10000.0,
        "calendar": ["2026-08-17", "2026-08-18", "2026-08-19"],
        "config": {"hold_days": 5, "tranche_n": 5},
        "last_signal_date": "2026-08-19",
        "lots": [{"code": "600000", "shares": 100, "buy_price": 9.0,
                  "open_signal_date": "2026-08-17", "open_date": "2026-08-18"}],
        "history": [
            {"signal_date": "2026-08-17", "exec_date": "2026-08-18",
             "fills": [{"code": "600222", "action": "sell", "shares": 100,
                        "price": 4.2, "fee": 5.0, "net": 415.0}]},
        ],
    }
    plan = {"signal_date": "2026-08-19", "generated_at": "2026-08-19T19:00:00",
            "hold": [{"code": "600000", "name": "浦发银行", "ref_close": 10.1}],
            "sell": [], "buy": [], "config": {"hold_days": 5}}
    _mk_live(tmp_path, pid, state, plan)

    d = build_today(tmp_path, pid)
    assert d["sold_today"] == []
    # 基准日已到 08-19, 昨天买的今天是存量: 昨收10 -> 今收10.1 = +10
    assert d["account"]["day_pnl"] == 10.0
    assert d["account"]["day_date"] == "2026-08-19"


def test_halted_stock_not_counted_as_today(tmp_path):
    """停牌股: K线最新日期落后于其他股 -> 当日变动记 0, 不进百分比。

    复现 build_today 里的聚合规则 (基准日 = 各股K线最新日期的最大值)。
    """
    from action_page import _day_quotes
    _write_kline(tmp_path, "600000", [("2026-08-17", 10.0), ("2026-08-18", 11.0)])
    _write_kline(tmp_path, "600001", [("2026-08-13", 8.0), ("2026-08-14", 9.0)])  # 停牌
    q = _day_quotes(tmp_path, ["600000", "600001"])
    day_date = max(v["date"] for v in q.values())
    assert day_date == "2026-08-18"
    # 停牌股的 date != 基准日, 页面逻辑应记 0 而不是把 08-14 的 +1 元算进今日
    assert q["600001"]["date"] != day_date
    # 正常股照算
    chg = (q["600000"]["last"] - q["600000"]["prev"]) * 200
    assert round(chg, 2) == 200.0
