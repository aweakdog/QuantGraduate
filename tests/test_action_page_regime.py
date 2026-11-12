"""网页空仓开关进度卡的数据: 口径必须与 live_signal.build_regime_series 一致 (广度 ≥ 阈值连续 k 天回场)。"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from action_page import _regime_status  # noqa: E402

CFG = {"regime_filter": "breadth", "regime_ma": 20, "regime_breadth": 0.4, "regime_confirm": 2}
DAYS = [date(2026, 11, d) for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]


def _cache(live: Path, rows):
    live.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "breadth_above_ma"]).to_parquet(live / "cache_market_ma20_1.parquet")


def test_need_two_days_when_below_threshold(tmp_path):
    _cache(tmp_path, [("2026-11-04", 0.288), ("2026-11-05", 0.31), ("2026-11-06", 0.3318), ("2026-11-09", 0.45)])
    r = _regime_status(tmp_path, CFG, {"signal_date": "2026-11-06", "in_cash": True, "breadth": 0.3318}, DAYS)
    assert r["breadth"] == 0.3318 and r["gap_pp"] == 6.8 and r["chg_1d_pp"] == 2.2
    assert r["streak"] == 0 and r["need_days"] == 2
    assert r["flip_signal_date"] == "2026-11-10" and r["act_date"] == "2026-11-11"     # 周五信号 -> 周一、周二达标 -> 周三买
    assert [x["date"] for x in r["recent"]] == ["2026-11-04", "2026-11-05", "2026-11-06"]   # 截到信号日, 不偷看之后


def test_one_day_already_counts(tmp_path):
    _cache(tmp_path, [("2026-11-05", 0.35), ("2026-11-06", 0.42)])
    r = _regime_status(tmp_path, CFG, {"signal_date": "2026-11-06", "in_cash": True, "breadth": 0.42}, DAYS)
    assert r["streak"] == 1 and r["need_days"] == 1 and r["gap_pp"] == 0.0
    assert r["flip_signal_date"] == "2026-11-09" and r["act_date"] == "2026-11-10"


def test_stale_cache_falls_back_to_plan_only(tmp_path):
    _cache(tmp_path, [("2026-11-04", 0.45), ("2026-11-05", 0.46)])                  # 缓存没有信号日那天
    r = _regime_status(tmp_path, CFG, {"signal_date": "2026-11-06", "in_cash": True, "breadth": 0.41}, DAYS)
    assert [x["date"] for x in r["recent"]] == ["2026-11-06"] and r["streak"] == 1 and r["chg_1d_pp"] is None


def test_weak_signal_day_explains_why_no_buy(tmp_path):
    """不空仓、有空位, 但最高预测不到 --min-pred 门槛: 横幅要说清楚是"模型没信心", 不是"等换仓日" """
    import json

    from action_page import build_today
    live = tmp_path / "data" / "live"
    live.mkdir(parents=True)
    state = {"cash": 100000.0, "initial_capital": 100000.0, "lots": [], "history": [],
             "calendar": ["2026-11-10", "2026-11-11"], "last_signal_date": "2026-11-11",
             "config": CFG | {"hold_days": 5, "tranche_n": 5, "exec_mode": "t1close"}}
    plan = {"signal_date": "2026-11-11", "generated_at": "2026-11-11T18:51:00", "in_cash": False, "is_rebal": True,
            "breadth": 0.575, "min_pred": 0.002, "buy": [], "sell": [], "hold": [],
            "recommend": [{"rank": 1, "code": "000301", "pred": 0.001783}, {"rank": 2, "code": "600029", "pred": 0.0015}]}
    (live / "state_qmt10w.json").write_text(json.dumps(state), encoding="utf-8")
    (live / "plan_qmt10w_2026-11-11.json").write_text(json.dumps(plan), encoding="utf-8")
    (live / "pipeline_status.json").write_text(json.dumps(
        {"ok": True, "finished_at": "2026-11-11T18:20:00", "kline_max_date": "2026-11-11"}), encoding="utf-8")
    d = build_today(tmp_path, "qmt10w")
    assert d["action"] == "none" and "0.18%" in d["subline"] and "0.2%" in d["subline"] and "补买" in d["subline"]


def test_invested_counts_bad_days_and_other_filters_skip(tmp_path):
    _cache(tmp_path, [("2026-11-05", 0.5), ("2026-11-06", 0.38)])
    r = _regime_status(tmp_path, CFG, {"signal_date": "2026-11-06", "in_cash": False, "breadth": 0.38}, DAYS)
    assert r["streak"] == 1 and r["need_days"] == 1 and r["flip_signal_date"] == "2026-11-09"
    assert _regime_status(tmp_path, CFG | {"regime_filter": "ma"}, {"signal_date": "2026-11-06", "breadth": 0.3}, DAYS) is None
    assert _regime_status(tmp_path, CFG, None, DAYS) is None
