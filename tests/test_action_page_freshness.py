"""_freshness: 计划落后于行情时, 必须报对原因

背景: 实盘模式(require_confirm)的条线在等你填真实成交价时会【故意】停住,
不记账也不出新信号。此时计划信号日必然落后于最新行情日, 但这不是故障。
旧实现一律报"当日流水线可能未跑或失败", 导致页面顶部让你确认成交、主横幅
却红底写"数据未更新" —— 两条信息自相矛盾, 会把人指去查一个不存在的故障。
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def live(tmp_path):
    """造一个最小的 data/live 目录, 行情最新日 = 2026-08-03"""
    d = tmp_path / "data" / "live"
    d.mkdir(parents=True)
    (d / "pipeline_status.json").write_text(json.dumps({
        "ok": True, "finished_at": "2026-08-03T19:55:38",
        "kline_max_date": "2026-08-03",
    }), encoding="utf-8")
    return tmp_path


# 覆盖 07-31 与 08-03 的交易日历 (中间跳过周末)
CAL = ["2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
       "2026-08-03", "2026-08-04", "2026-08-05"]


def test_awaiting_confirm_not_reported_as_pipeline_failure(live):
    """等确认导致的落后: reason=awaiting_confirm, 且不得出现"流水线"字样"""
    from action_page import _freshness
    state = {
        "calendar": CAL,
        "awaiting_confirm": {"signal_date": "2026-07-31",
                             "exec_date": "2026-08-03",
                             "since": "2026-08-03T17:52:15"},
    }
    f = _freshness(live, state, {"signal_date": "2026-07-31"})
    assert f["stale"] is True
    assert f["reason"] == "awaiting_confirm"
    # 关键: 不能再出现"可能未跑或失败"这种把人指去查故障的措辞
    assert "可能未跑或失败" not in f["note"]
    assert "数据未更新" not in f["note"]
    assert "确认" in f["note"]
    # 执行日就是最新交易日 -> 当晚数据刚到, 属正常等待, 不算逾期
    assert f["awaiting_overdue_days"] == 0


def test_awaiting_confirm_overdue_counts_trading_days(live):
    """执行日已比最新交易日早 N 个交易日: 升级提醒并给出天数"""
    from action_page import _freshness
    state = {
        "calendar": CAL,
        "awaiting_confirm": {"signal_date": "2026-07-28",
                             "exec_date": "2026-07-29",
                             "since": "2026-07-29T17:52:15"},
    }
    f = _freshness(live, state, {"signal_date": "2026-07-28"})
    assert f["reason"] == "awaiting_confirm"
    # 07-29 -> 08-03 之间: 07-30, 07-31, 08-03 = 3 个交易日
    assert f["awaiting_overdue_days"] == 3
    assert "3 个交易日" in f["note"]


def test_real_pipeline_lag_still_reported(live):
    """没有等确认却落后 = 真故障, 必须照旧报流水线可能未跑"""
    from action_page import _freshness
    f = _freshness(live, {"calendar": CAL}, {"signal_date": "2026-07-31"})
    assert f["stale"] is True
    assert f["reason"] == "pipeline"
    assert "流水线" in f["note"]


def test_awaiting_confirm_for_other_plan_is_not_an_excuse(live):
    """等确认的是另一份计划时, 不能拿它解释当前计划的落后

    否则一个陈旧的 awaiting_confirm 会把真实的流水线故障永久掩盖掉。
    """
    from action_page import _freshness
    state = {
        "calendar": CAL,
        "awaiting_confirm": {"signal_date": "2026-07-28",
                             "exec_date": "2026-07-29",
                             "since": "2026-07-29T17:52:15"},
    }
    f = _freshness(live, state, {"signal_date": "2026-07-31"})
    assert f["reason"] == "pipeline"
    assert "流水线" in f["note"]


def test_not_stale_when_plan_is_current(live):
    """计划已是最新交易日: 不报任何异常"""
    from action_page import _freshness
    f = _freshness(live, {"calendar": CAL}, {"signal_date": "2026-08-03"})
    assert f["stale"] is False
    assert f["reason"] == ""
    assert f["note"] == ""


def test_no_plan_at_all(live):
    from action_page import _freshness
    f = _freshness(live, {"calendar": CAL}, None)
    assert f["stale"] is True
    assert f["reason"] == "no_plan"
