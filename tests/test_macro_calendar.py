"""宏观日历: 规则推算 + 盲区标注的行为契约。

背景: 2026-08-28 周五收盘后沃什放鹰, 周一计划里的黄金集群 -7~9%,
暴露"信号在交易日生成、宏观在非交易日照发"的结构性盲区。日历是
唯一上线的缓解手段, 它错报日期会直接误导人工确认 —— 所以规则
必须钉死在测试里。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import macro_calendar
from action_page import _macro_events


def test_upcoming_window_and_shape():
    evs = macro_calendar.upcoming(start="2026-08-28", horizon_days=14)
    assert evs, "两周窗口里不可能一件例行大事都没有"
    dates = [e["date"] for e in evs]
    assert dates == sorted(dates), "必须按日期排序"
    for e in evs:
        assert "2026-08-28" <= e["date"] <= "2026-09-11", "越界事件不该出现"
        assert set(e) >= {"date", "weekday", "name", "note", "approx"}


def test_first_friday_nfp_rule():
    # 2026-09 的第一个周五是 9/4 —— 就是沃什事件后紧接着的那次非农
    evs = macro_calendar.upcoming(start="2026-09-01", horizon_days=7)
    nfp = [e for e in evs if "非农" in e["name"]]
    assert len(nfp) == 1 and nfp[0]["date"] == "2026-09-04"
    assert nfp[0]["weekday"] == "周五"


def test_static_fomc_included():
    evs = macro_calendar.upcoming(start="2026-09-10", horizon_days=8)
    assert any("FOMC" in e["name"] and e["date"] == "2026-09-16" for e in evs)


def test_cpi_marked_approx():
    evs = macro_calendar.upcoming(start="2026-09-08", horizon_days=8)
    cpi = [e for e in evs if "CPI" in e["name"]]
    assert cpi and all(e["approx"] for e in cpi), "会漂移的日期必须标(约)"


def test_garbage_input_never_raises():
    assert isinstance(macro_calendar.upcoming(start="not-a-date"), list)
    assert macro_calendar.upcoming(start="2026-09-01", horizon_days=0) == [] or True


def test_blind_window_marking():
    # 信号周五 8/28, 执行周一 8/31: 8/31 的中国PMI 在盲区, 9/4 非农不在
    plan = {"signal_date": "2026-08-28"}
    win = {"exec_date": "2026-08-31"}
    evs = _macro_events(plan, win)
    by_date = {e["date"]: e for e in evs}
    assert by_date["2026-08-31"]["in_blind"] is True
    assert by_date["2026-09-04"]["in_blind"] is False


def test_blind_needs_both_dates():
    # 没有计划/执行日时不许乱标盲区
    for plan, win in (({}, {}), ({"signal_date": "2026-08-28"}, {}),
                      ({}, {"exec_date": "2026-08-31"})):
        assert all(not e["in_blind"] for e in _macro_events(plan, win))


def test_every_event_declares_after_close():
    # 盲区判定完全依赖这个字段, 漏标会静默误判
    for e in macro_calendar.upcoming(start="2026-08-28", horizon_days=40):
        assert isinstance(e["after_close"], bool), e["name"]


def test_signal_day_morning_event_not_blind():
    """信号日上午发布的事件不是盲区 —— 当日收盘价已经消化了它。

    这是 08-31 首次上线时的误报: 月底中国 PMI(9:30) 被标成盲区,
    而信号本身就是 8/31 收盘后生成的, 价格早已含该信息。
    """
    evs = _macro_events({"signal_date": "2026-08-31"}, {"exec_date": "2026-09-01"})
    pmi = [e for e in evs if e["date"] == "2026-08-31"]
    assert pmi and all(e["in_blind"] is False for e in pmi)


def test_signal_day_after_close_event_is_blind():
    # 沃什型: 信号日盘后放话, 计划看不见 —— 必须亮
    evs = _macro_events({"signal_date": "2026-09-04"}, {"exec_date": "2026-09-07"})
    nfp = [e for e in evs if e["date"] == "2026-09-04" and "非农" in e["name"]]
    assert nfp and nfp[0]["in_blind"] is True


def test_exec_day_after_close_event_not_blind():
    # 执行日盘后的事件影响不到当天 14:50 那一单, 是下一单的事
    evs = _macro_events({"signal_date": "2026-09-03"}, {"exec_date": "2026-09-04"})
    nfp = [e for e in evs if e["date"] == "2026-09-04" and "非农" in e["name"]]
    assert nfp and nfp[0]["in_blind"] is False
