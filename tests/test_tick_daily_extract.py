"""逐笔日更抽取顺序: 最新的先抽; T1A 老缺口不再每班重试 (10-30: 两个老缺口本月被重试 90 次, 拖慢新数据)。"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tick_daily_extract as t  # noqa: E402


def test_newest_first_and_old_t1a_gaps_skipped():
    days, stale = t.plan_days({"20261029"}, {"20241127", "20250827", "20261029"}, date(2026, 10, 30))
    assert days == ["20261029"] and stale == ["20241127", "20250827"]


def test_recent_t1a_gap_still_retried_and_tk_gaps_never_skipped():
    days, stale = t.plan_days({"20250101", "20261027"}, {"20250101", "20261020"}, date(2026, 10, 30))
    assert days == ["20261027", "20261020", "20250101"] and stale == []   # tk 缺的老日子照补
