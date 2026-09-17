import numpy as np
import pandas as pd
import pytest

from scripts.research_risk import CorrelationGuard, close_returns


def panel():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(19)
    a = rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"A": a, "B": a * 2, "C": -a, "D": rng.normal(0, 0.01, len(dates))}, index=dates)


def test_positive_correlation_blocks_but_negative_does_not():
    returns = panel()
    guard = CorrelationGuard(returns)
    assert guard.blocks("B", {"A"}, returns.index[25], 0.8)
    assert not guard.blocks("C", {"A"}, returns.index[25], 0.8)
    assert not guard.blocks("B", set(), returns.index[25], 0.8)


def test_disabled_and_one_threshold_leave_selection_unchanged():
    returns = panel()
    guard = CorrelationGuard(returns)
    assert not guard.blocks("B", {"A"}, returns.index[25], 0)
    assert not guard.blocks("B", {"A"}, returns.index[25], 1)


def test_future_returns_cannot_change_signal_day_decision():
    returns = panel()
    day = returns.index[22]
    baseline = CorrelationGuard(returns).max_correlation("B", {"A"}, day)
    changed = returns.copy()
    changed.loc[changed.index > day, "B"] = -100 * changed.loc[changed.index > day, "A"]
    assert CorrelationGuard(changed).max_correlation("B", {"A"}, day) == pytest.approx(baseline)


def test_missing_history_is_reported_and_not_filled_with_zero():
    returns = panel()
    returns.loc[returns.index[:30], "B"] = np.nan
    guard = CorrelationGuard(returns)
    assert not guard.blocks("B", {"A"}, returns.index[35], 0.8)
    assert guard.unavailable == 1
    assert guard.comparisons == 1
    assert guard.max_correlation("missing", {"A"}, returns.index[35]) is None


def test_constant_returns_are_not_treated_as_correlated():
    returns = panel()
    returns["B"] = 0.0
    assert CorrelationGuard(returns).max_correlation("B", {"A"}, returns.index[-1]) is None


def test_window_excludes_old_correlations():
    returns = panel()
    returns.loc[returns.index[-20:], "B"] = -returns.loc[returns.index[-20:], "A"]
    assert CorrelationGuard(returns).max_correlation("B", {"A"}, returns.index[-1]) == pytest.approx(-1)


def test_newly_selected_stock_is_checked_against_later_candidates():
    returns = panel()
    held = {"C"}
    guard = CorrelationGuard(returns)
    day = returns.index[25]
    assert not guard.blocks("A", held, day, 0.8)
    held.add("A")
    assert guard.blocks("B", held, day, 0.8)


def test_return_calendar_does_not_bridge_missing_sessions():
    dates = pd.bdate_range("2024-01-01", periods=5)
    frames = {"000001": pd.DataFrame({"date": dates[[0, 1, 3, 4]], "close": [10, 11, 13, 14]})}
    returns = close_returns(frames, ["000001.SZ"], dates)
    assert returns.loc[dates[1], "000001"] == pytest.approx(0.1)
    assert pd.isna(returns.loc[dates[2], "000001"])
    assert pd.isna(returns.loc[dates[3], "000001"])
    assert returns.loc[dates[4], "000001"] == pytest.approx(14 / 13 - 1)


@pytest.mark.parametrize("window,minimum", [(1, 1), (20, 1), (20, 21)])
def test_invalid_windows_rejected(window, minimum):
    with pytest.raises(ValueError):
        CorrelationGuard(panel(), window=window, min_periods=minimum)
