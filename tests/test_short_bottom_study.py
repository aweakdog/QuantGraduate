"""做空末位研究: 档期轮换口径与成本。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import short_bottom_study as sb  # noqa: E402


def test_book_holds_from_d_plus_2_for_hold_days():
    idx = pd.bdate_range("2024-01-01", periods=10)
    rets = pd.DataFrame({"A": np.arange(10) / 100.0, "B": 0.0}, index=idx)
    book = sb.book_returns({idx[0]: ["A"]}, rets)
    assert book.isna().tolist() == [True, True] + [False] * 5 + [True] * 3     # 次日收盘进 -> 第 3 天起计 5 天
    assert book.iloc[2] == 0.02 and book.iloc[6] == 0.06
    two = sb.book_returns({idx[0]: ["A"], idx[1]: ["B"]}, rets)
    assert two.iloc[3] == (0.03 + 0.0) / 2                                      # 两档重叠时等权


def test_net_costs_and_sign():
    book = pd.Series([0.01, np.nan])
    s = sb.net(book, -1)
    assert np.isclose(s.iloc[0], -0.01 - 2 * sb.SIDE_COST / sb.HOLD - sb.BORROW / 244) and np.isnan(s.iloc[1])
    assert np.isclose(sb.net(book, 1).iloc[0], 0.01 - 2 * sb.SIDE_COST / sb.HOLD)
