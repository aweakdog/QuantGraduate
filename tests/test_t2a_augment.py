import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.build_t2a_augmented import COLS, RAW_COLS, lag_features, minute_features


def bars():
    minutes = [930, 931, 932, 933, 934, 935, 1300, 1301, 1302, 1303, 1304, 1305]
    r = np.array([0.001, -0.002, 0.003, -0.001, 0.004, -0.005] * 2)
    return pd.DataFrame({'date': '20240102', 'code': '000001', 'minute': minutes,
                         'close': 100 * np.exp(np.cumsum(r)), 'qc_vol_ratio': 1.0})


def test_lunch_jump_and_auctions_do_not_enter_continuous_moments():
    frame = bars()
    reference = minute_features(frame, min_returns=5)
    changed = frame.copy()
    changed.loc[changed.minute >= 1300, 'close'] *= 10
    changed = pd.concat([changed, pd.DataFrame({'date': ['20240102'] * 2, 'code': ['000001'] * 2,
        'minute': [925, 1457], 'close': [1e6, 1e-6], 'qc_vol_ratio': [1.0, 1.0]})], ignore_index=True)
    assert_frame_equal(reference, minute_features(changed, min_returns=5), atol=1e-10, rtol=1e-10)


def test_insufficient_or_bad_qc_returns_nan_not_zero():
    frame = bars()
    assert minute_features(frame)[RAW_COLS].isna().all().all()
    frame['qc_vol_ratio'] = 0.8
    assert minute_features(frame, min_returns=5)[RAW_COLS].isna().all().all()


def test_daily_moments_have_bounded_ratios():
    row = minute_features(bars(), min_returns=5).iloc[0]
    assert np.isfinite(row[RAW_COLS].to_numpy(dtype=float)).all()
    assert 0 <= row['t2a_down_share'] <= 1
    assert 0 <= row['t2a_jump_share'] <= 1
    assert row['t2a_rkurt'] >= 1


def test_lag_uses_market_calendar_and_preserves_live_edge():
    dates = pd.bdate_range('2024-01-01', periods=9)
    base = pd.DataFrame({'date': dates, 'code': '000001', 'original': np.arange(9)})
    daily = pd.DataFrame({'date': dates[:-1], 'code': '000001', **{c: np.arange(8, dtype=float) for c in RAW_COLS}})
    original = base.copy()
    result = lag_features(base, daily)
    assert_frame_equal(base, original)
    assert result.iloc[-1][COLS].astype(float).tolist() == [5.0] * 4
    assert result.iloc[4][COLS].isna().all()
    changed = daily.copy()
    changed.loc[changed.date >= dates[6], RAW_COLS] = 1000.0
    altered = lag_features(base, changed)
    assert_frame_equal(result[result.date <= dates[6]], altered[altered.date <= dates[6]])


def test_missing_day_is_not_skipped_by_lag_or_rolling():
    dates = pd.bdate_range('2024-01-01', periods=9)
    base = pd.DataFrame({'date': dates, 'code': '000001'})
    daily = pd.DataFrame({'date': dates.delete(5), 'code': '000001', **{c: 1.0 for c in RAW_COLS}})
    result = lag_features(base, daily)
    assert result.loc[result.date == dates[6], COLS].isna().all().all()
