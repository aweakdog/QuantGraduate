import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from scripts.research_labels import (
    apply_alignment,
    build_alignment_panel,
    common_test_start,
    label_horizon,
    prune_features,
    validate_cache_contract,
)


def fixture_data(n=40):
    dates = pd.bdate_range('2024-01-01', periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(17).normal(0, 0.01, n))), index=dates)
    base = pd.DataFrame({'date': dates, 'code': '000001.SZ', 'fwd_5d_ret': (close.shift(-5) / close - 1).to_numpy(), 'feature': np.arange(n)})
    return base, {'000001': pd.DataFrame({'date': dates, 'close': close.to_numpy()})}, dates


def test_shared_start_waits_until_both_purge_windows_have_250_days():
    base, klines, dates = fixture_data(300)
    panel = build_alignment_panel(base, klines)
    assert common_test_start(base, panel, requested=str(dates[255].date())) == str(dates[256].date())


def test_label_uses_next_close_to_sixth_close():
    base, klines, _ = fixture_data()
    panel = build_alignment_panel(base, klines)
    close = klines['000001'].close
    assert panel.lab1_t1close_5d.iloc[0] == pytest.approx(close.iloc[6] / close.iloc[1] - 1)
    assert panel.lab1_t1close_5d.iloc[-6:].isna().all()
    assert label_horizon('5d', 'legacy') == 5
    assert label_horizon('5d', 'common') == label_horizon('5d', 't1close') == 6


def test_common_arms_keep_same_rows_and_labeled_sample_set():
    base, klines, dates = fixture_data()
    original = base.copy()
    klines['000001'].loc[klines['000001'].date == dates[12], 'close'] = np.nan
    panel = build_alignment_panel(base, klines)
    control = apply_alignment(base, panel, 'common')
    aligned = apply_alignment(base, panel, 't1close')
    assert_frame_equal(base, original)
    assert_frame_equal(control[base.columns], base)
    assert_frame_equal(aligned[base.columns], base)
    assert control.lab1_common.equals(aligned.lab1_common)
    assert not panel.lab1_common.iloc[11]
    assert len(control) == len(aligned) == len(base)


def test_six_day_cutoff_excludes_unclosed_targets():
    base, klines, dates = fixture_data()
    panel = build_alignment_panel(base, klines)
    signal = 27
    eligible = base.date < dates[signal - label_horizon('5d', 't1close')]
    changed = {key: value.copy() for key, value in klines.items()}
    changed['000001'].loc[changed['000001'].date >= dates[signal], 'close'] *= 9
    other = build_alignment_panel(base, changed)
    assert_series_equal(panel.loc[eligible, 'lab1_t1close_5d'], other.loc[eligible, 'lab1_t1close_5d'])
    assert_series_equal(panel.loc[eligible, 'lab1_common'], other.loc[eligible, 'lab1_common'])
    unsafe = base.date < dates[signal - 5]
    assert not panel.loc[unsafe, 'lab1_t1close_5d'].equals(other.loc[unsafe, 'lab1_t1close_5d'])


def test_inconsistent_price_vintage_is_excluded_from_common_sample():
    base, klines, _ = fixture_data()
    base.loc[0, 'fwd_5d_ret'] += 0.01
    assert not build_alignment_panel(base, klines).lab1_common.iloc[0]


def test_duplicates_are_rejected():
    base, klines, _ = fixture_data()
    panel = build_alignment_panel(base, klines)
    with pytest.raises(ValueError):
        apply_alignment(base, pd.concat([panel, panel.iloc[:1]]), 'common')


def test_macro_pruning_preserves_stock_features_order_and_specialists():
    features = ['a50_futures_chg_5d', 'mkt_vol_20d', 'mtss_1d_ma20', 'ovn_mean_20d', 't1a_big_buy_ma5', 't1b_ovn20', 'cn_pmi', 'tk_rv_5m_xz_ma5']
    exo = prune_features(features, 'NOEXO')
    both = prune_features(features, 'NOMACRO')
    assert exo == [f for f in features if f not in ['a50_futures_chg_5d', 'cn_pmi']]
    assert both == [f for f in exo if not f.startswith('mkt_')]
    assert prune_features(features, 'FULL') == features
    assert prune_features(features, 'FULL') is not features


def test_cache_rejects_label_or_feature_mismatch_and_allows_legacy():
    meta = {'label': '5d'}
    validate_cache_contract(meta, 'legacy', None, ['x'])
    with pytest.raises(ValueError):
        validate_cache_contract(meta, 't1close', 'abc', ['x'])
    meta = {'label': '5d', 'label_alignment': 'common', 'label_horizon': 6,
            'label_panel_sha256': 'abc', 'selected_features': ['x']}
    validate_cache_contract(meta, 'common', 'abc', ['x'])
    with pytest.raises(ValueError):
        validate_cache_contract(meta, 'common', 'def', ['x'])
    with pytest.raises(ValueError):
        validate_cache_contract(meta, 'common', 'abc', ['y'])


@pytest.mark.parametrize('label,mode', [('1d', 'common'), ('2d', 't1close'), ('5d', 'wrong'),
                                        ('1d', 'purge6'), ('2d', 'common5')])
def test_invalid_alignment_rejected(label, mode):
    with pytest.raises(ValueError):
        label_horizon(label, mode)


def test_dissection_modes_split_control_into_purge_and_row_filter():
    """N3: CONTROL(common) = purge6(只多截断一天) + common5(只筛共同行), 两者各取其一"""
    from scripts.research_labels import uses_panel
    assert label_horizon('5d', 'purge6') == 6 and label_horizon('5d', 'common5') == 5
    assert not uses_panel('purge6') and not uses_panel('legacy')
    assert uses_panel('common5') and uses_panel('common') and uses_panel('t1close')
    with pytest.raises(ValueError):
        uses_panel('wrong')
    base, klines, _ = fixture_data()
    panel = build_alignment_panel(base, klines)
    assert_frame_equal(apply_alignment(base, panel, 'common5')[base.columns], base)
    with pytest.raises(ValueError):
        apply_alignment(base, panel, 'purge6')          # 不需要侧表的口径不得误用侧表
    # 缓存合同: purge6 用 6 日截断且无侧表; common5 用 5 日截断且必须带侧表指纹
    validate_cache_contract({'label': '5d', 'label_alignment': 'purge6', 'label_horizon': 6,
                             'label_panel_sha256': None}, 'purge6', None, ['x'])
    validate_cache_contract({'label': '5d', 'label_alignment': 'common5', 'label_horizon': 5,
                             'label_panel_sha256': 'abc'}, 'common5', 'abc', ['x'])
    with pytest.raises(ValueError):
        validate_cache_contract({'label': '5d', 'label_alignment': 'common', 'label_horizon': 6,
                                 'label_panel_sha256': 'abc'}, 'common5', 'abc', ['x'])
    with pytest.raises(ValueError):
        validate_cache_contract({'label': '5d'}, 'purge6', None, ['x'])   # legacy 缓存不能冒充 purge6
