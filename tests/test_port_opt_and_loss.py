import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_port_opt import TrailingVolLookup, inv_vol_weights, winsorize_demean


def test_winsorize_demean_clips_outliers_per_day_and_preserves_zero_mean():
    df = pd.DataFrame({
        "date": ["2025-01-02"] * 10 + ["2025-01-03"] * 10,
        "fwd_5d_ret": [-0.02, -0.01, 0.0, 0.005, 0.01, 0.012, 0.015, 0.02, 0.025, 0.80]
                      + [-0.50, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    })
    out = winsorize_demean(df, "fwd_5d_ret", 0.10)
    # 每日均值为 0
    for _, grp in out.groupby(df["date"]):
        assert abs(grp.mean()) < 1e-12
    # 极端值被明显压制 (原始 +0.80 减去均值约 +0.71, 缩尾后远小于 0.2)
    assert out.iloc[9] < 0.15
    assert out.iloc[10] > -0.15
    # 跨日独立, 第二天不受第一天 +0.80 影响
    assert out.iloc[19] > out.iloc[18] > out.iloc[17]


def test_trailing_vol_lookup_is_strictly_pit():
    dates = pd.bdate_range("2025-01-01", periods=30)
    # 前 25 天平稳小波动, 第 26 天暴涨 100%
    close = np.array([10.0 * (1.005 if i % 2 == 0 else 0.995) for i in range(25)] + [20.0, 10.0, 20.0, 10.0, 20.0])
    kl = {"000001": pd.DataFrame({"date": dates, "close": close})}
    lk = TrailingVolLookup(kl, window=20, min_periods=10)
    v_before = lk.get("000001.SZ", dates[24])
    v_after = lk.get("000001.SZ", dates[25])
    assert v_before is not None and v_after is not None
    assert v_before < 0.02
    assert v_after > 0.15


def test_inv_vol_weights_prefers_lower_vol_and_respects_clip_and_fallback():
    dates = pd.bdate_range("2025-01-01", periods=25)
    # A 低波动 (~0.5%), B 中波动 (~2%), C 极高波动 (~10%), D 缺数据
    kl = {
        "A": pd.DataFrame({"date": dates, "close": [10.0 * (1 + 0.005 * (1 if i % 2 else -1)) for i in range(25)]}),
        "B": pd.DataFrame({"date": dates, "close": [10.0 * (1 + 0.02 * (1 if i % 2 else -1)) for i in range(25)]}),
        "C": pd.DataFrame({"date": dates, "close": [10.0 * (1 + 0.10 * (1 if i % 2 else -1)) for i in range(25)]}),
    }
    lk = TrailingVolLookup(kl, window=20, min_periods=10)
    w = inv_vol_weights(["A", "B", "C", "D"], dates[-1], lk, lo=0.5, hi=2.0)
    assert abs(sum(w) - 1.0) < 1e-12
    assert w[0] > w[1] > w[2]
    # 缺失 D 回落到中位波动 B, 权重应与 B 一致
    assert abs(w[3] - w[1]) < 1e-9
    # 裁剪生效: 最高与最低之比不超过 4.0 (hi/lo)
    assert max(w) / min(w) <= 4.0 + 1e-9
