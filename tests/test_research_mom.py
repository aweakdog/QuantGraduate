import numpy as np
import pandas as pd
import pytest

from scripts.research_mom import mom5_panel, neutralize_day


def _klines(n=12):
    dates = pd.bdate_range("2026-01-05", periods=n)
    return {"000001": pd.DataFrame({"date": dates, "close": np.linspace(10, 21, n)}),
            "600000": pd.DataFrame({"date": dates, "close": np.linspace(20, 9, n)})}, dates


def test_mom5_uses_only_closes_up_to_signal_date():
    kl, dates = _klines()
    mom = mom5_panel(kl, ["000001.SZ", "600000.SH"])
    d = dates[7]
    assert mom.loc[d, "000001"] == pytest.approx(kl["000001"].close.iloc[7] / kl["000001"].close.iloc[2] - 1)
    assert mom.iloc[:5].isna().all().all()
    # 改掉信号日之后的价格, 信号日及以前的 mom5 不变
    kl2 = {c: f.assign(close=np.where(f.date > d, f.close * 5, f.close)) for c, f in kl.items()}
    pd.testing.assert_series_equal(mom5_panel(kl2, ["000001", "600000"]).loc[:d].stack(),
                                   mom.loc[:d].stack())


def _day(rng, n=200, load=0.8):
    mom = pd.Series(rng.normal(0, 0.05, n), index=[f"{600000 + i}" for i in range(n)])
    pred = load * mom.rank(pct=True).to_numpy() * 0.01 + rng.normal(0, 0.003, n)
    order = np.argsort(-pred)
    return [f"{mom.index[i]}.SH" for i in order], list(pred[order]), mom


def test_k0_is_bitwise_identity_and_k1_removes_linear_loading():
    rng = np.random.default_rng(3)
    ranked, vals, mom = _day(rng)
    same = neutralize_day(ranked, vals, mom, 0)
    assert same[0] == ranked and same[1] == vals and same[2] == same[3]
    codes, adj, before, after = neutralize_day(ranked, vals, mom, 1.0)
    assert before > 0.3 and abs(after) < 0.05
    assert adj == sorted(adj, reverse=True) and sorted(codes) == sorted(ranked)
    _, _, _, partial = neutralize_day(ranked, vals, mom, 0.34)
    assert 0 < partial < before and partial == pytest.approx(before * 0.66, abs=0.08)


def test_missing_mom_is_neutral_and_bad_inputs_rejected():
    rng = np.random.default_rng(5)
    ranked, vals, mom = _day(rng)
    codes, adj, _, _ = neutralize_day(ranked, vals, None, 0.5)   # 当天没有任何 mom5: 原序
    assert codes == ranked and adj == pytest.approx(vals)
    with pytest.raises(ValueError):
        neutralize_day(ranked, vals, mom, 1.5)
    with pytest.raises(ValueError):
        neutralize_day(ranked, vals[:-1], mom, 0.3)
