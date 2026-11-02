"""空仓期替代资产研究: 空仓段切分、进出成本、超额与判据。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cash_alt_study as ca  # noqa: E402

IDX = pd.date_range("2021-12-27", periods=8, freq="B")


def test_spells_and_costs():
    cash = pd.Series([False, True, True, False, True, False, False, True], index=IDX)
    assert ca.spells(cash) == [(IDX[1], IDX[2]), (IDX[4], IDX[4]), (IDX[7], IDX[7])]
    r = ca.held_returns(pd.Series(0.01, index=IDX), cash, cost=0.001)
    assert r.isna().tolist() == [True, False, False, True, False, True, True, False]
    assert round(r[IDX[1]], 6) == 0.009 and round(r[IDX[4]], 6) == 0.008        # 单日段: 进出各扣一次


def test_short_sign_and_carry():
    cash = pd.Series(True, index=IDX)
    r = ca.held_returns(pd.Series(-0.01, index=IDX), cash, cost=0.0, sign=-1, carry=0.252)
    assert round(r.iloc[3], 6) == round(0.01 - 0.001, 6)                        # 做空赚指数跌幅, 扣每日贴水


def test_evaluate_excess_and_splits():
    cash = pd.Series(True, index=IDX)
    asset = pd.Series(0.001, index=IDX)
    base = pd.Series(0.0, index=IDX)
    r = ca.evaluate(asset, base, cash)
    assert r["cash_days"] == 8 and r["spells"] == 1 and r["win_rate"] == 1.0
    assert r["S1_cum_excess"] > 0 and r["S2_cum_excess"] > 0                     # 跨 2021/2022 两半都有
    assert ca.gate_c1({**r, "worst_spell": -0.01}) == (r["ann_excess"] >= 0.015)
    assert not ca.gate_c1({**r, "S2_cum_excess": -0.001})
