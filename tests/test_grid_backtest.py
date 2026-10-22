"""网格回测器的成交语义: 震荡赚钱、单边下跌亏、T+1、最低佣金、跳空按开盘价、离场清仓、成本守恒。"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grid_backtest import GridParams, band_of, constant_mix, run_grid, target_of_band  # noqa: E402

NOFEE = dict(fee_rate=0.0, min_fee=0.0, slippage=0.0)


def bars(closes, wiggle=0.0, opens=None):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wiggle),
                         "low": np.minimum(o, c) * (1 - wiggle), "close": c}, index=idx)


def test_band_and_target_are_a_monotone_step_function():
    p = GridParams(step=0.02, levels=10, base=0.5)
    assert band_of(100, 100, p) == 0 and band_of(101.99, 100, p) == 0 and band_of(102.0, 100, p) == 1
    assert band_of(99.9, 100, p) == -1
    assert target_of_band(0, p) == 0.5 and target_of_band(-10, p) == 1.0 and target_of_band(10, p) == 0.0
    assert math.isclose(target_of_band(-3, p), 0.65) and target_of_band(-20, p) == 1.0


def test_oscillation_earns_one_step_per_round_trip():
    osc = bars([100, 96, 100, 96, 100, 96, 100, 96, 100, 96, 100] * 3)
    out, s = run_grid(osc, GridParams(step=0.02, **NOFEE))
    # 每个 100->96->100 来回: 98.04 买/100 卖 + 96.12 买/98.04 卖 = 两份各赚一个步长;
    # 每份 = 10万*0.05/100 = 50 股, 一个来回约 50*(1.96+1.92) ≈ 194 元, 16 个来回 ≈ 3.1 万... 量级 >2%
    assert s["total_return_pct"] > 2 and s["n_trades"] >= 60
    hold, hs = constant_mix(osc, 0.5, rebalance_days=0, p=GridParams(**NOFEE))
    assert s["total_return_pct"] > hs["total_return_pct"]                 # 比半仓持有多赚震荡钱
    down = bars(list(np.linspace(100, 70, 40)))
    _, sd = run_grid(down, GridParams(step=0.02, **NOFEE))
    assert sd["total_return_pct"] < -10                                   # 单边跌: 越跌越买, 亏得比半仓多
    _, hd = constant_mix(down, 0.5, p=GridParams(**NOFEE))
    assert sd["total_return_pct"] < hd["total_return_pct"]


def test_t1_blocks_same_day_round_trip_but_t0_allows_it():
    # 无底仓(base=0)。第 2 天: 开盘 100 -> 先探到 97 (买 1 格 @98.04) -> 再冲到 104 (想 @100 卖掉) -> 收 98;
    # 第 3 天收 97。T+0 当天就锁定了一个步长; T+1 卖不掉刚买的, 拿到第 3 天反而亏
    df = bars([100, 98, 97], opens=[100, 100, 97])
    df.iloc[1, df.columns.get_loc("low")] = 97.0
    df.iloc[1, df.columns.get_loc("high")] = 104.0
    _, s1 = run_grid(df, GridParams(step=0.02, levels=10, base=0.0, **NOFEE))
    _, s0 = run_grid(df, GridParams(step=0.02, levels=10, base=0.0, t0=True, **NOFEE))
    assert s0["total_return_pct"] > 0 > s1["total_return_pct"]
    assert s0["n_trades"] > s1["n_trades"]


def test_blocked_t1_sell_is_caught_up_next_day_even_if_price_flat():
    df = bars([100, 100, 100], opens=[100, 100, 100])
    df.iloc[1, df.columns.get_loc("low")] = 95.0          # 第 2 天探低买 2 格, 收回 100, T+1 卖不掉
    out, s = run_grid(df, GridParams(step=0.02, levels=10, base=0.0, **NOFEE))
    assert out["exposure"].iloc[1] > 0 and out["exposure"].iloc[2] == 0   # 第 3 天开盘补卖
    assert s["total_return_pct"] > 0


def test_min_fee_bites_small_grids():
    osc = bars([100, 98, 100, 98, 100, 98, 100] * 5)
    _, cheap = run_grid(osc, GridParams(step=0.015, fee_rate=1e-4, min_fee=0.1, slippage=0.0), capital=20000)
    _, dear = run_grid(osc, GridParams(step=0.015, fee_rate=1e-4, min_fee=5.0, slippage=0.0), capital=20000)
    assert dear["fees"] > cheap["fees"] * 10 and dear["total_return_pct"] < cheap["total_return_pct"]


def test_gap_down_fills_at_open_not_at_level():
    df = bars([100, 90, 90], opens=[100, 90, 90])          # 第二天直接低开 10%
    out, s = run_grid(df, GridParams(step=0.02, levels=10, **NOFEE))
    # 跳空跌过 98.04/96.12/94.23/92.38/90.57 共 5 档: 5 份全按 90 成交 (不是按挂单价), 当日净值变化只来自底仓下跌
    eq0, eq1 = out["equity"].iloc[0], out["equity"].iloc[1]
    assert math.isclose(eq1, eq0 * (1 - 0.5 * 0.1), rel_tol=1e-6)
    assert s["n_trades"] == 1 + 5 and out["exposure"].iloc[1] > 0.7


def test_inactive_days_flatten_and_reentry_rebuilds_base():
    df = bars([100] * 10)
    active = pd.Series([True] * 3 + [False] * 4 + [True] * 3, index=df.index)
    out, _ = run_grid(df, GridParams(**NOFEE), active=active)
    assert out["exposure"].iloc[2] > 0.49 and out["exposure"].iloc[4] == 0 and out["exposure"].iloc[8] > 0.49


def test_costs_are_conserved_in_equity():
    osc = bars([100, 97, 100, 97, 100] * 4, wiggle=0.005)
    _, a = run_grid(osc, GridParams(step=0.01, fee_rate=0.0, min_fee=0.0, slippage=0.0))
    _, b = run_grid(osc, GridParams(step=0.01, fee_rate=1e-3, min_fee=0.0, slippage=0.0))
    assert b["total_return_pct"] < a["total_return_pct"] and b["fees"] > 0
