"""可转债双低回测: 选最低双低、次日成交后第 2 天起计收益、换仓成本; 数据解析。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cb_data  # noqa: E402
import cb_study as cb  # noqa: E402


def _panel():
    days = pd.bdate_range("2022-01-03", periods=12)
    rows = []
    for i, d in enumerate(days):
        rows += [{"date": d, "code": "A", "close": 100 + i, "premium": 5.0},        # 双低 105+i, 每天涨约 1%
                 {"date": d, "code": "B", "close": 120.0, "premium": 30.0},         # 双低 150, 不涨
                 {"date": d, "code": "C", "close": 200.0, "premium": 0.0}]          # 价格 > 150, 不可选
    return pd.DataFrame(rows), days


def test_picks_lowest_and_lags_two_days():
    panel, days = _panel()
    ret, score, elig = cb.prepare(panel)
    assert not elig["C"].any() and elig["A"].iloc[cb.MIN_AGE - 1] and not elig["A"].iloc[cb.MIN_AGE - 2]
    r, k = cb.backtest(ret, score, 1, 5, days[0], days[-1])
    first = cb.MIN_AGE - 1                                     # 第一天有资格的信号日不一定是 t%r==0 那天
    assert k >= 1 and (r.iloc[: first + 2] == 0).all()        # 信号后第 2 天前没有持仓
    held = r[r != 0]
    assert len(held) and np.isclose(held.iloc[0], ret["A"].loc[held.index[0]] - cb.COST)   # 首次建仓扣单边


def test_benchmark_and_metrics():
    panel, days = _panel()
    ret, score, elig = cb.prepare(panel)
    b = cb.benchmark(ret, elig, days[0], days[-1])
    assert len(b) == len(days) and b.iloc[-1] >= 0
    m = cb.metrics(pd.Series([0.01, -0.005, 0.002], index=days[:3]))
    assert set(m) == {"ann", "sharpe", "mdd", "years"} and m["mdd"] < 0


def test_parse_eastmoney_rows():
    df = cb_data.parse([{"DATE": "2024-01-02 00:00:00", "FCLOSE": 118.5, "SWAPOR": 12.3},
                        {"DATE": "2024-01-03 00:00:00", "FCLOSE": None, "SWAPOR": 11.0}], "113527")
    assert df["close"].tolist()[0] == 118.5 and np.isnan(df["close"].iloc[1]) and df["code"].iloc[0] == "113527"
    assert cb_data.parse([], "x").empty
