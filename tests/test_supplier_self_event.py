"""
供应商自身事件修正模块 — 价格异常检测逻辑单元测试
"""
import pytest
import pandas as pd
import numpy as np
from pipeline.supplier_self_event import SelfEventChecker


@pytest.fixture
def checker():
    return SelfEventChecker()


@pytest.fixture
def normal_kline():
    """平稳行情（窄幅震荡），不触发任何异常检测"""
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    close = np.array([10.0, 10.01, 10.0, 9.99, 10.02, 10.03, 10.01, 10.0,
                      9.99, 9.98, 10.01, 10.0, 9.99, 10.01, 10.0, 10.02,
                      10.01, 10.0, 9.99, 10.01])
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def crash_kline():
    """大跌K线 — 事件前5日累计跌超8%"""
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    close = np.array([10.0, 10.1, 10.0, 9.9, 10.2, 10.3, 10.1, 10.0, 9.8, 9.7,
                      9.6, 9.5, 9.4, 9.3, 9.2, 9.1, 9.0, 8.8, 8.5, 7.8])
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def spike_kline():
    """大涨K线 — 事件前5日累计涨超3%"""
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    close = np.array([10.0, 10.1, 10.0, 9.9, 10.2, 10.3, 10.1, 10.0, 9.8, 9.7,
                      9.6, 9.5, 9.4, 9.3, 9.2, 9.1, 9.0, 9.3, 9.8, 10.6])
    return pd.DataFrame({"date": dates, "close": close})


# 最后一个交易日的日期: 2026-01-28 (20个工作日从2026-01-01开始)
_LAST_DATE = "2026-01-28"


class TestBacktestProxy:

    def test_normal_kline_returns_1(self, checker, normal_kline):
        """平稳行情 → 修正系数=1.0"""
        result = checker.backtest_proxy(normal_kline, _LAST_DATE)
        assert result["self_event_multiplier"] == 1.0, result["proxy_signal"]

    def test_crash_returns_088(self, checker, crash_kline):
        """-8%大跌 → 修正系数=0.88"""
        result = checker.backtest_proxy(crash_kline, _LAST_DATE)
        assert result["self_event_multiplier"] == 0.88, result["proxy_signal"]

    def test_spike_returns_105(self, checker, spike_kline):
        """+8%大涨 → 修正系数=1.05"""
        result = checker.backtest_proxy(spike_kline, _LAST_DATE)
        assert result["self_event_multiplier"] == 1.05, result["proxy_signal"]

    def test_negative_5pct_returns_092(self, checker):
        """-5%左右 → 修正系数=0.92"""
        dates = pd.date_range("2026-01-01", periods=12, freq="B")
        # 从第四天开始稳步下跌，最后5天累计约 -5%
        close = np.array([10.0, 10.1, 10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2, 9.1])
        df = pd.DataFrame({"date": dates, "close": close})
        result = checker.backtest_proxy(df, "2026-01-16")
        mult = result["self_event_multiplier"]
        assert mult == 0.92, f"Expected 0.92, got {mult}: {result['proxy_signal']}"

    def test_direction_aware_positive(self, checker):
        """2σ 单日大涨→1.05（累计<3%未触发阈值，但单日异常）"""
        dates = pd.date_range("2026-01-01", periods=17, freq="B")
        # 前12天极度平稳，最后5天：窄幅波动后跳涨+2.5%
        # 异常须放在 close[15]（事件前最后一天的价格），因为 backtest_proxy
        # 使用 df[df["date"] < event_dt] ——事件当天数据被排除
        close = np.array([10.0, 10.00, 10.01, 10.00, 9.99, 10.00, 10.01,
                          10.00, 9.99, 10.00, 10.01, 10.00,
                          10.01, 10.00, 10.01, 10.26, 10.30])
        # close[15]=10.26 → +2.5%（事件前最后一天）；close[16]=10.30（事件当天排除）
        df = pd.DataFrame({"date": dates, "close": close})
        result = checker.backtest_proxy(df, str(dates[-1].date()))
        mult = result["self_event_multiplier"]
        # 累计 +2.5%（<3%），当日 +2.5%（异常）→ direction-aware → 1.05
        assert mult == 1.05, f"期望1.05, got {mult}: {result['proxy_signal']}"

    def test_direction_aware_negative(self, checker):
        """2σ 单日大跌→0.95（累计>-3%未触发阈值，但单日异常）"""
        dates = pd.date_range("2026-01-01", periods=17, freq="B")
        # 同上但最后一跳下跌-2.6%，异常在 close[15]（事件前最后一天）
        close = np.array([10.0, 10.00, 10.01, 10.00, 9.99, 10.00, 10.01,
                          10.00, 9.99, 10.00, 10.01, 10.00,
                          10.01, 10.00, 10.01, 9.75, 9.72])
        # close[15]=9.75 → -2.6%（事件前最后一天）；close[16]=9.72（事件当天排除）
        df = pd.DataFrame({"date": dates, "close": close})
        result = checker.backtest_proxy(df, str(dates[-1].date()))
        mult = result["self_event_multiplier"]
        # 累计 -2.6%（>-3%），当日 -2.6%（异常）→ direction-aware → 0.95
        assert mult == 0.95, f"期望0.95, got {mult}: {result['proxy_signal']}"

    def test_no_data_returns_1(self, checker):
        """空数据 → 修正系数=1.0"""
        df = pd.DataFrame({"date": [], "close": []})
        result = checker.backtest_proxy(df, "2026-01-01")
        assert result["self_event_multiplier"] == 1.0

    def test_zero_close_handling(self, checker):
        """收盘价含0 → 不报错"""
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        close = np.array([10.0, 10.1, 0, 9.9, 10.2, 10.3, 10.1, 10.0, 9.8, 9.7])
        df = pd.DataFrame({"date": dates, "close": close})
        result = checker.backtest_proxy(df, "2026-01-14")
        assert "self_event_multiplier" in result
