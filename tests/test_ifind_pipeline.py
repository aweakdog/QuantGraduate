"""
测试 iFinD 基本面数据管线 — 重试逻辑 + 数据解析
"""
import json
from unittest.mock import patch, Mock

import pandas as pd
import pytest
import requests

from pipeline.ifind_funda_pipeline import mcp_call, parse_table, standardize_financials


class TestMcpCall:
    """mcp_call 重试与退避逻辑"""

    def test_success_first_try(self):
        """首次成功应直接返回结果"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        with patch("pipeline.ifind_funda_pipeline.requests.post", return_value=mock_resp) as mock_post:
            result = mcp_call("test_tool", {"q": "test"})
            assert result == {"result": "ok"}
            assert mock_post.call_count == 1

    def test_retry_on_429(self):
        """429 限流应触发指数退避重试后成功"""
        resp_429 = Mock(status_code=429)
        # 429 会触发 raise_for_status() → HTTPError
        resp_429.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        resp_ok = Mock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {"result": "ok"}

        with patch("pipeline.ifind_funda_pipeline.requests.post",
                   side_effect=[resp_429, resp_ok]) as mock_post:
            with patch("pipeline.ifind_funda_pipeline.time.sleep") as mock_sleep:
                result = mcp_call("test_tool", {"q": "test"}, max_retries=2)
                assert result == {"result": "ok"}
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(10)

    def test_all_429_returns_empty(self):
        """全部 429 超出重试次数应返回 {}"""
        resp_429 = Mock(status_code=429)
        resp_429.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")

        with patch("pipeline.ifind_funda_pipeline.requests.post",
                   return_value=resp_429) as mock_post:
            with patch("pipeline.ifind_funda_pipeline.time.sleep"):
                result = mcp_call("test_tool", {"q": "test"}, max_retries=2)
                assert result == {}
                assert mock_post.call_count == 2

    def test_retry_on_network_error(self):
        """网络错误也应触发重试"""
        with patch("pipeline.ifind_funda_pipeline.requests.post",
                   side_effect=[requests.ConnectionError("timeout"),
                                Mock(status_code=200, json=lambda: {"ok": True})]) as mock_post:
            with patch("pipeline.ifind_funda_pipeline.time.sleep"):
                result = mcp_call("test_tool", {"q": "test"}, max_retries=2)
                assert result == {"ok": True}
                assert mock_post.call_count == 2


class TestParseTable:
    """Markdown 表格解析"""

    def test_parse_valid_table(self):
        md = """| 年度 | 营业收入 | 净利润 |
| --- | --- | --- |
| 2020 | 100亿 | 10亿 |
| 2021 | 120亿 | 12亿 |"""
        df = parse_table(md)
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["年度", "营业收入", "净利润"]

    def test_parse_too_few_lines(self):
        assert parse_table("|a|b|") is None

    def test_parse_malformed(self):
        assert parse_table("not a table") is None


class TestStandardizeFinancials:
    """基本面数据标准化"""

    def test_standardize_normal(self):
        df_raw = pd.DataFrame({
            "年度": ["2020-12-31", "2021-12-31"],
            "营业收入": ["100亿", "120亿"],
            "净利润": ["10亿", "12亿"],
            "市盈率(PE,TTM)": ["20.5", "18.3"],
        })
        result = standardize_financials(df_raw, "600519")
        assert result is not None
        assert len(result) == 2
        assert list(result.columns) == ["date", "code", "revenue", "profit", "eps",
                                         "bps", "roe", "pe", "pb", "mcap",
                                         "total_assets", "debt_ratio",
                                         "gross_margin", "operate_cf"]
        # 营业收入 100亿 → 100 * 1e8
        assert result.iloc[0]["revenue"] == 100e8
        assert result.iloc[0]["pe"] == 20.5

    def test_standardize_bad_year_filtered(self):
        """年份超出 2010-2030 应被过滤"""
        df_raw = pd.DataFrame({
            "年度": ["9046-12-31", "2020-12-31"],
            "营业收入": ["x", "100亿"],
            "净利润": ["x", "10亿"],
        })
        result = standardize_financials(df_raw, "600519")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["revenue"] == 100e8

    def test_standardize_empty(self):
        result = standardize_financials(None, "600519")
        assert result is None

    def test_standardize_no_date_col(self):
        df_raw = pd.DataFrame({"foo": ["1"], "bar": ["2"]})
        result = standardize_financials(df_raw, "600519")
        assert result is None
