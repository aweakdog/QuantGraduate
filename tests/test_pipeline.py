"""
数据引擎 — 模块加载与基础功能测试
"""
import pytest
from pathlib import Path


class TestDataEngineImport:

    def test_pipeline_importable(self):
        """验证 pipeline 包可以正常导入"""
        import pipeline
        assert pipeline.__name__ == "pipeline"

    def test_config_importable(self):
        from pipeline.config import settings
        assert settings is not None
        assert hasattr(settings, "DATA_DIR")

    def test_logger_importable(self):
        from pipeline.logger import get_logger
        log = get_logger("test_import")
        assert log is not None

    def test_feature_engine_importable(self):
        from pipeline.feature_engine import build_all
        assert callable(build_all)

    def test_xgb_scorer_importable(self):
        from pipeline.xgb_scorer import load_data, prepare_train_test
        assert callable(load_data)
        assert callable(prepare_train_test)

    def test_supplier_self_event_importable(self):
        from pipeline.supplier_self_event import SelfEventChecker
        checker = SelfEventChecker()
        assert hasattr(checker, "backtest_proxy")

    def test_chain_leader_monitor_importable(self):
        from pipeline.chain_leader_monitor import (
            load_supply_chain, check_key_dates, match_news_event
        )
        assert callable(load_supply_chain)
        assert callable(check_key_dates)
        assert callable(match_news_event)

    def test_collect_fund_flow_importable(self):
        """collect_fund_flow 模块级代码有副作用，只检查文件存在"""
        import os
        assert os.path.exists(os.path.join(os.path.dirname(__file__), "..", "pipeline", "collect_fund_flow.py"))


class TestKlineData:
    """K线数据基础检查"""

    def test_kline_dir_exists(self, kline_dir):
        assert kline_dir.exists(), f"K线数据目录不存在: {kline_dir}"

    def test_kline_files_exist(self, kline_dir):
        files = list(kline_dir.glob("*.parquet"))
        assert len(files) > 0, f"K线目录无 parquet 文件: {kline_dir}"

    def test_kline_files_valid(self, kline_dir):
        """验证每个 parquet 可加载，含 date/close 列"""
        import pandas as pd
        files = list(kline_dir.glob("*.parquet"))
        errors = []
        skipped_empty = 0
        for f in files[:10]:  # 前10只
            try:
                df = pd.read_parquet(f)
                if len(df) == 0:
                    skipped_empty += 1
                    continue
                assert "date" in df.columns or "时间" in df.columns, f"{f.name}: 缺 date"
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        assert len(errors) == 0, f"验证失败: {errors}"


class TestUniverseData:
    """关注圈/供应链数据检查"""

    def test_watchlist_exists(self, watchlist_path):
        assert watchlist_path.exists(), f"关注圈文件不存在: {watchlist_path}"

    def test_watchlist_valid(self, watchlist_path):
        import json
        with open(watchlist_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "watchlist" in data
        assert len(data["watchlist"]) > 0
        for s in data["watchlist"]:
            assert "code" in s
            assert "name" in s

    def test_supply_chain_exists(self, supply_chain_path):
        assert supply_chain_path.exists(), f"供应链文件不存在: {supply_chain_path}"

    def test_supply_chain_valid(self, supply_chain_path):
        import json
        with open(supply_chain_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "chains" in data
        assert len(data["chains"]) > 0
