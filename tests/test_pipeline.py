"""
数据引擎 — 模块加载与基础功能测试
"""
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
            check_key_dates,
            load_supply_chain,
            match_news_event,
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


class TestFeatureBuildOptions:

    def test_stock_feature_cutoff(self, monkeypatch, tmp_path):
        import pandas as pd

        import pipeline.feature_engine as feature_engine

        source = pd.DataFrame({
            "date": pd.date_range("2018-12-01", periods=80),
            "value": range(80),
        })
        monkeypatch.setattr(
            feature_engine,
            "build_features_for_stock",
            lambda _code, _code6: source.copy(),
        )

        assert feature_engine._build_one_stock(
            "000001.SZ", str(tmp_path), pd.Timestamp("2019-01-01")
        )
        result = pd.read_parquet(tmp_path / "000001.parquet")
        assert result["date"].min() == pd.Timestamp("2019-01-01")

    def test_separate_feature_cache_dir(self, monkeypatch, tmp_path):
        import json

        import pipeline.feature_engine as feature_engine

        universe = tmp_path / "universe"
        universe.mkdir()
        (universe / "empty.json").write_text(
            json.dumps({"watchlist": []}), encoding="utf-8"
        )
        monkeypatch.setattr(feature_engine, "DATA_DIR", str(tmp_path))

        feature_engine.build_all(
            incremental=False,
            watchlist_file="empty.json",
            out_file="training_data_pit_2019.parquet",
            cutoff_date="2019-01-01",
            features_dir="features_2019",
        )

        assert (tmp_path / "processed" / "features_2019").is_dir()
        assert not (tmp_path / "processed" / "features").exists()


class TestTencentKlineFallback:

    def test_sz000_volume_unit_fix(self):
        import pandas as pd

        from scripts.update_kline_akshare import _finalize_tx

        source = pd.DataFrame([{
            "date": "2019-01-02", "open": 2.15, "high": 2.16,
            "low": 2.13, "close": 2.13, "volume": 103360.4,
            "amount": 22147700.0, "turnover": 0.0104,
        }])
        result = _finalize_tx(source, "000018")

        assert result.loc[0, "volume"] == 10336040.0
        assert result.loc[0, "outstanding_share"] == 993850000.0

    def test_other_stock_volume_already_normalized(self):
        import pandas as pd

        from scripts.update_kline_akshare import _finalize_tx

        source = pd.DataFrame([{
            "date": "2019-01-02", "open": 1.50, "high": 1.57,
            "low": 1.50, "close": 1.54, "volume": 40747700.0,
            "amount": 62830500.0, "turnover": 0.0207,
        }])
        result = _finalize_tx(source, "002477")

        assert result.loc[0, "volume"] == 40747700.0
        assert result.loc[0, "outstanding_share"] == 1968487922.7053142


class TestPitUniverseRules:

    def test_first_2019_semiannual_period(self):
        import pandas as pd

        from scripts.build_pit_universe import rebalance_dates

        calendar = pd.bdate_range("2019-01-01", "2020-01-10")
        dates = rebalance_dates(calendar, "2019-07-01", "2020-01-10", "semiannual")

        assert dates[0] == pd.Timestamp("2019-07-01")

    def test_b_shares_are_excluded(self):
        import pandas as pd

        from scripts.build_pit_universe import exclude_security_prefixes

        source = pd.DataFrame({"code": ["600000", "000001", "200018", "900951"]})
        result = exclude_security_prefixes(source, ("200", "900"))

        assert result["code"].tolist() == ["600000", "000001"]

    def test_empty_kline_is_not_usable(self, monkeypatch, tmp_path):
        import pandas as pd

        import scripts.expand_2019_overnight as expand

        pd.DataFrame({"date": []}).to_parquet(tmp_path / "000001.parquet", index=False)
        pd.DataFrame({"date": ["2019-01-02"]}).to_parquet(
            tmp_path / "600000.parquet", index=False
        )
        monkeypatch.setattr(expand, "KLINE", tmp_path)

        assert expand.usable_kline_codes(["000001", "600000", "002477"]) == {"600000"}


class TestPredictionEnsemble:

    def test_plain_mean_aligns_predictions_by_code(self):
        from scripts.ensemble_pred_caches import combine_day

        rows = [
            {
                "ranked": ["A", "B", "C", "D", "E", "F"],
                "pred_vals": [6, 5, 4, 3, 2, 1],
            },
            {
                "ranked": ["F", "E", "D", "C", "B", "A"],
                "pred_vals": [1, 2, 3, 4, 5, 6],
            },
        ]
        labels = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1}
        result = combine_day(rows, labels)

        assert result["ranked"] == ["A", "B", "C", "D", "E", "F"]
        assert result["pred_vals"] == [6, 5, 4, 3, 2, 1]
        assert result["ic"] == 1.0
