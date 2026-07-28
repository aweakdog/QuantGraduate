"""
测试配置中心 — 验证 settings 所有属性正确解析
"""
from pathlib import Path
import pytest
from pipeline.config import settings


@pytest.fixture
def cfg():
    return settings


class TestSettingsPaths:
    """settings 路径解析"""

    def test_project_root_exists(self, project_root):
        assert project_root.exists()
        assert (project_root / "pipeline").exists()
        assert (project_root / "pyproject.toml").exists()

    def test_data_dir_exists(self, data_dir):
        assert data_dir.exists() or data_dir.is_symlink()
        assert (data_dir / "universe").exists()

    def test_kline_dir(self, kline_dir):
        cfg = settings
        assert kline_dir == cfg.DATA_DIR / "raw" / "kline"

    def test_fund_flow_dir(self, fund_flow_dir):
        assert "fund_flow" in str(fund_flow_dir)

    def test_supply_chain_path(self, supply_chain_path):
        cfg = settings
        assert supply_chain_path == cfg.UNIVERSE_DIR / "supply_chain_map.json"

    def test_watchlist_path(self, watchlist_path):
        cfg = settings
        assert watchlist_path == cfg.UNIVERSE_DIR / "watchlist.json"

    def test_backtest_dir(self, backtest_dir, project_root):
        assert backtest_dir == project_root / "backtest"

    def test_scores_path(self):
        assert settings.SCORES_PATH == settings.PROCESSED_DIR / "daily_scores.json"

    def test_training_data_path(self):
        assert settings.TRAINING_DATA_PATH == settings.PROCESSED_DIR / "training_data.parquet"

    def test_model_path(self):
        assert settings.MODEL_PATH == settings.MODEL_DIR / "xgb_model.pkl"


class TestSettingsEnv:
    """环境变量覆盖"""

    def test_mode_default(self, cfg):
        assert cfg.MODE in ("live", "backtest")

    def test_ths_python_path(self, cfg):
        py = cfg.THS_PYTHON
        assert "python.exe" in py
        assert "envs" in py


class TestSettingsCredentials:
    """凭据（仅验证结构，不验证实际值）"""

    def test_ths_username(self, cfg):
        user = cfg.THS_USERNAME
        assert isinstance(user, str) and len(user) > 0

    def test_ths_password(self, cfg):
        pwd = cfg.THS_PASSWORD
        assert isinstance(pwd, str)


class TestSettingsValidate:
    """配置启动校验"""

    def test_validate_returns_list(self, cfg):
        warnings = cfg.validate()
        assert isinstance(warnings, list)
        assert cfg._validated

    def test_validate_lists_missing_optional_dirs(self, cfg):
        """缺失的可选路径应出现在警告中但不会断言失败"""
        warnings = cfg.validate()
        # 至少有关注圈文件校验
        wc_warnings = [w for w in warnings if "关注圈" in w]
        # 如果关注圈确实不存在则应出现在列表中
        if not cfg.WATCHLIST_PATH.exists():
            assert len(wc_warnings) >= 1

    def test_validate_mode_check(self, cfg):
        import os
        original = os.environ.get("QUANT_MODE")
        try:
            os.environ["QUANT_MODE"] = "invalid_mode"
            # 强制重新加载 settings（单例需重新创建）
            import importlib
            mod = importlib.import_module("pipeline.config")
            importlib.reload(mod)
            from pipeline.config import settings as reloaded
            warns = reloaded.validate()
            mode_warns = [w for w in warns if "MODE" in w]
            assert len(mode_warns) >= 1
        finally:
            if original is not None:
                os.environ["QUANT_MODE"] = original
            else:
                os.environ.pop("QUANT_MODE", None)
