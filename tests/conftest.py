"""
pytest 共享 fixtures

用法:
  pytest tests/ -v
"""
import sys

import pytest
from pathlib import Path
from pipeline.config import settings


@pytest.fixture(autouse=True)
def _isolate_access_log(tmp_path, monkeypatch):
    """测试里的 TestClient 请求会经过网页中间件记访问日志; 不隔离就会写进生产的
    data/live/access_log.jsonl (10-18 在 041 上发现一批 ip=testclient 的记录)。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import access_log
    except ImportError:
        return
    monkeypatch.setattr(access_log, "LOG_PATH", tmp_path / "access_log.jsonl")
    monkeypatch.setattr(access_log, "STATS_PATH", tmp_path / "access_stats.json")
    if hasattr(access_log, "_stats_cache"):
        monkeypatch.setattr(access_log, "_stats_cache", None)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return settings.PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return settings.DATA_DIR


@pytest.fixture(scope="session")
def kline_dir() -> Path:
    return settings.KLINE_DIR


@pytest.fixture(scope="session")
def fund_flow_dir() -> Path:
    return settings.FUND_FLOW_DIR


@pytest.fixture(scope="session")
def universe_dir() -> Path:
    return settings.UNIVERSE_DIR


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return settings.PROCESSED_DIR


@pytest.fixture(scope="session")
def supply_chain_path() -> Path:
    return settings.SUPPLY_CHAIN_PATH


@pytest.fixture(scope="session")
def watchlist_path() -> Path:
    return settings.WATCHLIST_PATH


@pytest.fixture(scope="session")
def backtest_dir() -> Path:
    return settings.BACKTEST_DIR
