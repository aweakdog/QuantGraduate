"""
pytest 共享 fixtures

用法:
  pytest tests/ -v
"""
import pytest
from pathlib import Path
from pipeline.config import settings


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
