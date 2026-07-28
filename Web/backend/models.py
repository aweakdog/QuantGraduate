"""
Pydantic models for type-safe data transfer in the quant-strategy Web backend.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    """Represents a single training run stored in the `runs` table."""
    id: int
    name: str
    created_at: datetime
    train_start: date
    test_start: date
    test_end: Optional[str] = None
    universe_source: str = "关注圈"
    buy_pct: float = Field(ge=0, description="Buy cost in percent")
    sell_pct: float = Field(ge=0, description="Sell cost in percent")
    slip_pct: float = Field(ge=0, description="Slippage in percent")
    top_n: int = Field(ge=1)
    n_features: int = Field(ge=0)
    n_days: int = Field(ge=0)
    min_train_days: int = 250
    sample_interval: int = 5
    sharpe_raw: Optional[float] = None
    sharpe_sampled: Optional[float] = None
    max_dd: Optional[float] = None
    win_rate: Optional[float] = None
    ic_mean: Optional[float] = None
    annual_return: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"  # completed | failed | running


class DailyReturn(BaseModel):
    """One row in the `daily_returns` table."""
    id: int
    run_id: int
    date: date
    top_ret: float
    ic: float
    cum_return: Optional[float] = None


class FeatureImportance(BaseModel):
    """One row in the `feature_importance` table."""
    id: int
    run_id: int
    feature: str
    gain: float


class RunCreate(BaseModel):
    """Input model for creating a new run record (no id / created_at)."""
    name: str
    train_start: date
    test_start: date
    test_end: Optional[str] = None
    universe_source: str = "关注圈"
    buy_pct: float = Field(ge=0)
    sell_pct: float = Field(ge=0)
    slip_pct: float = Field(ge=0)
    top_n: int = Field(ge=1)
    n_features: int = Field(ge=0)
    n_days: int = Field(ge=0)
    min_train_days: int = 250
    sample_interval: int = 5
    model_params: dict[str, Any] = Field(default_factory=dict)


class TrainParams(BaseModel):
    """Training parameters passed to trainer.train()."""
    # Date range
    train_start: date
    test_start: date
    test_end: Optional[date] = None   # 回测终点(含); None=最新日
    # Cost parameters (in %)
    buy_pct: float = 0.03
    sell_pct: float = 0.03
    slip_pct: float = 0.01
    initial_capital: float = 2_000_000
    top_n: int = 3
    # 训练集来源: 关注圈(默认池) / 自选股(self_selected.json) / 自定义(自定义代码)
    universe_source: str = "关注圈"
    custom_codes: list[str] = Field(default_factory=list)
    # 回测 / 采样旋钮 (全部可设)
    min_train_days: int = 250    # 最短训练天数(窗口内不足则跳过该日)
    sample_interval: int = 5     # 非重叠 Sharpe 采样间隔(日)
    skip_next_rec: bool = False  # 跳过 NEXT_REC 补算 (消融实验加速)
    # Model hyper-parameters
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.03
    num_leaves: int = 15
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    random_state: int = 42
    n_jobs: int = 32
    # 自定义特征集: None=用默认过滤(排除标签/leak/_21d/_cross); 传 list 则优先使用
    features: Optional[list[str]] = None
    # Run metadata
    run_name: Optional[str] = None


class TrainResult(BaseModel):
    """Complete result returned by trainer.train()."""
    run_id: int
    name: str
    n_days: int
    n_features: int
    sharpe_raw: float
    sharpe_sampled: float
    max_dd: float
    win_rate: float
    ic_mean: float
    annual_return: float
    elapsed_seconds: float
    daily_returns: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    model_params: dict[str, Any]
    date_range: list[str] = []  # 固定 x 轴范围 [训练起始, 最后交易日] (含 warm-up 区间)
    # ── INFO_LABEL 渲染所需元信息 (TrainParams 透传) ──
    top_n: int = 3
    train_start: str = ""
    test_start: str = ""
    test_end: Optional[str] = None
    buy_pct: float = 0.0
    sell_pct: float = 0.0
    slip_pct: float = 0.0
    initial_capital: float = 2_000_000
