"""Run Web trainer.py with same params as the ngrok instance to compare results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Web"))

from backend.trainer import train, TrainParams
from backend import paths as _p

# Force v24 (same as remote, has fwd_1d_exec_ret)
_p.latest_training_data = lambda: _p.PROCESSED_DIR / "training_data_v24.parquet"
import backend.trainer as _t
_t._TRAIN_DATA_PATH = _p.PROCESSED_DIR / "training_data_v24.parquet"

print(f"Training data: {_t._TRAIN_DATA_PATH}")

params = TrainParams(
    train_start="2022-09-01",
    test_start="2025-06-01",
    test_end="2025-07-16",
    buy_pct=0.03,
    sell_pct=0.03,
    slip_pct=0.01,
    initial_capital=2_000_000,
    top_n=3,
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42,
    n_jobs=10,
    run_name="local_compare",
    skip_next_rec=True,
)

result = train(params)
print(f"\n{'='*60}")
print(f"  n_days={result.n_days}")
print(f"  ic_mean={result.ic_mean}")
print(f"  annual_return={result.annual_return}")
print(f"  sharpe_raw={result.sharpe_raw}")
print(f"  sharpe_sampled={result.sharpe_sampled}")
print(f"  max_dd={result.max_dd}")
print(f"  win_rate={result.win_rate}")
