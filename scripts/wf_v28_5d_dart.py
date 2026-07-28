"""
Walk-Forward 回测 v28: fwd_5d_ret (5日收盘→收盘) + DART boosting
标签: fwd_5d_ret = close_{T+5} / close_T - 1 (T收盘买, 持仓5天, T+5收盘卖)
参数: n=151, d=4, lr=0.03, n_jobs=10, boosting_type=dart
执行: 收盘价, 整数手, 涨跌停限制, 最低5元佣金, 持仓5天换仓
"""
import pandas as pd, numpy as np, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb
from backtest import execution as ex

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(description="WF v28 5d close-to-close DART")
parser.add_argument("--test-start", type=str, default="2023-01-01")
parser.add_argument("--test-end", type=str, default="2026-07-16")
parser.add_argument("--initial-capital", type=float, default=100000.0)
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v24.parquet"

LABEL_RAW = "fwd_5d_ret"
LABEL = "fwd_5d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
             "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret", "fwd_5d_excess"}
EXCLUDED_FEATS = {"mf_pct_1d", "mf_pct_1d_ma5", "mf_pct_1d_ma20",
                  "macd_signal", "macd_signal_ma5", "macd_signal_ma20"}

LOCKED_PARAMS = dict(
    n_estimators=151, max_depth=4, learning_rate=0.03,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=50, random_state=42, n_jobs=10, verbosity=-1,
    boosting_type="dart",
)

TRADE_COST = 0.0006
INIT_CAPITAL = args.initial_capital
TEST_START = args.test_start
TEST_END = args.test_end
HOLD_DAYS = 5

_tag = "v28_5d_dart"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

print(f"Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[LABEL_RAW])
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

all_cols = [c for c in df.columns if c not in SKIP_COLS and c not in EXCLUDED_FEATS and is_valid_feat(c)]
features = [f for f in all_cols if f not in LEAKAGE_FEATS]
print(f"  {len(df)} rows, {df['code'].nunique()} codes, {len(features)} features")
print(f"  Dates: {df['date'].min().date()} ~ {df['date'].max().date()}")

_dmask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END is not None:
    _dmask &= df["date"] <= pd.Timestamp(TEST_END)
dates = sorted(df[_dmask]["date"].unique())
if not dates:
    print(f"ERROR: No data after {TEST_START}")
    exit(1)

MIN_TRAIN_DAYS = 250
print(f"\nWalk-Forward (daily expanding): {len(dates)} prediction days ({dates[0].date()} ~ {dates[-1].date()})")
print(f"Model: DART n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
      f"lr={LOCKED_PARAMS['learning_rate']} n_jobs={LOCKED_PARAMS['n_jobs']}")
print(f"Label: {LABEL_RAW} (close_T → close_T+5, hold {HOLD_DAYS} days)")

daily_results = []
t0 = datetime.now()

for day_idx, pred_date in enumerate(dates):
    train_mask = df["date"] < pred_date
    train_df = df[train_mask]
    if train_df["date"].nunique() < MIN_TRAIN_DAYS:
        continue

    X_train = train_df.groupby("code")[features].transform(lambda s: s.ffill().fillna(0))
    y_train = train_df[LABEL].copy()

    model = lgb.LGBMRegressor(**LOCKED_PARAMS)
    model.fit(X_train, y_train)

    test_mask = df["date"] == pred_date
    X_test = df.loc[test_mask, features].copy()
    y_test = df.loc[test_mask, LABEL].copy()
    codes_test = df.loc[test_mask, "code"].values

    for c in features:
        if X_test[c].isna().any():
            X_test[c] = X_test[c].fillna(0)

    preds = model.predict(X_test)

    if len(preds) > 5:
        ic, _ = spearmanr(preds, y_test)
    else:
        ic = np.nan

    test_df = pd.DataFrame({"code": codes_test, "pred": preds, "label": y_test.values,
                            "raw_label": df.loc[test_mask, LABEL_RAW].values})
    n_test = len(test_df)
    top_n = min(3, n_test)
    top3 = test_df.nlargest(top_n, "pred")
    # 5天只扣一次买卖成本（不是每天）
    top3_ret = top3["label"].mean() - TRADE_COST * 2
    top3_raw_ret = top3["raw_label"].mean() - TRADE_COST * 2

    actual_topk = set(np.argsort(y_test)[-top_n:])
    pred_topk = set(np.argsort(preds)[-top_n:])
    hit = len(actual_topk & pred_topk) / top_n if top_n > 0 else 0

    rand_idx = rng.choice(n_test, size=top_n, replace=False)
    rand_ret = float(test_df.iloc[rand_idx]["label"].mean() - TRADE_COST * 2)
    rand_raw_ret = float(test_df.iloc[rand_idx]["raw_label"].mean() - TRADE_COST * 2)
    rand_holdings = list(test_df.iloc[rand_idx]["code"].values)

    daily_results.append({
        "date": str(pred_date.date()),
        "n_train": len(X_train),
        "n_test": n_test,
        "ic": round(ic, 4) if not np.isnan(ic) else None,
        "top3_ret": round(top3_ret, 6),
        "top3_raw_ret": round(top3_raw_ret, 6),
        "rand_ret": round(rand_ret, 6),
        "rand_raw_ret": round(rand_raw_ret, 6),
        "hit_rate": round(hit, 3),
        "long_short": round(preds.mean(), 6) if len(preds) > 0 else 0,
        "holdings": list(top3["code"].values),
        "rand_holdings": rand_holdings,
    })

    if day_idx % 20 == 0 or day_idx == len(dates) - 1:
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  [{day_idx+1}/{len(dates)}] {pred_date.date()} "
              f"IC={ic:.4f} top3={top3_ret:+.4f} train={len(X_train):,d} "
              f"({elapsed:.0f}s)")

rdf = pd.DataFrame(daily_results)
rdf["ic"] = rdf["ic"].astype(float)
valid = rdf.dropna(subset=["ic"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf = rdf.sort_values("date").reset_index(drop=True)

# Execution simulation with close-price, hold 5 days
def _load_klines_fast(data_dir, codes):
    cache = {}
    for code in codes:
        code6 = str(code)[:6]
        path = data_dir / "raw" / "kline" / f"{code6}.parquet"
        if not path.exists(): continue
        kline = pd.read_parquet(path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        kline.attrs["date_to_pos"] = {pd.Timestamp(d): j for j, d in enumerate(kline["date"].values)}
        cache[code] = kline
    return cache

def _next_bar_fast(kline, signal_date):
    pos = kline.attrs["date_to_pos"].get(signal_date)
    if pos is None or pos + 1 >= len(kline): return None
    c = kline.iloc[pos + 1]; p = kline.iloc[pos]
    return {"date": c["date"], "index": pos + 1, "previous_close": float(p["close"]),
            "open": float(c["open"]), "close": float(c["close"])}

ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast

# Patch simulate_portfolio: close price + hold 5 days (only buy when no positions)
import inspect
src = inspect.getsource(ex.simulate_portfolio)
src = src.replace('bar["open"]', 'bar["close"]')
# Change holding period: only sell after 5 bars instead of 1
# The original logic sells all positions and buys new ones every day
# We need to: skip selling/buying for 4 days, only act every 5th day
# Simplest approach: filter dates to every 5th day
# Actually, let's just use every 5th prediction day for execution
rdf_exec = rdf.iloc[::HOLD_DAYS].copy()
print(f"\n[exec] Execution every {HOLD_DAYS} days: {len(rdf_exec)} rebalance days")

exec(src, ex.__dict__)
simulate_portfolio = ex.simulate_portfolio

print("[exec] Simulating portfolio (close price, full constraints, 5-day hold)...")
pf_portfolio, execution_stats = simulate_portfolio(rdf_exec, DATA_DIR, INIT_CAPITAL, TRADE_COST, "holdings")
print(f"[exec] final={pf_portfolio['portfolio_value'].iloc[-1]:.2f}")

pf_rand, random_execution_stats = simulate_portfolio(rdf_exec, DATA_DIR, INIT_CAPITAL, TRADE_COST, "rand_holdings")

# Theoretical returns (every 5 days)
pf_theo = rdf_exec[["date", "top3_ret"]].copy()
pf_theo["cum_raw"] = (1 + pf_theo["top3_ret"]).cumprod() - 1

sharpe_raw = float(pf_theo["top3_ret"].mean() / pf_theo["top3_ret"].std() * np.sqrt(252/HOLD_DAYS)) if pf_theo["top3_ret"].std() > 0 else 0
cum_series = (1 + pf_theo["top3_ret"]).cumprod()
max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
win_rate = float((pf_theo["top3_ret"] > 0).mean())
n_rebal = len(rdf_exec)
ann_ret = float((1 + pf_theo["top3_ret"]).prod() ** (252/(n_rebal*HOLD_DAYS)) - 1) if n_rebal > 0 else 0
total_cost = execution_stats["total_cost_paid"] / INIT_CAPITAL

print(f"\n{'='*60}")
print(f"  v28 5-Day Hold DART Results")
print(f"{'='*60}")
print(f"  Prediction days: {len(rdf)}, Rebalance days: {n_rebal}")
print(f"  IC: mean={valid['ic'].mean():.4f} std={valid['ic'].std():.4f}")
print(f"  Top3 excess (5d): mean={rdf_exec['top3_ret'].mean():+.6f}")
print(f"  Cum return (theoretical): {pf_theo['cum_raw'].iloc[-1]*100:.1f}%")
print(f"  Portfolio value: ¥{pf_portfolio['portfolio_value'].iloc[-1]:,.0f} (from ¥{INIT_CAPITAL:,})")
print(f"  Sharpe: {sharpe_raw:.2f}")
print(f"  Max DD: {max_dd*100:.1f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Annualized: {ann_ret*100:.1f}%")
print(f"  Total cost: {total_cost*100:.1f}% of capital")
print(f"  Rejected: buy={execution_stats['rejected_buy']} sell={execution_stats['rejected_sell']}")
print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

output = {
    "label": LABEL_RAW,
    "model": f"LightGBM DART n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
             f"lr={LOCKED_PARAMS['learning_rate']}, n_jobs={LOCKED_PARAMS['n_jobs']}, 5-day hold",
    "features": len(features),
    "period": f"{rdf['date'].iloc[0].strftime('%Y-%m-%d')} ~ {rdf['date'].iloc[-1].strftime('%Y-%m-%d')}",
    "n_prediction_days": len(rdf),
    "n_rebalance_days": n_rebal,
    "hold_days": HOLD_DAYS,
    "initial_capital": INIT_CAPITAL,
    "summary": {
        "ic_mean": round(valid['ic'].mean(), 4),
        "ic_std": round(valid['ic'].std(), 4),
        "top3_excess_mean": round(float(rdf_exec['top3_ret'].mean()), 6),
        "cum_return_pct": round(float(pf_theo['cum_raw'].iloc[-1])*100, 1),
        "annualized_return_pct": round(ann_ret*100, 1),
        "sharpe": round(sharpe_raw, 2),
        "max_dd_pct": round(max_dd*100, 1),
        "win_rate_pct": round(win_rate*100, 1),
        "hit_rate": round(float(rdf['hit_rate'].mean()), 3),
        "total_cost_est_pct": round(total_cost*100, 2),
        "execution_rejected_buy": execution_stats["rejected_buy"],
        "execution_rejected_sell": execution_stats["rejected_sell"],
    },
    "daily": daily_results,
    "portfolio": {
        "initial_capital": INIT_CAPITAL,
        "final_value": round(float(pf_portfolio['portfolio_value'].iloc[-1]), 2),
        "total_return_pct": round((float(pf_portfolio['portfolio_value'].iloc[-1]) / INIT_CAPITAL - 1) * 100, 2),
        "daily_values": [
            {"date": r["date"], "value": r["portfolio_value"], "daily_ret": r["daily_ret"],
             "rolling_100d_ret_pct": r["rolling_100d_ret_pct"],
             "turnover_bought": r["turnover_bought"], "turnover_sold": r["turnover_sold"],
             "buy_cost": r["buy_cost"], "sell_cost": r["sell_cost"], "total_cost": r["total_cost"]}
            for _, r in pf_portfolio.iterrows()
        ],
    },
    "execution": {
        "stats": {k: v for k, v in execution_stats.items() if k != "trade_log"},
        "trades": execution_stats["trade_log"],
    },
    "random_baseline": {
        "final_value": round(float(pf_rand['portfolio_value'].iloc[-1]), 2),
        "total_return_pct": round((float(pf_rand['portfolio_value'].iloc[-1]) / INIT_CAPITAL - 1) * 100, 2),
    },
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved: {OUT_PATH}")
