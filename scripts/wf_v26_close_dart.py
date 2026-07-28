"""
Walk-Forward 回测 v26: 收盘价执行 + DART boosting
标签: fwd_1d_t1_close_ret = close_{t+2}/close_{t+1} - 1 (T+1收盘买, T+2收盘卖)
参数: n=151, d=4, lr=0.03, n_jobs=10, boosting_type=dart
执行: 收盘价, 整数手, 涨跌停限制, 最低5元佣金
"""
import pandas as pd, numpy as np, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb
from backtest import execution as ex

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(description="WF v26 close-price DART")
parser.add_argument("--test-start", type=str, default="2023-01-01")
parser.add_argument("--test-end", type=str, default="2026-07-16")
parser.add_argument("--initial-capital", type=float, default=100000.0)
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v26.parquet"

LABEL_RAW = "fwd_1d_t1_close_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
             "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret"}
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

_tag = "v26_close_dart"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ── Load ──
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

# ── Walk-Forward ──
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
    top3_ret = top3["label"].mean() - TRADE_COST
    top3_raw_ret = top3["raw_label"].mean() - TRADE_COST

    actual_topk = set(np.argsort(y_test)[-top_n:])
    pred_topk = set(np.argsort(preds)[-top_n:])
    hit = len(actual_topk & pred_topk) / top_n if top_n > 0 else 0

    rand_idx = rng.choice(n_test, size=top_n, replace=False)
    rand_ret = float(test_df.iloc[rand_idx]["label"].mean() - TRADE_COST)
    rand_raw_ret = float(test_df.iloc[rand_idx]["raw_label"].mean() - TRADE_COST)
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

# ── Summary ──
rdf = pd.DataFrame(daily_results)
rdf["ic"] = rdf["ic"].astype(float)
valid = rdf.dropna(subset=["ic"])
rdf["date"] = pd.to_datetime(rdf["date"])
rdf = rdf.sort_values("date").reset_index(drop=True)

# ── Execution simulation with close-price patch ──
def _load_klines_fast(data_dir, codes):
    cache = {}
    for code in codes:
        code6 = str(code)[:6]
        path = data_dir / "raw" / "kline" / f"{code6}.parquet"
        if not path.exists():
            continue
        kline = pd.read_parquet(path)
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        kline.attrs["date_to_pos"] = {pd.Timestamp(d): j for j, d in enumerate(kline["date"].values)}
        cache[code] = kline
    return cache

def _next_bar_fast(kline, signal_date):
    pos = kline.attrs["date_to_pos"].get(signal_date)
    if pos is None or pos + 1 >= len(kline):
        return None
    c = kline.iloc[pos + 1]
    p = kline.iloc[pos]
    return {"date": c["date"], "index": pos + 1, "previous_close": float(p["close"]),
            "open": float(c["open"]), "close": float(c["close"])}

ex._load_klines = _load_klines_fast
ex._next_bar = _next_bar_fast

# Patch simulate_portfolio to use close price
import inspect
src = inspect.getsource(ex.simulate_portfolio).replace('bar["open"]', 'bar["close"]')
exec(src, ex.__dict__)
simulate_portfolio = ex.simulate_portfolio

print("\n[exec] Simulating portfolio (close price, full constraints)...")
pf_portfolio, execution_stats = simulate_portfolio(rdf, DATA_DIR, INIT_CAPITAL, TRADE_COST, "holdings")
print(f"[exec] final={pf_portfolio['portfolio_value'].iloc[-1]:.2f}")

pf_rand, random_execution_stats = simulate_portfolio(rdf, DATA_DIR, INIT_CAPITAL, TRADE_COST, "rand_holdings")

# Stats
rand_sharpe = float(pf_rand["daily_ret"].mean() / pf_rand["daily_ret"].std() * np.sqrt(252)) if pf_rand["daily_ret"].std() > 0 else 0
rand_win = float((pf_rand["daily_ret"] > 0).mean())
rand_cum = float((1 + pf_rand["daily_ret"]).prod() - 1)
rand_ann = float((1 + pf_rand["daily_ret"]).prod() ** (252/len(pf_rand)) - 1) if len(pf_rand) > 0 else 0
rand_cum_series = (1 + pf_rand["daily_ret"]).cumprod()
rand_dd = float((rand_cum_series / rand_cum_series.expanding().max() - 1).min())

pf = pf_portfolio[["date", "daily_ret"]].copy()
pf = pf.rename(columns={"daily_ret": "top3_ret"})
pf["date"] = pd.to_datetime(pf["date"])
pf = pf.sort_values("date")
pf["cum_raw"] = (1 + pf["top3_ret"]).cumprod() - 1

pf_sampled = pf.iloc[::5]
sharpe = float(pf_sampled["top3_ret"].mean() / pf_sampled["top3_ret"].std() * np.sqrt(252/5)) if pf_sampled["top3_ret"].std() > 0 else 0
sharpe_raw = float(pf["top3_ret"].mean() / pf["top3_ret"].std() * np.sqrt(252)) if pf["top3_ret"].std() > 0 else 0
cum_series = (1 + pf["top3_ret"]).cumprod()
max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
win_rate = float((pf["top3_ret"] > 0).mean())
total_cost = execution_stats["total_cost_paid"] / INIT_CAPITAL
n_days = len(rdf)
ann_ret = float((1 + pf["top3_ret"]).prod() ** (252/n_days) - 1) if n_days > 0 else 0

# Print summary
print(f"\n{'='*60}")
print(f"  v26 Close-Price DART Results")
print(f"{'='*60}")
print(f"  Days: {len(rdf)}")
print(f"  IC: mean={valid['ic'].mean():.4f} std={valid['ic'].std():.4f}")
print(f"  Top3 excess: mean={rdf['top3_ret'].mean():+.6f}")
print(f"  Cum return (raw): {pf['cum_raw'].iloc[-1]*100:.1f}%")
print(f"  Portfolio value: ¥{pf_portfolio['portfolio_value'].iloc[-1]:,.0f} (from ¥{INIT_CAPITAL:,})")
print(f"  Sharpe: {sharpe:.2f} (raw={sharpe_raw:.2f})")
print(f"  Max DD: {max_dd*100:.1f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Annualized: {ann_ret*100:.1f}%")
print(f"  Total cost: {total_cost*100:.1f}% of capital")
print(f"  Rejected: buy={execution_stats['rejected_buy']} sell={execution_stats['rejected_sell']}")
print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

# Save
output = {
    "label": LABEL,
    "model": f"LightGBM DART n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
             f"lr={LOCKED_PARAMS['learning_rate']}, n_jobs={LOCKED_PARAMS['n_jobs']}, close-price execution",
    "features": len(features),
    "period": f"{rdf['date'].iloc[0].strftime('%Y-%m-%d')} ~ {rdf['date'].iloc[-1].strftime('%Y-%m-%d')}",
    "n_prediction_days": len(rdf),
    "initial_capital": INIT_CAPITAL,
    "summary": {
        "ic_mean": round(valid['ic'].mean(), 4),
        "ic_std": round(valid['ic'].std(), 4),
        "top3_excess_mean": round(float(rdf['top3_ret'].mean()), 6),
        "cum_return_pct": round(float(pf['cum_raw'].iloc[-1])*100, 1),
        "annualized_return_pct": round(ann_ret*100, 1),
        "sharpe": round(sharpe, 2),
        "sharpe_raw": round(sharpe_raw, 2),
        "max_dd_pct": round(max_dd*100, 1),
        "win_rate_pct": round(win_rate*100, 1),
        "hit_rate": round(float(rdf['hit_rate'].mean()), 3),
        "total_cost_est_pct": round(total_cost*100, 2),
        "execution_rejected_buy": execution_stats["rejected_buy"],
        "execution_rejected_sell": execution_stats["rejected_sell"],
        "execution_missing_bars": execution_stats["missing_bars"],
    },
    "daily": daily_results,
    "portfolio": {
        "initial_capital": INIT_CAPITAL,
        "final_value": round(float(pf_portfolio['portfolio_value'].iloc[-1]), 2),
        "total_return_pct": round((float(pf_portfolio['portfolio_value'].iloc[-1]) / INIT_CAPITAL - 1) * 100, 2),
        "avg_turnover_sold": round(float(pf_portfolio['turnover_sold'].mean()), 2),
        "avg_turnover_bought": round(float(pf_portfolio['turnover_bought'].mean()), 2),
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
        "sharpe": round(rand_sharpe, 2),
        "max_dd_pct": round(rand_dd*100, 1),
        "win_rate_pct": round(rand_win*100, 1),
        "cum_return_pct": round(rand_cum*100, 1),
        "annualized_return_pct": round(rand_ann*100, 1),
        "daily_values": [
            {"date": r["date"], "value": r["portfolio_value"], "daily_ret": r["daily_ret"],
             "rolling_100d_ret_pct": r["rolling_100d_ret_pct"]}
            for _, r in pf_rand.iterrows()
        ],
    },
    "random_execution": {
        "stats": {k: v for k, v in random_execution_stats.items() if k != "trade_log"},
        "trades": random_execution_stats["trade_log"],
    },
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved: {OUT_PATH}")
