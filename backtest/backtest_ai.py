"""
AI 主线专属回测 (LightGBM Regression, 每日扩展重训)
- 数据: training_data_theme_AI.parquet (33 stocks, 2022-2026)
- 模型: n=100 d=4 lr=0.03, 64线程CPU
- 交易成本: 0.06% per sell
- 输出: 累积收益, Sharpe, 最大回撤, 日收益序列
"""
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import warnings
from scipy.stats import spearmanr
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
THEME = "AI"
TRAIN_PATH = DATA_DIR / "processed" / f"training_data_theme_{THEME}.parquet"
OUT_DIR = DATA_DIR / "backtest"

LABEL_RAW = "fwd_1d_ret"

# --- Feature exclusion rules ---
SKIP_COLS = {"date", "code",
             "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_2d_excess", "fwd_5d_excess", "fwd_21d_excess"}
LEAKAGE_FEATS = ["ret_1d", "ret_2d", "ret_5d", "ret_21d"]

def is_valid_feat(name):
    """Exclude _21d, _cross, _ma2 features"""
    if "_21d" in name:
        return False
    if name.endswith("_cross"):
        return False
    if name.endswith("_ma2"):
        return False
    return True

# --- Load ---
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)

print(f"  Shape: {df.shape}")
print(f"  Date: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  Stocks: {df['code'].nunique()}")
print(f"  Stocks list: {sorted(df['code'].unique())}")

# Clean infs
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

# Drop rows missing label
df = df.dropna(subset=[LABEL_RAW])
print(f"  After dropna(label): {len(df)} rows")

# --- Build features ---
ALL_COLS = set(df.columns)
feature_cols = sorted([
    c for c in ALL_COLS
    if c not in SKIP_COLS
    and c not in LEAKAGE_FEATS
    and is_valid_feat(c)
])
print(f"  Features selected: {len(feature_cols)}")

# --- Daily Expanding Retrain ---
unique_dates = sorted(df["date"].unique())
print(f"  Total trading days: {len(unique_dates)}")

# Need minimum training days. Start prediction after 250 trading days (~1 year)
MIN_TRAIN_DAYS = 250
MIN_TRAIN_ROWS = 250 * 10  # ~2500 rows minimum

predict_dates = unique_dates[MIN_TRAIN_DAYS:]

print(f"  Prediction period: {predict_dates[0].date()} ~ {predict_dates[-1].date()}")
print(f"  Prediction days: {len(predict_dates)}")

# --- Storage ---
all_predictions = []  # each: {date, code, pred, actual_excess}

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting daily expanding retrain...")
print(f"  Config: LightGBM n_estimators=100 max_depth=4 lr=0.03")

for i, target_date in enumerate(predict_dates):
    # Training data: all data up to day before target
    train_mask = df["date"] < target_date
    test_mask = df["date"] == target_date
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    if len(df_train) < MIN_TRAIN_ROWS:
        continue
    if len(df_test) < 3:  # need at least 3 stocks to compare
        continue
    
    # --- Compute cross-sectional excess (both train and test) ---
    # Use train data mean for demeaning on test? No, excess is relative to TODAY's cross-section
    cs_mean_train = df_train.groupby("date")[LABEL_RAW].transform("mean")
    cs_mean_test = df_test[LABEL_RAW].mean()
    
    y_train = df_train[LABEL_RAW] - cs_mean_train
    y_test = df_test[LABEL_RAW] - cs_mean_test
    
    # --- Features ---
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    # Fill NaN with per-column median from train
    for c in feature_cols:
        if X_train[c].isna().any():
            med = X_train[c].median()
            fill_val = med if pd.notna(med) else 0.0
            X_train[c] = X_train[c].fillna(fill_val)
            X_test[c] = X_test[c].fillna(fill_val)
    
    # Additional NaN check: replace remaining NaN with 0
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)
    
    # --- Train ---
    try:
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.03,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42,
            n_jobs=64, verbosity=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    except Exception as e:
        if i < 5:
            print(f"  [Warning] Training failed for {target_date.date()}: {e}")
        continue
    
    # Store predictions
    for j in range(len(df_test)):
        all_predictions.append({
            "date": target_date,
            "code": df_test.iloc[j]["code"],
            "pred": float(preds[j]),
            "actual_ret": float(df_test.iloc[j][LABEL_RAW]),
            "actual_excess": float(y_test.iloc[j])
        })
    
    if (i + 1) % 100 == 0:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Progress: {i+1}/{len(predict_dates)} days")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training complete.")
print(f"  Total predictions: {len(all_predictions)}")

# --- Portfolio Simulation ---
pred_df = pd.DataFrame(all_predictions)
pred_df["date"] = pd.to_datetime(pred_df["date"])

# Sort by date for portfolio construction
pred_df = pred_df.sort_values(["date", "pred"], ascending=[True, False])

# === Portfolio Construction ===
print("\n" + "=" * 70)
print("PORTFOLIO SIMULATION")
print("=" * 70)

def simulate_portfolio(pred_df, top_n, cost_per_trade=0.0006, name=""):
    """
    每日选Top-N等权持有T+1天，含交易成本(0.06% sell cost)
    cost_per_trade: 每笔交易的费率（卖出时扣除）
    """
    daily_rets = []
    prev_holdings = set()
    
    all_dates = sorted(pred_df["date"].unique())
    
    for d in all_dates:
        day_data = pred_df[pred_df["date"] == d].copy()
        
        if len(day_data) < top_n:
            # Not enough stocks available
            continue
        
        day_data = day_data.nlargest(top_n, "pred")
        holdings = set(day_data["code"].values)
        
        # Calculate portfolio return: equal weight average of actual returns
        pf_ret = day_data["actual_ret"].mean()
        
        # Transaction cost: sell all stocks not held previously (turnover)
        sold = prev_holdings - holdings
        if prev_holdings:
            # Cost is cost_per_trade * fraction of stocks sold
            turnover = len(sold) / max(len(prev_holdings), top_n)
            cost = turnover * cost_per_trade
        else:
            cost = 0
        
        net_ret = pf_ret - cost
        
        daily_rets.append({
            "date": d,
            "ret": pf_ret,
            "cost": cost,
            "net_ret": net_ret,
            "n_holdings": len(holdings),
            "holdings": ",".join(sorted(holdings)),
            "turnover_sold": len(sold),
        })
        
        prev_holdings = holdings
    
    pf = pd.DataFrame(daily_rets).sort_values("date")
    
    if len(pf) == 0:
        return {"cum_ret": 0, "annual_ret": 0, "sharpe": 0, "max_dd": 0, "pf_df": pf}
    
    # Cumulative return
    pf["cum"] = (1 + pf["net_ret"]).cumprod()
    cum_ret = (pf["cum"].iloc[-1] - 1) * 100
    
    # Annualized return
    n_years = (pf["date"].max() - pf["date"].min()).days / 365.25
    annual_ret = (pf["cum"].iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    
    # Sharpe (non-overlapping 5-day sampling)
    pf_sampled = pf.iloc[::5]
    if len(pf_sampled) > 5 and pf_sampled["net_ret"].std() > 0:
        sharpe = pf_sampled["net_ret"].mean() / pf_sampled["net_ret"].std() * np.sqrt(252/5)
    else:
        sharpe = 0
    
    # Max drawdown
    cum_max = pf["cum"].cummax()
    drawdown = (pf["cum"] - cum_max) / cum_max * 100
    max_dd = drawdown.min()
    
    # Hit rate: predicted top N vs actual top N
    hit_count = 0
    total_tries = 0
    for d in all_dates:
        day_data = pred_df[pred_df["date"] == d]
        if len(day_data) < top_n * 2:
            continue
        pred_top = set(day_data.nlargest(top_n, "pred")["code"])
        actual_top = set(day_data.nlargest(top_n, "actual_ret")["code"])
        hit_count += len(pred_top & actual_top)
        total_tries += top_n
    hit_rate = hit_count / total_tries if total_tries > 0 else 0
    
    # IC
    ic_vals = []
    for d in all_dates:
        day_data = pred_df[pred_df["date"] == d]
        if len(day_data) < 5:
            continue
        ic, _ = spearmanr(day_data["pred"], day_data["actual_ret"])
        ic_vals.append(ic)
    mean_ic = np.mean(ic_vals) if ic_vals else 0
    ir = mean_ic / np.std(ic_vals) if len(ic_vals) > 1 and np.std(ic_vals) > 0 else 0
    
    # Average daily turnover
    avg_turnover = pf["turnover_sold"].mean() / top_n * 100 if len(pf) > 0 else 0
    total_cost_bps = pf["cost"].sum() * 10000  # in bps
    
    print(f"\n  [{name}] Top{top_n} Portfolio:")
    print(f"    累计收益: {cum_ret:.1f}%")
    print(f"    年化收益: {annual_ret:.1f}%")
    print(f"    Sharpe (5d采样): {sharpe:.2f}")
    print(f"    最大回撤: {max_dd:.1f}%")
    print(f"    日均换手率: {avg_turnover:.0f}%")
    print(f"    总交易成本: {total_cost_bps:.0f} bps")
    print(f"    IC mean: {mean_ic:.4f}, IR: {ir:.2f}")
    print(f"    Hit rate (Top-{top_n}命中): {hit_rate*100:.1f}%")
    print(f"    交易天数: {len(pf)}")
    
    return {
        "cum_ret_pct": round(cum_ret, 1),
        "annual_ret_pct": round(annual_ret, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 1),
        "avg_turnover_pct": round(avg_turnover, 0),
        "total_cost_bps": round(total_cost_bps, 0),
        "ic_mean": round(mean_ic, 4),
        "ir": round(ir, 2),
        "hit_rate_pct": round(hit_rate * 100, 1),
        "n_trading_days": len(pf),
        "pf_df": pf,
    }

# Run for Top N=1,2,3,4,5
results = {}
for n in [1, 2, 3, 4, 5]:
    results[f"top{n}"] = simulate_portfolio(pred_df, n, name=THEME)

# --- Summary ---
print("\n" + "=" * 70)
print(f"SUMMARY: {THEME} THEME | {len(pred_df)} predictions | "
      f"{pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
print("=" * 70)

# Best N
best_n = None
best_sharpe = -1
for n_name, r in results.items():
    n = int(n_name.replace("top", ""))
    print(f"\n  Top{n}: cum={r['cum_ret_pct']}% | "
          f"annual={r['annual_ret_pct']}% | "
          f"Sharpe={r['sharpe']} | "
          f"maxDD={r['max_dd_pct']}% | "
          f"turnover={r['avg_turnover_pct']}% | "
          f"IC={r['ic_mean']}")
    if r['sharpe'] > best_sharpe:
        best_sharpe = r['sharpe']
        best_n = n

# Save daily returns for reference
summary = {
    "theme": THEME,
    "n_stocks": 33,
    "feature_count": len(feature_cols),
    "model": "LightGBM Regression n=100 d=4 lr=0.03 daily expanding",
    "period": f"{pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}",
    "predictions": len(all_predictions),
    "best_n": best_n,
    "best_sharpe": best_sharpe,
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "pf_df"} 
                for k, v in results.items()},
}

# Save results
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"backtest_{THEME}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved: {out_path}")

# Save daily returns CSV for reference
pf_daily = results[f"top{best_n}"]["pf_df"]
pf_daily_path = OUT_DIR / f"daily_rets_{THEME}_top{best_n}.csv"
pf_daily.to_csv(pf_daily_path, index=False)
print(f"Saved: {pf_daily_path}")

print(f"\n=== Best: Top{best_n}, Sharpe={best_sharpe:.2f} ===")
