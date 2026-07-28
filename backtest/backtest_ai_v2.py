"""
AI Theme Backtest - Minimal version with file logging
"""
import sys, os
log_file = open("D:/myAI/Claw/ai_backtest_log.txt", "w", encoding="utf-8")

def log(msg):
    log_file.write(msg + "\n")
    log_file.flush()
    print(msg, flush=True)

log("=== AI Theme Backtest Started ===")
log("Python: " + sys.version)
log("Encoding: " + sys.getdefaultencoding())

import pandas as pd
import numpy as np
log("Imported pandas " + pd.__version__ + ", numpy " + np.__version__)

import lightgbm as lgb
log("Imported lightgbm " + lgb.__version__)

from scipy.stats import spearmanr
import json, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

# --- Load ---
DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
THEME = "AI"
TRAIN_PATH = DATA_DIR / "processed" / f"training_data_theme_{THEME}.parquet"

log(f"Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "code"]).reset_index(drop=True)
log(f"  Shape: {df.shape}, Date: {df['date'].min().date()} ~ {df['date'].max().date()}, Stocks: {df['code'].nunique()}")

# Clean
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

LABEL_RAW = "fwd_1d_ret"
df = df.dropna(subset=[LABEL_RAW])
log(f"  After dropna: {len(df)} rows")

# Features
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_2d_excess", "fwd_5d_excess", "fwd_21d_excess"}
LEAKAGE_FEATS = ["ret_1d", "ret_2d", "ret_5d", "ret_21d"]

def is_valid_feat(name):
    if "_21d" in name: return False
    if name.endswith("_cross"): return False
    if name.endswith("_ma2"): return False
    return True

features = sorted([c for c in df.columns if c not in SKIP_COLS and c not in LEAKAGE_FEATS and is_valid_feat(c)])
log(f"  Features: {len(features)}")

# --- Daily Expanding Retrain ---
unique_dates = sorted(df["date"].unique())
log(f"  Total trading days: {len(unique_dates)}")

MIN_TRAIN_DAYS = 250
predict_dates = unique_dates[MIN_TRAIN_DAYS:]
log(f"  Prediction: {predict_dates[0].date()} ~ {predict_dates[-1].date()} ({len(predict_dates)} days)")

all_predictions = []
N = len(predict_dates)

log(f"\nStarting daily expanding retrain (n=100, d=4, lr=0.03, 64 threads)...")

for i, target_date in enumerate(predict_dates):
    train_mask = df["date"] < target_date
    test_mask = df["date"] == target_date
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    if len(df_train) < 2500:
        continue
    if len(df_test) < 3:
        continue
    
    # Excess return
    cs_mean_train = df_train.groupby("date")[LABEL_RAW].transform(np.mean)
    cs_mean_test = df_test[LABEL_RAW].mean()
    y_train = df_train[LABEL_RAW].values - cs_mean_train.values
    y_test = df_test[LABEL_RAW].values - cs_mean_test
    
    X_train = df_train[features].copy().fillna(0.0)
    X_test = df_test[features].copy().fillna(0.0)
    
    # Fill NaN with train median
    for c in features:
        if X_train[c].isna().any():
            med = X_train[c].median()
            fill_val = med if pd.notna(med) else 0.0
            X_train[c] = X_train[c].fillna(fill_val)
            X_test[c] = X_test[c].fillna(fill_val)
    
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)
    
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
            log(f"  [WARN] Train failed at {target_date.date()}: {e}")
        continue
    
    for j in range(len(df_test)):
        all_predictions.append({
            "date": target_date,
            "code": str(df_test.iloc[j]["code"]),
            "pred": float(preds[j]),
            "actual_ret": float(df_test.iloc[j][LABEL_RAW]),
            "actual_excess": float(y_test[j])
        })
    
    if (i + 1) % 100 == 0:
        log(f"  [{i+1}/{N}] Days completed")

log(f"\nTotal predictions: {len(all_predictions)}")

# --- Portfolio Simulation ---
pred_df = pd.DataFrame(all_predictions)
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df = pred_df.sort_values(["date", "pred"], ascending=[True, False])

log("\n=== PORTFOLIO SIMULATION ===")

def simulate(pred_df, top_n, cost_rate=0.0006):
    daily_rets = []
    prev_holdings = set()
    
    for d in sorted(pred_df["date"].unique()):
        day_data = pred_df[pred_df["date"] == d]
        if len(day_data) < top_n:
            continue
        
        day_data = day_data.nlargest(top_n, "pred")
        holdings = set(day_data["code"].values)
        pf_ret = day_data["actual_ret"].mean()
        
        sold = prev_holdings - holdings
        turnover = len(sold) / top_n if prev_holdings else 0
        cost = turnover * cost_rate
        net_ret = pf_ret - cost
        
        daily_rets.append({"date": d, "ret": pf_ret, "cost": cost, "net_ret": net_ret,
                          "n_holdings": len(holdings), "turnover_sold": len(sold)})
        prev_holdings = holdings
    
    pf = pd.DataFrame(daily_rets).sort_values("date")
    if len(pf) == 0:
        return {"cum_ret": 0, "annual_ret": 0, "sharpe": 0, "max_dd": 0, "pf": pf, "hit_rate": 0}
    
    pf["cum"] = (1 + pf["net_ret"]).cumprod()
    cum_ret = (pf["cum"].iloc[-1] - 1) * 100
    
    n_years = (pf["date"].max() - pf["date"].min()).days / 365.25
    annual_ret = (pf["cum"].iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    
    pf_sampled = pf.iloc[::5]
    sharpe = pf_sampled["net_ret"].mean() / pf_sampled["net_ret"].std() * np.sqrt(252/5) if len(pf_sampled) > 5 and pf_sampled["net_ret"].std() > 0 else 0
    
    cum_max = pf["cum"].cummax()
    dd = (pf["cum"] - cum_max) / cum_max * 100
    max_dd = float(dd.min())
    
    # Hit rate
    hit_count = 0
    total_tries = 0
    for d in sorted(pred_df["date"].unique()):
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
    for d in sorted(pred_df["date"].unique()):
        day_data = pred_df[pred_df["date"] == d]
        if len(day_data) < 5:
            continue
        ic, _ = spearmanr(day_data["pred"], day_data["actual_ret"])
        if not np.isnan(ic):
            ic_vals.append(ic)
    mean_ic = np.mean(ic_vals) if ic_vals else 0
    ir = mean_ic / np.std(ic_vals) if len(ic_vals) > 1 and np.std(ic_vals) > 0 else 0
    
    avg_turnover = pf["turnover_sold"].mean() / top_n * 100
    total_cost = pf["cost"].sum() * 10000
    
    log(f"\n  Top{top_n}: cum={cum_ret:.1f}% | annual={annual_ret:.1f}% | Sharpe={sharpe:.2f} | maxDD={max_dd:.1f}% | turnover={avg_turnover:.0f}% | cost={total_cost:.0f}bps | IC={mean_ic:.4f} | Hit={hit_rate*100:.1f}%")
    
    return {
        "top_n": top_n,
        "cum_ret_pct": round(cum_ret, 1),
        "annual_ret_pct": round(annual_ret, 1),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 1),
        "avg_turnover_pct": round(avg_turnover, 0),
        "total_cost_bps": round(total_cost, 0),
        "ic_mean": round(mean_ic, 4),
        "ir": round(ir, 2),
        "hit_rate_pct": round(hit_rate * 100, 1),
        "n_trades": len(pf),
        "pf": pf,
    }

results = {}
for n in [1, 2, 3, 4, 5]:
    results[f"top{n}"] = simulate(pred_df, n)

# --- Summary ---
log("\n=== SUMMARY ===")
log(f"Theme: {THEME}, Stocks: 33, Features: {len(features)}")
log(f"Period: {pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
log(f"Trading days: {len(unique_dates)}, Prediction days: {len(predict_dates)}")
log(f"Total predictions: {len(all_predictions)}")

best_n = None
best_sharpe = -1
for n in [1, 2, 3, 4, 5]:
    r = results[f"top{n}"]
    log(f"  Top{n}: cum={r['cum_ret_pct']:>7.1f}% annual={r['annual_ret_pct']:>7.1f}% Sharpe={r['sharpe']:>5.2f} maxDD={r['max_dd_pct']:>6.1f}% Hit={r['hit_rate_pct']:>5.1f}% IC={r['ic_mean']:>.4f} turnover={r['avg_turnover_pct']:.0f}%")
    if r['sharpe'] > best_sharpe:
        best_sharpe = r['sharpe']
        best_n = n

log(f"\n=== Best: Top{best_n}, Sharpe={best_sharpe:.2f} ===")

# Save
OUT_DIR = DATA_DIR / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "theme": THEME, "n_stocks": 33, "feature_count": len(features),
    "model": "LightGBM n=100 d=4 lr=0.03 daily expanding 64t",
    "period": f"{pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}",
    "best_n": best_n, "best_sharpe": best_sharpe,
    "results": {k: {kk: vv for kk, vv in v.items() if kk != "pf"} for k, v in results.items()},
}

with open(OUT_DIR / f"backtest_{THEME}.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

pf_best = results[f"top{best_n}"]["pf"]
pf_best.to_csv(OUT_DIR / f"daily_rets_{THEME}_top{best_n}.csv", index=False)

log(f"\nSaved to {OUT_DIR}")
log("=== COMPLETE ===")
log_file.close()
