"""
Walk-Forward 回测 → 80/20 时间分割 (LightGBM on training_data_v13)

目标: T+1 截面超额收益预测 → Top-N 选股 (N=3,4,5)

v13 标签: fwd_1d_excess (单目标, T天截面去均值)
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────
DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v22.parquet"
OUT_PATH = DATA_DIR / "processed" / "walk_forward_lgbm_v22.json"

LABEL_RAW = "fwd_1d_ret"
LABEL = "fwd_1d_excess"

LEAKAGE_FEATS = ["ret_1d", "ret_2d", "ret_5d", "ret_21d"]

SKIP_COLS = {"date", "code",
             "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_2d_excess", "fwd_5d_excess", "fwd_21d_excess"}

# Include MA5/MA20 rolling features, exclude _21d and _cross
def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ─── Load & Clean ────────────────────────────────────────
print(f"Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2020-01-01"].copy()

for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=[LABEL_RAW])
print(f"  After cleaning: {len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  Stocks: {df['code'].nunique()}")

# ─── 截面超额标签 ────────────────────────────────────────
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

print(f"  Label: {LABEL} (截面 demean)")
print(f"    mean={df[LABEL].mean():.6f}, std={df[LABEL].std():.4f}")

# ─── LightGBM ────────────────────────────────────────────
try:
    import lightgbm as lgb
    print(f"  lightgbm: {lgb.__version__}")
except ImportError:
    print("ERROR: lightgbm not installed"); sys.exit(1)

# 80/20 chronological split
unique_dates = sorted(df["date"].unique())
split_idx = int(len(unique_dates) * 0.8)
TRAIN_END = unique_dates[split_idx - 1]
TEST_START = unique_dates[split_idx]

ALL_FEATS = sorted([c for c in df.columns if c not in SKIP_COLS])
features = [f for f in ALL_FEATS if f not in LEAKAGE_FEATS and is_valid_feat(f)]

print(f"  80/20 split: train <= {TRAIN_END.date()}, test >= {TEST_START.date()}")
print(f"  Features: {len(features)} (MA5/MA20, no ret_*, _21d, _cross)")

# ─── Train/Test Split ────────────────────────────────────
train_mask = df["date"] <= TRAIN_END
test_mask = df["date"] >= TEST_START

X_train = df.loc[train_mask, features].copy()
y_train = df.loc[train_mask, LABEL].copy()
dates_train = df.loc[train_mask, "date"].copy()
codes_train = df.loc[train_mask, "code"].copy()

X_test = df.loc[test_mask, features].copy()
y_test = df.loc[test_mask, LABEL].copy()
dates_test = df.loc[test_mask, "date"].copy()
codes_test = df.loc[test_mask, "code"].copy()

# Fill NaN with train median
for c in features:
    if X_train[c].isna().any():
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med if pd.notna(med) else 0)
        X_test[c] = X_test[c].fillna(med if pd.notna(med) else 0)

# ─── Backtest Function ───────────────────────────────────
def run_backtest(X_tr, y_tr, X_te, y_te, dates_te, codes_te, feat_list, label_name, exp_name):
    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.02,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, random_state=42, n_jobs=1, verbosity=-1
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)

    # IC
    ic, _ = spearmanr(pred, y_te)

    # Long-Short: 每日期货式做多Top10% 做空Bottom10%
    test_df = pd.DataFrame({"date": dates_te, "code": codes_te, "pred": pred, "label": y_te.values})
    n = len(test_df)
    sorted_idx = np.argsort(pred)

    top_n = max(n // 10, 1)
    top_decile_ret = y_te.iloc[sorted_idx[-top_n:]].mean()
    bottom_decile_ret = y_te.iloc[sorted_idx[:top_n]].mean()
    ls = top_decile_ret - bottom_decile_ret

    # Hit rate: top decile predictions → actual top decile
    actual_top = set(np.argsort(y_te)[-top_n:])
    pred_top = set(sorted_idx[-top_n:])
    hit_rate = len(actual_top & pred_top) / len(pred_top) if pred_top else 0

    # Sharpe — 非重叠5日采样
    unique_dates_te = np.sort(test_df["date"].unique())
    sampled_dates = set(unique_dates_te[::5])

    def _daily_ls(g):
        n_g = len(g)
        if n_g < 10: return np.nan
        k = max(n_g // 10, 1)
        return g.nlargest(k, "pred")["label"].mean() - g.nsmallest(k, "pred")["label"].mean()

    daily_ls = test_df[test_df["date"].isin(sampled_dates)].groupby("date").apply(_daily_ls).dropna()
    sharpe = daily_ls.mean() / daily_ls.std() * np.sqrt(252/5) if len(daily_ls) > 1 and daily_ls.std() > 0 else 0
    daily_ls_raw = test_df.groupby("date").apply(_daily_ls).dropna()
    sharpe_raw = daily_ls_raw.mean() / daily_ls_raw.std() * np.sqrt(252) if len(daily_ls_raw) > 1 and daily_ls_raw.std() > 0 else 0

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_list).sort_values(ascending=False)

    print(f"  [{exp_name}] IC={ic:.4f}, LS={ls:.4f}, Sharpe={sharpe:.2f} (raw={sharpe_raw:.2f}), Hit={hit_rate:.4f}")

    return {
        "ic": round(ic, 4), "long_short": round(ls, 4),
        "hit_rate": round(hit_rate, 4),
        "sharpe": round(sharpe, 2), "sharpe_raw_overlap": round(sharpe_raw, 2),
        "top5_features": {k: round(v, 0) for k, v in imp.head(5).items()},
        "model": model, "predictions": pred,
    }

# ─── Portfolio Simulation ─────────────────────────────────
def run_portfolio(test_df, pred, top_n):
    """每日选 Top-N，等权持有 T+1，计算组合收益"""
    df_pf = test_df.copy()
    df_pf["pred"] = pred

    daily_ret = []
    for date, g in df_pf.groupby("date"):
        g_sorted = g.nlargest(top_n, "pred")
        if len(g_sorted) >= top_n:
            daily_ret.append({"date": date, "ret": g_sorted["label"].mean()})

    pf = pd.DataFrame(daily_ret).sort_values("date")
    pf["cum"] = (1 + pf["ret"]).cumprod()

    # Non-overlap Sharpe
    pf_sampled = pf.iloc[::5]
    pf_sharpe = pf_sampled["ret"].mean() / pf_sampled["ret"].std() * np.sqrt(252/5) if pf_sampled["ret"].std() > 0 else 0

    avg_turnover = 0  # simplified
    cum_ret = (pf["cum"].iloc[-1] - 1) * 100 if len(pf) > 0 else 0

    return {"sharpe": round(pf_sharpe, 2), "cum_ret": round(cum_ret, 2), "turnover": 0}

# ─── Run Experiment ─────────────────────────────────────
print()
print("=" * 70)
print("Experiment: T+1 excess (short features, n=400 d=4)")
print("=" * 70)
results = run_backtest(X_train, y_train, X_test, y_test, dates_test, codes_test, features, LABEL, "T+1")

# ─── Portfolio: Top-N ─────────────────────────────────────
print()
print("=" * 70)
print("Portfolio Simulation: Top N daily rotation (N=3,4,5)")
print("=" * 70)

test_df_full = pd.DataFrame({
    "date": dates_test, "code": codes_test, "label": y_test.values
})

pf_results = {}
for n in [3, 4, 5]:
    pf = run_portfolio(test_df_full, results["predictions"], n)
    print(f"  Portfolio Top{n}: Sharpe={pf['sharpe']}, cum_ret={pf['cum_ret']}%")
    pf_results[f"top{n}"] = pf

# ─── Summary ─────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY — 80/20 split from 2020")
print("=" * 70)
print()
print(f"  Label: {LABEL} = fwd_1d_ret - cross_sectional_mean")
print(f"  Train: 2020 ~ {TRAIN_END.date()}, Test: {TEST_START.date()} ~ 2026")
print(f"  Features: {len(features)} (MA5/MA20, no ret_*, _21d, _cross)")
print()

output = {
    "label": LABEL,
    "label_desc": "T+1截面超额收益 (fwd_1d_ret demeaned by date)",
    "model": "LightGBM n=400 depth=4 lr=0.03, short features only",
    "split": f"80/20 from 2020: train <= {TRAIN_END.date()}, test >= {TEST_START.date()}",
    "experiment": {
        "window": f"{TRAIN_END.date()}~{TEST_START.date()}",
        "train_size": len(X_train), "test_size": len(X_test),
        "features": len(features),
        **{k: v for k, v in results.items() if k not in ("model", "predictions")},
    },
    "portfolio": pf_results,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_PATH}")
