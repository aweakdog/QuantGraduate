"""
合并关注圈(245只) → 重新训练 → 回测
"""
import pandas as pd, numpy as np, lightgbm as lgb, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.feature_engine import (
    read_kline, calc_technical_features, calc_labels,
    calc_fund_features
)

DATA = "data"

# 1. Load original + new fund flow
ff_orig = pd.read_parquet(f"{DATA}/raw/fund_flow_full/fundflow_history.parquet")
ff_orig["date"] = pd.to_datetime(ff_orig["date"])
ff_new = pd.read_parquet(f"{DATA}/raw/fund_flow_full/oos_fundflow.parquet")
ff_new["date"] = pd.to_datetime(ff_new["date"])
ff_all = pd.concat([ff_orig, ff_new]).drop_duplicates(["date","code"], keep="last").sort_values(["code","date"])
print(f"Fund flow: {ff_all['code'].nunique()} stocks")

# 2. New stock codes
new_codes_6 = sorted(ff_new["code"].unique())
new_full = []
for c in new_codes_6:
    mkt = ".SZ" if c.startswith(("0","3")) else ".SH"
    new_full.append(f"{c}{mkt}")
print(f"New stocks: {len(new_full)}")

# 3. Load original training data
orig = pd.read_parquet(f"{DATA}/processed/training_data_v5.parquet")
orig["date"] = pd.to_datetime(orig["date"])
print(f"Original: {orig.shape}, {orig['code'].nunique()} stocks")

# 4. Build features for new stocks
existing_cols = set(orig.columns)
new_rows = []
for i, code_full in enumerate(new_full):
    code6 = code_full[:6]
    df_k = read_kline(code6)
    if df_k is None or len(df_k) < 60:
        continue
    tech = calc_technical_features(df_k)
    if tech is None:
        continue
    labels = calc_labels(df_k)
    result = tech.merge(labels, on="date", how="left")
    tech_cols = [c for c in tech.columns if c != "date"]
    for c in tech_cols:
        result[c] = result[c].shift(1)

    # Fund flow
    sub = ff_all[ff_all["code"] == code6].copy()
    if len(sub) > 0:
        fund = calc_fund_features(sub)
        if fund is not None:
            fund["date"] = pd.to_datetime(fund["date"]) + pd.Timedelta(days=1)
            result = result.merge(fund, on="date", how="left")

    # Macro
    try:
        df_pmi = pd.read_parquet(f"{DATA}/raw/macro/中国PMI.parquet")
        df_pmi["date"] = pd.to_datetime(
            df_pmi["月份"].str.replace("年", "-").str.replace("月份", "-01"), errors="coerce"
        )
        df_pmi["date"] = df_pmi["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        df_pmi_v = df_pmi[["date", "制造业-指数"]].rename(columns={"制造业-指数": "cn_pmi"})
        result = result.merge(df_pmi_v, on="date", how="left")
        result["cn_pmi"] = result["cn_pmi"].ffill()
    except:
        pass
    try:
        df_ism = pd.read_parquet(f"{DATA}/raw/macro/美国ISM制造业PMI.parquet")
        df_ism["date"] = pd.to_datetime(df_ism["日期"], errors="coerce")
        df_ism["date"] = df_ism["date"] + pd.Timedelta(days=1)
        df_ism_v = df_ism[["date", "今值"]].rename(columns={"今值": "us_ism_pmi"})
        result = result.merge(df_ism_v, on="date", how="left")
        result["us_ism_pmi"] = result["us_ism_pmi"].ffill()
    except:
        pass

    result["has_leader"] = 0
    result["leader_count"] = 0
    result["leader_exp"] = 0
    result["leader_binding_sum"] = 0.0
    result["code"] = code_full
    new_rows.append(result)

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(new_full)}] {code_full}")

if not new_rows:
    print("No new stocks built!")
    sys.exit(1)

new_df = pd.concat(new_rows, ignore_index=True)
print(f"New: {new_df.shape}, {new_df['code'].nunique()} stocks")

# 5. Merge with original
combined = pd.concat([orig, new_df], ignore_index=True)
combined = combined.sort_values(["date", "code"]).reset_index(drop=True)
# Add missing columns from original that new stocks don't have
for c in existing_cols - set(combined.columns):
    combined[c] = np.nan
print(f"Combined: {combined.shape}, {combined['code'].nunique()} stocks")

# 6. Train & backtest
combined = combined[combined["date"] >= "2020-01-01"].copy()
for c in combined.select_dtypes(include=[np.number]).columns:
    combined[c] = combined[c].replace([np.inf, -np.inf], np.nan)
combined = combined.dropna(subset=["fwd_1d_ret"])

dates = sorted(combined["date"].unique())
split = int(len(dates) * 0.8)
train_end, test_start = dates[split - 1], dates[split]

skip = {"date", "code", "fwd_1d_ret", "fwd_5d_ret", "fwd_10d_ret", "fwd_20d_ret"}
feats = [c for c in combined.columns if c not in skip]
label = "fwd_1d_excess"
combined[label] = combined.groupby("date")["fwd_1d_ret"].transform(lambda x: x - x.mean())

train = combined[combined["date"] <= train_end]
test = combined[combined["date"] >= test_start]

X_train = train[feats].values
y_train = train[label].values
med = np.nanmedian(X_train, axis=0)
med = np.where(np.isnan(med), 0, med)
X_train = np.where(np.isnan(X_train), med, X_train)
y_train = np.clip(y_train, -0.5, 0.5)

model = lgb.LGBMRegressor(
    n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.05,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
)
model.fit(X_train, y_train)

test = test.copy()
test["pred"] = model.predict(test[feats].values)

dates_list = sorted(test["date"].unique())
TC = 0.0006

print(f"\n{'='*55}")
print(f"  MERGED WATCHLIST ({combined['code'].nunique()} stocks)")
print(f"{'='*55}")
print(f"  Train: {train['code'].nunique()} stocks, {len(train)} rows")
print(f"  Test:  {test['code'].nunique()} stocks, {len(test)} rows")
print(f"  Period: {test_start.date()} ~ {dates_list[-1].date()}")
print()
print(f"  {'Top N':>6} {'Return':>10} {'Ann.':>10} {'Sharpe':>8} {'MDD':>8} {'Win%':>6}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

for top_n in [2, 3, 4, 5, 7]:
    picks = {}
    for d in dates_list:
        g = test[test["date"] == d]
        picks[d] = set(g.sort_values("pred", ascending=False).head(top_n)["code"].values)

    h = set()
    r = []
    for i in range(len(dates_list) - 1):
        t = dates_list[i]
        pt = picks[t]
        if i == 0:
            h = pt
            continue
        t1 = dates_list[i + 1]
        a = test[(test["date"] == t1) & (test["code"].isin(h))]
        if len(a) == 0:
            h = picks[t1]
            continue
        gr = a["fwd_1d_ret"].mean()
        nh = picks[t1]
        sp = len(h - nh) / len(h) if h else 0
        r.append(gr - sp * TC)
        h = nh

    if len(r) < 10:
        continue
    rr = pd.Series(r)
    cum = (1 + rr).prod() - 1
    ann = (1 + cum) ** (252 / len(rr)) - 1
    peak = (1 + rr).cumprod().expanding().max()
    mdd = ((1 + rr).cumprod() / peak - 1).min()
    sharpe = rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else 0
    win = (rr > 0).mean()
    print(f"  {top_n:>6} {cum*100:>9.1f}% {ann*100:>9.1f}% {sharpe:>7.2f} {mdd*100:>7.1f}% {win*100:>5.1f}%")

# Save model
import joblib
model_path = f"{DATA}/processed/model_merged_v1.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved: {model_path}")
