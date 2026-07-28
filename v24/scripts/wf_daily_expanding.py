"""
Walk-Forward 每日扩展回测: 2025-09 ~ latest
训练: Regression, n=400 d=4 lr=0.03, 64线程
标签: fwd_1d_excess (截面demean)
特征: MA5/20, 排除_21d/_cross/ret_1d/2d/5d/21d
"""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser(description="Walk-Forward 每日扩展回测")
parser.add_argument("--reverse", action="store_true", help="反向策略: 选预测分最低的Top3买入")
parser.add_argument("--train-start", type=str, default=None,
                    help="训练数据起点(含); 默认=None用全量(2022-09起)")
parser.add_argument("--test-start", type=str, default="2025-09-01",
                    help="回测起点(含); 默认 2025-09-01")
parser.add_argument("--test-end", type=str, default=None,
                    help="回测终点(含); 默认=None用最新日")
parser.add_argument("--train-data", type=str, default="training_data_v22.parquet",
                    help="训练集parquet文件名(位于data/processed); v22默认, v23实验传training_data_v23.parquet")
args = parser.parse_args()
REVERSE = args.reverse
TRAIN_START = args.train_start
TEST_START = args.test_start
TEST_END = args.test_end

DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
TRAIN_DATA = args.train_data
TRAIN_PATH = DATA_DIR / "processed" / TRAIN_DATA

def _out_suffix():
    s = ""
    if TRAIN_START is not None:
        s += f"_tr{TRAIN_START}"
    if TEST_START != "2025-09-01":
        s += f"_ts{TEST_START}"
    if TEST_END is not None:
        s += f"_te{TEST_END}"
    return s
# 从训练集文件名提取版本标签(v22/v23...), 保证v22默认输出不变, v23独立
_TAG = TRAIN_DATA.replace("training_data_", "").replace(".parquet", "")
OUT_NAME = (f"wf_daily_{_TAG}_reverse" if REVERSE else f"wf_daily_{_TAG}") + _out_suffix() + ".json"
OUT_PATH = DATA_DIR / "processed" / OUT_NAME

LABEL_RAW = "fwd_1d_exec_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret"}

# ── 锁定超参数 (2026-07-07 用户三次确认, 单一真相源) ──
# 任何训练/smoke/打印必须引用此字典, 禁止散写 n=100/400 或 lr=0.02/0.03
LOCKED_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=50, random_state=42, n_jobs=64, verbosity=-1
)
def smoke_test(df, features, LABEL):
    """跑前关卡: 实际fit一次(min样本)打印真实模型参数, 断言=锁定值, 否则中止"""
    print("[smoke] 快速参数/流程校验 (最后90天窗口实际训练)...")
    all_d = sorted(df["date"].unique())
    cut = all_d[-90]
    sub = df[df["date"] >= cut].copy()
    Xs = sub.groupby("code")[features].transform(lambda s: s.ffill().fillna(0))
    ys = sub[LABEL]
    m = lgb.LGBMRegressor(**LOCKED_PARAMS)
    m.fit(Xs, ys)
    p = m.get_params()
    print(f"[smoke] 实际训练参数: n_estimators={p['n_estimators']} max_depth={p['max_depth']} "
          f"lr={p['learning_rate']}")
    assert p["n_estimators"] == 400 and p["max_depth"] == 4 and abs(p["learning_rate"] - 0.03) < 1e-9, \
        "❌ 致命: 实际训练参数≠锁定值(400/0.03), 中止以防重跑浪费"
    print("[smoke] ✅ 参数=锁定值, 流程关卡通过, 开始全量WF")

MIN_TRAIN_DAYS = 250
TRADE_COST = 0.0006  # 6bp per side
# TEST_START 由 argparse --test-start 控制 (默认 "2025-09-01"), 不再硬编码以免覆盖传入值

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ── Load ──
print("Loading...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[LABEL_RAW])
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

if TRAIN_START is not None:
    _ts = pd.Timestamp(TRAIN_START)
    df = df[df["date"] >= _ts].copy()
    print(f"  [train-start] 过滤训练数据 >= {_ts.date()} → {len(df):,} rows")

all_cols = [c for c in df.columns if c not in SKIP_COLS and is_valid_feat(c)]
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

print(f"\nWalk-Forward: {len(dates)} prediction days ({dates[0].date()} ~ {dates[-1].date()})")
print(f"Model: Regression n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
      f"lr={LOCKED_PARAMS['learning_rate']}, MIN_TRAIN={MIN_TRAIN_DAYS}")

# ── 跑前关卡: 实际训练一次验证参数+流程, 不一致直接中止 ──
smoke_test(df, features, LABEL)

daily_results = []
last_ic = 0
t0 = datetime.now()

for day_idx, pred_date in enumerate(dates):
    # Training data: all data before pred_date
    train_mask = df["date"] < pred_date
    train_df = df[train_mask]
    unique_dates = train_df["date"].nunique()
    if unique_dates < MIN_TRAIN_DAYS:
        continue

    X_train = train_df[features].copy()
    y_train = train_df[LABEL].copy()

    # ffill per stock, NOT across stocks (⚠️ 跨股票泄漏灾)
    X_train = train_df.groupby("code")[features].transform(lambda s: s.ffill().fillna(0))

    # Train (每日扩展重训, 固定400棵树, 无eval/early-stopping — eval会导致训练随窗口爆炸变慢)
    model = lgb.LGBMRegressor(**LOCKED_PARAMS)
    model.fit(X_train, y_train)

    # Predict on pred_date
    test_mask = df["date"] == pred_date
    X_test = df.loc[test_mask, features].copy()
    y_test = df.loc[test_mask, LABEL].copy()
    codes_test = df.loc[test_mask, "code"].values

    for c in features:
        if X_test[c].isna().any():
            X_test[c] = X_test[c].fillna(0)

    preds = model.predict(X_test)

    # IC
    if len(preds) > 5:
        ic, _ = spearmanr(preds, y_test)
        last_ic = ic if not np.isnan(ic) else last_ic
    else:
        ic = np.nan

    # Top3 excess return (with transaction cost)
    test_df = pd.DataFrame({"code": codes_test, "pred": preds, "label": y_test.values})
    n_test = len(test_df)
    top_n = min(3, n_test)
    if REVERSE:
        top3 = test_df.nsmallest(top_n, "pred")   # 反向: 买预测分最低的
    else:
        top3 = test_df.nlargest(top_n, "pred")
    top3_ret = top3["label"].mean() - TRADE_COST  # 仅卖出 0.06% (不计买入/滑点)

    # Hit rate
    actual_topk = set(np.argsort(y_test)[-top_n:])
    pred_topk = set(np.argsort(preds)[-top_n:])
    hit = len(actual_topk & pred_topk) / top_n if top_n > 0 else 0

    daily_results.append({
        "date": str(pred_date.date()),
        "n_train": len(X_train),
        "n_test": n_test,
        "ic": round(ic, 4) if not np.isnan(ic) else None,
        "top3_ret": round(top3_ret, 6),
        "hit_rate": round(hit, 3),
        "long_short": round(preds.mean(), 6) if len(preds) > 0 else 0,
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

# Cum returns (non-overlapping sampling)
pf = pd.DataFrame(rdf[["date", "top3_ret"]].copy())
pf["date"] = pd.to_datetime(pf["date"])
pf = pf.sort_values("date")
pf["cum_raw"] = (1 + pf["top3_ret"]).cumprod() - 1

# Non-overlap Sharpe (every 5th day)
pf_sampled = pf.iloc[::5]
sharpe = float(pf_sampled["top3_ret"].mean() / pf_sampled["top3_ret"].std() * np.sqrt(252/5)) if pf_sampled["top3_ret"].std() > 0 else 0

# Overlap Sharpe (raw)
sharpe_raw = float(pf["top3_ret"].mean() / pf["top3_ret"].std() * np.sqrt(252)) if pf["top3_ret"].std() > 0 else 0

# Max DD
cum_series = (1 + pf["top3_ret"]).cumprod()
running_max = cum_series.expanding().max()
drawdown = (cum_series - running_max) / running_max
max_dd = float(drawdown.min())

# Win rate
win_rate = float((pf["top3_ret"] > 0).mean())

# Total cost estimate
total_trades = len(rdf)  # 卖出次数 = 每日TopN × 换手率
total_cost = total_trades * TRADE_COST

print(f"\n{'='*60}")
print(f"WALK-FORWARD RESULTS ({'REVERSE Bottom3' if REVERSE else 'Top3'}, daily expanding, "
      f"{TEST_START} ~ {TEST_END or 'latest'})")
print(f"{'='*60}")
print(f"  Days: {len(rdf)}")
print(f"  IC: mean={valid['ic'].mean():.4f} std={valid['ic'].std():.4f}")
print(f"  Top3 excess: mean={rdf['top3_ret'].mean():+.6f}")
print(f"  Cum return (raw): {pf['cum_raw'].iloc[-1]*100:.1f}%")
print(f"  Sharpe: {sharpe:.2f} (raw={sharpe_raw:.2f})")
print(f"  Max DD: {max_dd*100:.1f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Hit rate: {rdf['hit_rate'].mean():.3f}")
print(f"  Est total cost: {total_cost*100:.2f}% of capital")

# IC by month
rdf["month"] = rdf["date"].str[:7]
monthly_ic = rdf.groupby("month")["ic"].mean()
print(f"\n  Monthly IC:")
for m, v in monthly_ic.items():
    if pd.isna(v): continue
    bar = "+" * max(1, int(abs(v)*1000)) if v > 0 else "-" * max(1, int(abs(v)*1000))
    print(f"    {m}: {v:+.4f} {bar}")

# Annualized return
n_days = len(rdf)
ann_ret = float((1 + pf["top3_ret"]).prod() ** (252/n_days) - 1) if n_days > 0 else 0
print(f"\n  Annualized return: {ann_ret*100:.1f}%")
print(f"  Total elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

# Save
output = {
    "label": LABEL,
    "model": f"LightGBM Regression n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
              f"lr={LOCKED_PARAMS['learning_rate']}, 64线程CPU, daily expanding, fixed 400 trees"
              f", {'REVERSE Bottom3' if REVERSE else 'Top3'}",
    "features": len(features),
    "period": f"{rdf['date'].iloc[0]} ~ {rdf['date'].iloc[-1]}",
    "n_prediction_days": len(rdf),
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
    },
    "monthly_ic": {str(k): round(float(v), 4) for k, v in monthly_ic.items()},
    "daily": daily_results[-10:],  # last 10 days
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_PATH}")
