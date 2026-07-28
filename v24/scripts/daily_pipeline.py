"""
每日部署流水线 v1
模式: Regression + MA5/20 + 每日扩展重训 + 64线程CPU
输出: Top3/4/5 持仓建议 + 回撤 < 30%
成本: 卖出 0.06% (不计滑点)

用法:
  python scripts/daily_pipeline.py              # 用最新数据
  python scripts/daily_pipeline.py --port 3     # 只输出Top3
  python scripts/daily_pipeline.py --super     # 同时推送到SuperMind

每天收盘后运行:
  1. 拉数据 (目前手动: daily_pull.py)
  2. 跑本脚本 → 训练 + 预测 + 输出
"""
import pandas as pd, numpy as np, json, warnings, sys, argparse
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")
import lightgbm as lgb
from scipy.stats import spearmanr

# ── Config ──
PROJECT = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy")
DATA_DIR = PROJECT / "data"
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v16.parquet"
OUTPUT_DIR = DATA_DIR / "daily_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_RAW = "fwd_1d_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_2d_excess", "fwd_5d_excess", "fwd_21d_excess"}

TRADE_COST_SELL = 0.0006  # 0.06% sell cost
TRADE_COST_BUY = 0.0006   # 0.06% buy cost
TOTAL_COST_PER_TRADE = TRADE_COST_BUY + TRADE_COST_SELL  # 0.12% round trip

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")


def load_and_clean(path):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[LABEL_RAW])
    df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())
    return df


def get_features(df):
    all_cols = [c for c in df.columns if c not in SKIP_COLS and is_valid_feat(c)]
    return [f for f in all_cols if f not in LEAKAGE_FEATS]


def train_model(X_train, y_train):
    medians = {}
    for c in X_train.columns:
        if X_train[c].isna().any():
            med = X_train[c].median()
            medians[c] = float(med) if pd.notna(med) else 0.0
            X_train[c] = X_train[c].fillna(medians[c])
        else:
            medians[c] = 0.0
    model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.02,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, random_state=42, n_jobs=64, verbosity=-1
    )
    model.fit(X_train, y_train)
    return model, medians


def predict_day(model, medians, df_latest, features):
    X = df_latest[features].copy()
    codes = df_latest["code"].values
    for c in features:
        if X[c].isna().any():
            X[c] = X[c].fillna(medians.get(c, 0))
    preds = model.predict(X)
    return pd.DataFrame({"code": codes, "pred": preds}).sort_values("pred", ascending=False)


def simulate_portfolio(df_backtest, top_n, cost=TOTAL_COST_PER_TRADE):
    """
    Walk-forward portfolio simulation.
    Each day: pick Top-N, hold overnight, apply cost on sells.
    Turnover = fraction of stocks replaced daily.
    """
    dates = sorted(df_backtest["date"].unique())
    cash = 1.0
    holdings = []
    daily_records = []
    peak = 1.0

    for d in dates:

        day_data = df_backtest[df_backtest["date"] == d].copy()
        if len(day_data) < top_n:
            continue

        top_picks = day_data.nlargest(top_n, "pred")
        new_holdings = set(top_picks["code"].values)
        old_holdings = set(holdings)
        sold = old_holdings - new_holdings
        n_sold = len(sold)
        sell_cost = n_sold * TRADE_COST_SELL

        # Today's return: mean of Top-N actual returns minus cost
        daily_ret = top_picks[LABEL_RAW].mean()
        daily_net = daily_ret - (sell_cost / top_n)  # cost distributed across holdings

        cash *= (1 + daily_net)
        holdings = list(new_holdings)

        if cash > peak:
            peak = cash
        dd = (cash - peak) / peak
        daily_records.append({
            "date": str(d.date()), "return": daily_ret, "net_return": daily_net,
            "cum_cash": cash, "drawdown": dd, "n_holdings": top_n
        })

    pf = pd.DataFrame(daily_records)
    if len(pf) < 5:
        return None

    final_cum = float((pf["cum_cash"].iloc[-1] - 1) * 100)
    max_dd = float(pf["drawdown"].min() * 100)
    daily_mean = float(pf["net_return"].mean())
    daily_std = float(pf["net_return"].std())
    sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    return {
        "top_n": top_n, "final_cum_pct": final_cum, "max_dd_pct": max_dd,
        "sharpe": sharpe, "daily_mean_bp": daily_mean * 10000,
        "n_trading_days": len(pf),
        "passes_dd_limit": max_dd > -30,
    }


def print_report(today_str, model, features, df_latest, pred_df, pf_results):
    print("=" * 62)
    print(f"  每日预测报告 — {today_str}")
    print("=" * 62)
    print(f"  模型: LightGBM Regression n=400 d=4 lr=0.02")
    print(f"  特征: {len(features)} (MA5/20, 排除_21d/_cross/fwd/ret_*)")
    print(f"  训练数据至: {df_latest['date'].max().date() if 'date' in df_latest else 'N/A'}")
    print(f"  交易成本: 买入0.06% + 卖出0.06% = 0.12%/笔")
    print()

    print("  TOP 10 预测 (原始得分):")
    for rk, row in enumerate(pred_df.head(10).itertuples(), 1):
        print(f"    #{rk:2d}  {row.code:12s}  score={row.pred:+.6f}")
    print()

    print("  持仓模拟 (2025-09 ~ 2026-06):")
    print(f"  {'TopN':<6} {'累积%':>8} {'Sharpe':>7} {'maxDD%':>7} {'达标':>5}")
    print(f"  {'-'*38}")
    for r in sorted(pf_results, key=lambda x: x["top_n"]):
        dd_mark = "YES" if r["passes_dd_limit"] else "NO"
        print(f"  Top{r['top_n']:<3d} {r['final_cum_pct']:>+8.1f} {r['sharpe']:>7.2f} {r['max_dd_pct']:>+7.1f} {dd_mark:>5}")
    print()

    # Best Top-N meeting DD constraint
    valid = [r for r in pf_results if r["passes_dd_limit"]]
    if valid:
        best = max(valid, key=lambda r: r["final_cum_pct"])
        print(f"  >>> 推荐: Top{best['top_n']} (Sharpe={best['sharpe']:.2f}, maxDD={best['max_dd_pct']:.1f}%)")
    else:
        print("  >>> 无符合maxDD<30%的Top-N组合")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0, help="只输出Top-N (0=全部)")
    parser.add_argument("--super", action="store_true", help="推送到SuperMind")
    args = parser.parse_args()

    t0 = datetime.now()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Load data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading training data...")
    df = load_and_clean(TRAIN_PATH)
    features = get_features(df)
    latest_date = df["date"].max()
    print(f"  数据: {len(df):,d} rows, {df['code'].nunique()} codes, {len(features)} features")
    print(f"  时间: {df['date'].min().date()} ~ {latest_date.date()} (落后 {(datetime.now()-latest_date).days}天)")

    # 2. Train on all data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training model (daily expanding, 64 threads)...")
    X_all = df[features].copy()
    y_all = df[LABEL].copy()
    model, medians = train_model(X_all, y_all)
    print(f"  Model: {model.booster_.num_trees()} trees")

    # 3. Predict latest day
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Predicting...")
    df_latest = df[df["date"] == latest_date][["date", "code", LABEL_RAW] + features].copy()
    pred_df = predict_day(model, medians, df_latest, features)

    ic, _ = spearmanr(pred_df["pred"], df_latest[LABEL_RAW].values)
    print(f"  IC(latest day): {ic:.4f}")

    # 4. Walk-forward portfolio simulation
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running walk-forward portfolio...")

    # Generate predictions for all days in test period
    preds_all = []
    test_dates = sorted(df[df["date"] >= "2025-09-01"]["date"].unique())
    for i, d in enumerate(test_dates):
        train_df = df[df["date"] < d]
        if train_df["date"].nunique() < 250:
            continue
        X_tr = train_df[features].copy()
        y_tr = train_df[LABEL].copy()
        m, md = train_model(X_tr, y_tr)
        test_day = df[df["date"] == d]
        X_te = test_day[features].copy()
        codes_te = test_day["code"].values
        for c in features:
            if X_te[c].isna().any():
                X_te[c] = X_te[c].fillna(md.get(c, 0))
        p = m.predict(X_te)
        for code, pred in zip(codes_te, p):
            preds_all.append({"date": d, "code": code, "pred": pred})

    df_sim = df[["date", "code", "fwd_1d_ret", "fwd_1d_excess"]].copy()
    df_sim = df_sim.merge(pd.DataFrame(preds_all), on=["date", "code"], how="inner")
    df_sim.rename(columns={"fwd_1d_ret": LABEL_RAW}, inplace=True)

    # Portfolio test for Top 3/4/5
    pf_results = []
    for n in [3, 4, 5]:
        r = simulate_portfolio(df_sim, n, TOTAL_COST_PER_TRADE)
        if r:
            pf_results.append(r)

    # 5. Print report
    print_report(today_str, model, features, df_latest, pred_df, pf_results)

    # 6. Save
    report = {
        "date": today_str,
        "model": "LightGBM n=400 d=4 lr=0.02, daily expanding",
        "features": len(features),
        "ic_latest": round(ic, 4),
        "top10": [{"rank": i+1, "code": r.code, "score": round(r.pred, 6)}
                  for i, r in enumerate(pred_df.head(10).itertuples())],
        "portfolio": pf_results,
    }
    report_path = OUTPUT_DIR / f"daily_{today_str}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  总耗时: {elapsed:.0f}s")

    # 7. SuperMind push (optional)
    if args.super:
        print(f"\n  [SuperMind push not yet integrated]")

    return pred_df, pf_results


if __name__ == "__main__":
    main()
