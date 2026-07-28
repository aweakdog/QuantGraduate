"""
Walk-Forward v32: 交集换仓 + 5d标签 + 三大改进
改进:
  1. 大盘趋势特征: 等权市场指数 MA/波动率/动量/趋势信号
  2. 隔夜历史特征: 个股隔夜收益率 rolling mean/std/skew
  3. 行业中性化: 用概念分组 demean 标签和部分特征
标签: fwd_5d_ret (close_T → close_T+5)
执行: 收盘买, 收盘卖, 交集换仓, 整数手, 涨跌停, 最低5元佣金
参数: DART n=151 d=4 lr=0.03 n_jobs=10
"""
import pandas as pd, numpy as np, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from collections import defaultdict
warnings.filterwarnings("ignore")
import lightgbm as lgb

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(description="WF v32 enhanced 5d intersection")
parser.add_argument("--test-start", type=str, default="2023-01-01")
parser.add_argument("--test-end", type=str, default="2024-12-31")
parser.add_argument("--initial-capital", type=float, default=100000.0)
parser.add_argument("--top-n", type=int, default=3)
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v24.parquet"
KLINE_DIR = DATA_DIR / "raw" / "kline"

LABEL_RAW = "fwd_1d_ret"
LABEL = "fwd_1d_excess"
LABEL_HORIZON = 1
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
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
TOP_N = args.top_n

_tag = "v32_enhanced_1d_dart"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

_COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open",
            "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount"}

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ═══════════════════════════════════════════════════════════════
# 1. 大盘趋势特征
# ═══════════════════════════════════════════════════════════════
def compute_market_features():
    """从所有 kline 计算等权市场指数, 衍生趋势特征"""
    print("  Computing market index features from klines...")
    all_daily = []
    for p in sorted(KLINE_DIR.glob("*.parquet")):
        try:
            kl = pd.read_parquet(p)
            kl = kl.rename(columns=_COL_MAP)
            kl["date"] = pd.to_datetime(kl["date"])
            all_daily.append(kl[["date", "close", "open"]].copy())
        except Exception:
            continue

    market = pd.concat(all_daily, ignore_index=True)
    # 等权市场收益 (daily close-to-close)
    mkt_daily = market.groupby("date").agg(
        mkt_close=("close", "mean"),
        mkt_open=("open", "mean"),
    ).sort_index()

    # 市场日收益
    mkt_daily["mkt_ret_1d"] = mkt_daily["mkt_close"].pct_change()
    # 隔夜收益 (open / prev_close - 1)
    mkt_daily["mkt_overnight"] = mkt_daily["mkt_open"] / mkt_daily["mkt_close"].shift(1) - 1
    # 日内收益 (close / open - 1)
    mkt_daily["mkt_intraday"] = mkt_daily["mkt_close"] / mkt_daily["mkt_open"] - 1

    # 均线
    mkt_daily["mkt_ma5"] = mkt_daily["mkt_close"].rolling(5).mean()
    mkt_daily["mkt_ma20"] = mkt_daily["mkt_close"].rolling(20).mean()
    mkt_daily["mkt_ma60"] = mkt_daily["mkt_close"].rolling(60).mean()

    # 趋势信号: 价格在均线之上=1, 之下=0
    mkt_daily["mkt_above_ma5"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma5"]).astype(int)
    mkt_daily["mkt_above_ma20"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma20"]).astype(int)
    mkt_daily["mkt_above_ma60"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma60"]).astype(int)

    # 动量
    mkt_daily["mkt_mom_5d"] = mkt_daily["mkt_close"] / mkt_daily["mkt_close"].shift(5) - 1
    mkt_daily["mkt_mom_20d"] = mkt_daily["mkt_close"] / mkt_daily["mkt_close"].shift(20) - 1

    # 波动率
    mkt_daily["mkt_vol_20d"] = mkt_daily["mkt_ret_1d"].rolling(20).std()
    mkt_daily["mkt_vol_5d"] = mkt_daily["mkt_ret_1d"].rolling(5).std()

    # 趋势强度: MA5 / MA20 - 1
    mkt_daily["mkt_trend_strength"] = mkt_daily["mkt_ma5"] / mkt_daily["mkt_ma20"] - 1

    mkt_daily = mkt_daily.reset_index()
    mkt_daily["date"] = pd.to_datetime(mkt_daily["date"])
    mkt_features = [c for c in mkt_daily.columns if c.startswith("mkt_")]
    print(f"    Market features: {len(mkt_features)} ({', '.join(mkt_features[:8])}...)")

    # Replace inf/nan
    for c in mkt_features:
        mkt_daily[c] = mkt_daily[c].replace([np.inf, -np.inf], np.nan)

    return mkt_daily[["date"] + mkt_features], mkt_features

# ═══════════════════════════════════════════════════════════════
# 2. 隔夜历史特征
# ═══════════════════════════════════════════════════════════════
def compute_overnight_features(codes):
    """从 kline 计算个股隔夜收益历史特征"""
    print("  Computing overnight features from klines...")
    all_overnight = []
    for code in codes:
        code6 = str(code)[:6]
        path = KLINE_DIR / f"{code6}.parquet"
        if not path.exists():
            continue
        try:
            kl = pd.read_parquet(path)
            kl = kl.rename(columns=_COL_MAP)
            kl["date"] = pd.to_datetime(kl["date"])
            kl = kl.sort_values("date").reset_index(drop=True)
            # 隔夜收益: open_T / close_{T-1} - 1
            kl["overnight_ret"] = kl["open"] / kl["close"].shift(1) - 1
            # 日内收益: close_T / open_T - 1
            kl["intraday_ret"] = kl["close"] / kl["open"] - 1

            # Rolling 特征
            kl["ovn_mean_5d"] = kl["overnight_ret"].rolling(5).mean()
            kl["ovn_mean_20d"] = kl["overnight_ret"].rolling(20).mean()
            kl["ovn_std_20d"] = kl["overnight_ret"].rolling(20).std()
            kl["ovn_sum_5d"] = kl["overnight_ret"].rolling(5).sum()
            kl["ovn_pos_ratio_20d"] = (kl["overnight_ret"] > 0).rolling(20).mean()

            kl["code"] = code
            feat_cols = ["date", "code", "overnight_ret", "intraday_ret",
                        "ovn_mean_5d", "ovn_mean_20d", "ovn_std_20d",
                        "ovn_sum_5d", "ovn_pos_ratio_20d"]
            all_overnight.append(kl[feat_cols].copy())
        except Exception:
            continue

    ovn_df = pd.concat(all_overnight, ignore_index=True)
    ovn_features = ["overnight_ret", "intraday_ret", "ovn_mean_5d", "ovn_mean_20d",
                    "ovn_std_20d", "ovn_sum_5d", "ovn_pos_ratio_20d"]
    for c in ovn_features:
        ovn_df[c] = ovn_df[c].replace([np.inf, -np.inf], np.nan)
    print(f"    Overnight features: {len(ovn_features)} for {ovn_df['code'].nunique()} stocks")
    return ovn_df, ovn_features

# ═══════════════════════════════════════════════════════════════
# 3. 行业/概念中性化
# ═══════════════════════════════════════════════════════════════
def load_concept_groups():
    """加载概念分组, 用第一个概念作为行业代理"""
    concept_path = DATA_DIR / "universe" / "concept_stock_map.json"
    concept_data = json.loads(concept_path.read_text())
    stock_to_concepts = concept_data.get("stock_to_concepts", {})
    # 用第一个概念作为行业代理
    code_to_group = {}
    for code, concepts in stock_to_concepts.items():
        if concepts:
            code_to_group[str(code)[:6]] = concepts[0]
    print(f"    Concept groups: {len(set(code_to_group.values()))} groups, {len(code_to_group)} stocks")
    return code_to_group

def neutralize_by_group(df, col, group_col="group", date_col="date"):
    """按日期和分组对列做 demean (中性化)"""
    return df.groupby(date_col)[col].transform(
        lambda x: x - x.mean()
    ) if group_col not in df.columns else df.groupby([date_col, group_col])[col].transform(
        lambda x: x - x.mean() if len(x) > 1 else x - x.mean()
    )

# ═══════════════════════════════════════════════════════════════
# Kline helpers for execution
# ═══════════════════════════════════════════════════════════════
def load_all_klines():
    cache = {}
    for p in sorted(KLINE_DIR.glob("*.parquet")):
        code6 = p.stem
        kl = pd.read_parquet(p)
        kl = kl.rename(columns=_COL_MAP)
        kl["date"] = pd.to_datetime(kl["date"])
        kl = kl.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        cache[code6] = kl
    return cache

def get_close(klines, code, date):
    code6 = str(code)[:6]
    if code6 not in klines:
        return None
    kl = klines[code6]
    row = kl[kl["date"] == pd.Timestamp(date)]
    if len(row) == 0:
        return None
    return float(row.iloc[0]["close"])

def is_limit_up(klines, code, date):
    code6 = str(code)[:6]
    if code6 not in klines:
        return False
    kl = klines[code6]
    idx = kl.index[kl["date"] == pd.Timestamp(date)]
    if len(idx) == 0:
        return False
    pos = idx[0]
    if pos < 1:
        return False
    r = kl.iloc[pos]
    prev_close = kl.iloc[pos - 1]["close"]
    limit_price = prev_close * 1.1
    return float(r["close"]) >= limit_price * 0.999

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
print(f"Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[LABEL_RAW])

# ── Add concept group ──
print("Adding concept groups for neutralization...")
code_to_group = load_concept_groups()
df["group"] = df["code"].map(lambda c: code_to_group.get(str(c)[:6], "unknown"))

# ── Compute and merge market features ──
print("Computing enhanced features...")
mkt_df, mkt_features = compute_market_features()
df = df.merge(mkt_df, on="date", how="left")
for c in mkt_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ── Compute and merge overnight features ──
all_codes = df["code"].unique()
ovn_df, ovn_features = compute_overnight_features(all_codes)
df = df.merge(ovn_df[["date", "code"] + ovn_features], on=["date", "code"], how="left")

# ── Compute label: industry-neutralized excess return ──
# demean by date+group instead of just date
df[LABEL] = df.groupby(["date", "group"])[LABEL_RAW].transform(lambda x: x - x.mean() if len(x) > 1 else x - x.groupby(df.loc[x.index, "date"]).transform("mean"))

# ── Feature selection ──
all_cols = [c for c in df.columns if c not in SKIP_COLS and c not in EXCLUDED_FEATS and is_valid_feat(c)]
features = [f for f in all_cols if f not in LEAKAGE_FEATS and f not in {"group"}]
print(f"  {len(df)} rows, {df['code'].nunique()} codes, {len(features)} features (base + {len(mkt_features)} market + {len(ovn_features)} overnight)")
print(f"  Dates: {df['date'].min().date()} ~ {df['date'].max().date()}")

_dmask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END is not None:
    _dmask &= df["date"] <= pd.Timestamp(TEST_END)
dates = sorted(df[_dmask]["date"].unique())
if not dates:
    print(f"ERROR: No data after {TEST_START}")
    exit(1)

MIN_TRAIN_DAYS = 250
print(f"\nWalk-Forward: {len(dates)} prediction days ({dates[0].date()} ~ {dates[-1].date()})")
print(f"Strategy: intersection rebalance, top{TOP_N}, 5d close-to-close, enhanced features")

# ── Walk-forward training ──
daily_preds = []
t0 = datetime.now()

for day_idx, pred_date in enumerate(dates):
    # Exclude last LABEL_HORIZON days to prevent label leakage
    cutoff_idx = max(day_idx - LABEL_HORIZON, 0)
    cutoff_date = dates[cutoff_idx] if cutoff_idx > 0 else pred_date
    train_mask = df["date"] < cutoff_date
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

    test_df = pd.DataFrame({"code": codes_test, "pred": preds, "label": y_test.values})
    top = test_df.nlargest(TOP_N, "pred")
    top_codes = list(top["code"].values)

    daily_preds.append({
        "date": pred_date,
        "top_codes": top_codes,
        "ic": ic,
        "n_train": len(X_train),
    })

    if day_idx % 50 == 0 or day_idx == len(dates) - 1:
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  [{day_idx+1}/{len(dates)}] {pred_date.date()} IC={ic:.4f} ({elapsed:.0f}s)")

print(f"\nTraining done in {(datetime.now()-t0).total_seconds():.0f}s")
print(f"Loading kline data for execution...")
klines = load_all_klines()
print(f"  Loaded {len(klines)} kline files")

# ── Intersection rebalance simulation ──
print(f"\n[exec] Simulating intersection rebalance portfolio...")

capital = INIT_CAPITAL
holdings = {}
daily_records = []
trade_log = []
total_buy_cost = 0.0
total_sell_cost = 0.0
rejected_buy = 0
rejected_sell = 0
n_rebalances = 0
n_full_turnover = 0
n_partial_turnover = 0
n_no_change = 0

for i, dp in enumerate(daily_preds):
    pred_date = dp["date"]
    new_top = set(str(c) for c in dp["top_codes"])
    current_holdings = set(holdings.keys())

    to_sell = current_holdings - new_top
    to_buy = new_top - current_holdings
    to_keep = current_holdings & new_top

    if len(to_sell) == 0 and len(to_buy) == 0:
        n_no_change += 1
    elif len(to_sell) == len(current_holdings) and len(to_buy) == len(new_top):
        n_full_turnover += 1
    else:
        n_partial_turnover += 1
    n_rebalances += 1

    # Sell first
    sell_proceeds = 0.0
    sell_cost_total = 0.0
    for code in list(to_sell):
        h = holdings[code]
        close_price = get_close(klines, code, pred_date)
        if close_price is None:
            rejected_sell += 1
            continue
        gross = h["shares"] * close_price
        fee = max(gross * TRADE_COST, 5.0)
        net = gross - fee
        sell_proceeds += net
        sell_cost_total += fee
        trade_log.append({
            "date": str(pred_date.date()), "code": code, "action": "sell",
            "shares": h["shares"], "price": close_price, "gross": gross,
            "fee": fee, "net": net, "reason": "not_in_top",
        })
        del holdings[code]

    # Buy with available cash
    available_capital = capital + sell_proceeds
    n_to_buy = len(to_buy)

    if n_to_buy > 0:
        allocation_per_stock = available_capital / n_to_buy
        buy_cost_total = 0.0
        for code in to_buy:
            close_price = get_close(klines, code, pred_date)
            if close_price is None:
                rejected_buy += 1
                continue
            if is_limit_up(klines, code, pred_date):
                rejected_buy += 1
                trade_log.append({
                    "date": str(pred_date.date()), "code": code, "action": "reject_buy",
                    "shares": 0, "price": close_price, "gross": 0, "fee": 0, "net": 0,
                    "reason": "limit_up",
                })
                continue
            lot_cost = close_price * 100
            if lot_cost > allocation_per_stock:
                rejected_buy += 1
                continue
            shares = int(allocation_per_stock / lot_cost) * 100
            if shares <= 0:
                rejected_buy += 1
                continue
            gross = shares * close_price
            fee = max(gross * TRADE_COST, 5.0)
            total_deducted = gross + fee
            available_capital -= total_deducted
            buy_cost_total += fee
            holdings[code] = {"shares": shares, "buy_price": close_price, "buy_date": pred_date}
            trade_log.append({
                "date": str(pred_date.date()), "code": code, "action": "buy",
                "shares": shares, "price": close_price, "gross": gross,
                "fee": fee, "net": -total_deducted, "reason": "new_top",
            })
    else:
        buy_cost_total = 0.0

    capital = available_capital

    # Portfolio value
    portfolio_value = capital
    for code, h in holdings.items():
        cp = get_close(klines, code, pred_date)
        if cp is not None:
            portfolio_value += h["shares"] * cp
        else:
            portfolio_value += h["shares"] * h["buy_price"]

    if i > 0:
        prev_value = daily_records[-1]["portfolio_value"]
        daily_ret = portfolio_value / prev_value - 1 if prev_value > 0 else 0.0
    else:
        daily_ret = 0.0

    total_buy_cost += buy_cost_total
    total_sell_cost += sell_cost_total

    daily_records.append({
        "date": str(pred_date.date()),
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(capital, 2),
        "daily_ret": round(daily_ret, 6),
        "n_holdings": len(holdings),
        "holdings": list(holdings.keys()),
        "to_sell": list(to_sell),
        "to_buy": list(to_buy),
        "to_keep": list(to_keep),
        "sell_cost": round(sell_cost_total, 2),
        "buy_cost": round(buy_cost_total, 2),
        "ic": round(dp["ic"], 4) if not np.isnan(dp["ic"]) else None,
    })

# Force liquidate
last_date = daily_preds[-1]["date"]
final_value = capital
for code, h in holdings.items():
    cp = get_close(klines, code, last_date)
    if cp is not None:
        gross = h["shares"] * cp
        fee = max(gross * TRADE_COST, 5.0)
        final_value += gross - fee
        trade_log.append({
            "date": str(last_date.date()), "code": code, "action": "force_sell",
            "shares": h["shares"], "price": cp, "gross": gross,
            "fee": fee, "net": gross - fee, "reason": "end_of_backtest",
        })
    else:
        final_value += h["shares"] * h["buy_price"]

# ── Metrics ──
rdf = pd.DataFrame(daily_records)
rdf["date"] = pd.to_datetime(rdf["date"])
n_days = len(rdf)

rdf_s = rdf.iloc[::5]
sharpe = float(rdf_s["daily_ret"].mean() / rdf_s["daily_ret"].std() * np.sqrt(252/5)) if rdf_s["daily_ret"].std() > 0 else 0
sharpe_raw = float(rdf["daily_ret"].mean() / rdf["daily_ret"].std() * np.sqrt(252)) if rdf["daily_ret"].std() > 0 else 0

cum_series = (1 + rdf["daily_ret"]).cumprod()
max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
win_rate = float((rdf["daily_ret"] > 0).mean())
ann_ret = float((1 + rdf["daily_ret"]).prod() ** (252/n_days) - 1) if n_days > 0 else 0
total_return = (final_value / INIT_CAPITAL - 1) * 100
total_cost = (total_buy_cost + total_sell_cost) / INIT_CAPITAL * 100

ic_vals = [r["ic"] for r in daily_records if r["ic"] is not None]
ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
ic_std = float(np.std(ic_vals)) if ic_vals else 0.0

avg_holdings = float(rdf["n_holdings"].mean())
avg_turnover = float((rdf["to_sell"].apply(len) + rdf["to_buy"].apply(len)).mean())

print(f"\n{'='*60}")
print(f"  v32 Enhanced 5d Intersection DART Results")
print(f"{'='*60}")
print(f"  Days: {n_days}")
print(f"  IC: mean={ic_mean:.4f} std={ic_std:.4f}")
print(f"  Portfolio: ¥{final_value:,.0f} (from ¥{INIT_CAPITAL:,})")
print(f"  Total return: {total_return:+.1f}%")
print(f"  Annualized: {ann_ret*100:+.1f}%")
print(f"  Sharpe: {sharpe:.2f} (raw={sharpe_raw:.2f})")
print(f"  Max DD: {max_dd*100:.1f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Total cost: {total_cost:.1f}% of capital")
print(f"  Rebalances: {n_rebalances} (full={n_full_turnover} partial={n_partial_turnover} no_change={n_no_change})")
print(f"  Avg holdings: {avg_holdings:.1f} stocks")
print(f"  Avg turnover: {avg_turnover:.1f} stocks/day")
print(f"  Rejected: buy={rejected_buy} sell={rejected_sell}")
print(f"  Trades: {len(trade_log)}")
print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

# ── Save JSON ──
output = {
    "label": LABEL_RAW,
    "model": f"LightGBM DART n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
             f"lr={LOCKED_PARAMS['learning_rate']}, enhanced (market+overnight+neutral), intersection rebalance",
    "features": len(features),
    "enhancements": {
        "market_features": mkt_features,
        "overnight_features": ovn_features,
        "neutralization": "concept_group_demean",
    },
    "period": f"{rdf['date'].iloc[0].strftime('%Y-%m-%d')} ~ {rdf['date'].iloc[-1].strftime('%Y-%m-%d')}",
    "n_days": n_days,
    "initial_capital": INIT_CAPITAL,
    "top_n": TOP_N,
    "summary": {
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 1),
        "annualized_return_pct": round(ann_ret*100, 1),
        "sharpe": round(sharpe, 2),
        "sharpe_raw": round(sharpe_raw, 2),
        "max_dd_pct": round(max_dd*100, 1),
        "win_rate_pct": round(win_rate*100, 1),
        "total_cost_pct": round(total_cost, 1),
        "n_rebalances": n_rebalances,
        "n_full_turnover": n_full_turnover,
        "n_partial_turnover": n_partial_turnover,
        "n_no_change": n_no_change,
        "avg_holdings": round(avg_holdings, 1),
        "avg_turnover": round(avg_turnover, 1),
        "rejected_buy": rejected_buy,
        "rejected_sell": rejected_sell,
        "n_trades": len(trade_log),
    },
    "daily": daily_records,
    "trades": trade_log,
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved: {OUT_PATH}")
