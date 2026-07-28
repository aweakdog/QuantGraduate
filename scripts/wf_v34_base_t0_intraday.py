"""
Walk-Forward v34: 底仓 T+0 日内策略
核心思路: 提前建底仓实现合法 T+0, 捕获日内收益(open->close), 避开隔夜负收益

流程:
  收盘 T-1: 根据模型预测买入 top-K 股票作为底仓 (K = 资金能覆盖的低价股数)
  开盘 T:   卖掉非 top3 的底仓 (T+1 合法), 保留 top3
  日内 T:   top3 底仓持有, 赚 open->close 日内收益
  收盘 T:   卖掉 top3 底仓 (T+1 合法), 用当天新预测重建底仓

标签: fwd_1d_exec_ret (open_T -> close_T, 纯日内收益)
训练: 全部 216 只股票
执行: 只在底仓子集内选 top3, 底仓每天根据预测动态调整
特征: v33 的特征筛选 (importance + corr dedup)
择时: IC 置信度 + 波动率目标仓位
"""
import pandas as pd, numpy as np, json, warnings, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(description="WF v34 base position T+0 intraday")
parser.add_argument("--test-start", type=str, default="2022-09-01")
parser.add_argument("--test-end", type=str, default="2026-07-16")
parser.add_argument("--initial-capital", type=float, default=100000.0)
parser.add_argument("--top-n", type=int, default=3)
parser.add_argument("--n-features", type=int, default=80)
parser.add_argument("--corr-threshold", type=float, default=0.9)
parser.add_argument("--target-vol", type=float, default=0.15)
parser.add_argument("--ic-window", type=int, default=10)
parser.add_argument("--ic-bad-days", type=int, default=3)
parser.add_argument("--base-ratio", type=float, default=0.7, help="fraction of capital for base positions")
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v24.parquet"
KLINE_DIR = DATA_DIR / "raw" / "kline"

LABEL_RAW = "fwd_1d_exec_ret"  # open -> close, intraday return
LABEL = "fwd_1d_exec_excess"
LABEL_HORIZON = 1
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret",
             "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
             "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret", "fwd_1d_exec_excess"}
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
N_FEATURES = args.n_features
CORR_THRESHOLD = args.corr_threshold
TARGET_VOL = args.target_vol
IC_WINDOW = args.ic_window
IC_BAD_DAYS = args.ic_bad_days
BASE_RATIO = args.base_ratio

_tag = "v34_base_t0_intraday"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

_COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open",
            "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount"}

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ═══════════════════════════════════════════════════════════════
# Market features
# ═══════════════════════════════════════════════════════════════
def compute_market_features():
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
    mkt_daily = market.groupby("date").agg(
        mkt_close=("close", "mean"), mkt_open=("open", "mean"),
    ).sort_index()
    mkt_daily["mkt_ret_1d"] = mkt_daily["mkt_close"].pct_change()
    mkt_daily["mkt_overnight"] = mkt_daily["mkt_open"] / mkt_daily["mkt_close"].shift(1) - 1
    mkt_daily["mkt_intraday"] = mkt_daily["mkt_close"] / mkt_daily["mkt_open"] - 1
    mkt_daily["mkt_ma5"] = mkt_daily["mkt_close"].rolling(5).mean()
    mkt_daily["mkt_ma20"] = mkt_daily["mkt_close"].rolling(20).mean()
    mkt_daily["mkt_ma60"] = mkt_daily["mkt_close"].rolling(60).mean()
    mkt_daily["mkt_above_ma5"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma5"]).astype(int)
    mkt_daily["mkt_above_ma20"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma20"]).astype(int)
    mkt_daily["mkt_above_ma60"] = (mkt_daily["mkt_close"] > mkt_daily["mkt_ma60"]).astype(int)
    mkt_daily["mkt_mom_5d"] = mkt_daily["mkt_close"] / mkt_daily["mkt_close"].shift(5) - 1
    mkt_daily["mkt_mom_20d"] = mkt_daily["mkt_close"] / mkt_daily["mkt_close"].shift(20) - 1
    mkt_daily["mkt_vol_20d"] = mkt_daily["mkt_ret_1d"].rolling(20).std()
    mkt_daily["mkt_vol_5d"] = mkt_daily["mkt_ret_1d"].rolling(5).std()
    mkt_daily["mkt_trend_strength"] = mkt_daily["mkt_ma5"] / mkt_daily["mkt_ma20"] - 1
    mkt_daily = mkt_daily.reset_index()
    mkt_daily["date"] = pd.to_datetime(mkt_daily["date"])
    mkt_features = [c for c in mkt_daily.columns if c.startswith("mkt_")]
    for c in mkt_features:
        mkt_daily[c] = mkt_daily[c].replace([np.inf, -np.inf], np.nan)
    print(f"    Market features: {len(mkt_features)}")
    return mkt_daily[["date"] + mkt_features], mkt_features

# ═══════════════════════════════════════════════════════════════
# Overnight features
# ═══════════════════════════════════════════════════════════════
def compute_overnight_features(codes):
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
            kl["overnight_ret"] = kl["open"] / kl["close"].shift(1) - 1
            kl["intraday_ret"] = kl["close"] / kl["open"] - 1
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
# Concept groups
# ═══════════════════════════════════════════════════════════════
def load_concept_groups():
    concept_path = DATA_DIR / "universe" / "concept_stock_map.json"
    concept_data = json.loads(concept_path.read_text())
    stock_to_concepts = concept_data.get("stock_to_concepts", {})
    code_to_group = {}
    for code, concepts in stock_to_concepts.items():
        if concepts:
            code_to_group[str(code)[:6]] = concepts[0]
    print(f"    Concept groups: {len(set(code_to_group.values()))} groups, {len(code_to_group)} stocks")
    return code_to_group

# ═══════════════════════════════════════════════════════════════
# Feature selection
# ═══════════════════════════════════════════════════════════════
def select_features(df, all_features, label_col, n_top=80, corr_thresh=0.9):
    print(f"  Feature selection: {len(all_features)} -> top {n_top} -> corr dedup ({corr_thresh})...")
    sample_mask = df["date"] < pd.Timestamp(TEST_START)
    sample_df = df[sample_mask].copy()
    if len(sample_df) < 10000:
        sample_df = df.copy()
    X_sample = sample_df.groupby("code")[all_features].transform(lambda s: s.ffill().fillna(0))
    y_sample = sample_df[label_col].copy()
    sel_params = dict(LOCKED_PARAMS)
    sel_params["n_estimators"] = 50
    sel_params["boosting_type"] = "gbdt"
    sel_model = lgb.LGBMRegressor(**sel_params)
    sel_model.fit(X_sample, y_sample)
    imp = pd.DataFrame({
        "feature": all_features,
        "importance": sel_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    top_feats = imp.head(n_top)["feature"].tolist()
    corr_matrix = sample_df[top_feats].corr().abs()
    selected = []
    dropped = set()
    for feat in top_feats:
        if feat in dropped:
            continue
        selected.append(feat)
        correlated = corr_matrix.index[corr_matrix[feat] > corr_thresh].tolist()
        for c in correlated:
            if c != feat and c not in dropped:
                dropped.add(c)
    print(f"    Selected: {len(selected)} features (dropped {len(top_feats) - len(selected)} by corr)")
    print(f"    Top 10: {imp.head(10)['feature'].tolist()}")
    return selected, imp

# ═══════════════════════════════════════════════════════════════
# Kline helpers
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

def get_ohlc(klines, code, date):
    """Return (open, close) for a given code and date."""
    code6 = str(code)[:6]
    if code6 not in klines:
        return None, None
    kl = klines[code6]
    row = kl[kl["date"] == pd.Timestamp(date)]
    if len(row) == 0:
        return None, None
    return float(row.iloc[0]["open"]), float(row.iloc[0]["close"])

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
    prev_close = kl.iloc[pos - 1]["close"]
    limit_price = prev_close * 1.1
    return float(kl.iloc[pos]["close"]) >= limit_price * 0.999

def is_limit_down_open(klines, code, date):
    """Check if open price is at limit down (can't sell)."""
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
    prev_close = kl.iloc[pos - 1]["close"]
    limit_down_price = prev_close * 0.9
    return float(kl.iloc[pos]["open"]) <= limit_down_price * 1.001

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
print(f"Loading {TRAIN_PATH.name}...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

# Check if fwd_1d_exec_ret exists
if LABEL_RAW not in df.columns:
    print(f"ERROR: {LABEL_RAW} not in training data. Available fwd columns:")
    print([c for c in df.columns if c.startswith("fwd_")])
    exit(1)

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

# ── Compute label: industry-neutralized excess intraday return ──
df[LABEL] = df.groupby(["date", "group"])[LABEL_RAW].transform(
    lambda x: x - x.mean() if len(x) > 1 else x - x.groupby(df.loc[x.index, "date"]).transform("mean")
)

# ── Feature selection ──
all_cols = [c for c in df.columns if c not in SKIP_COLS and c not in EXCLUDED_FEATS and is_valid_feat(c)]
all_features = [f for f in all_cols if f not in LEAKAGE_FEATS and f not in {"group"}]
print(f"  {len(df)} rows, {df['code'].nunique()} codes, {len(all_features)} candidate features")
selected_features, importance_df = select_features(df, all_features, LABEL, N_FEATURES, CORR_THRESHOLD)
features = selected_features
print(f"  Final features: {len(features)}")

# ── Market vol for position sizing ──
mkt_vol_series = mkt_df.set_index("date")["mkt_vol_20d"].copy()
mkt_vol_annual = mkt_vol_series * np.sqrt(252)
mkt_position = (TARGET_VOL / mkt_vol_annual).clip(0.3, 1.0)

_dmask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END is not None:
    _dmask &= df["date"] <= pd.Timestamp(TEST_END)
dates = sorted(df[_dmask]["date"].unique())
if not dates:
    print(f"ERROR: No data after {TEST_START}")
    exit(1)

MIN_TRAIN_DAYS = 250
print(f"\nWalk-Forward: {len(dates)} prediction days ({dates[0].date()} ~ {dates[-1].date()})")
print(f"Strategy: base position T+0 intraday, top{TOP_N}, {len(features)} features")
print(f"Label: {LABEL_RAW} (open->close intraday)")
print(f"Timing: IC window={IC_WINDOW}, bad days={IC_BAD_DAYS}, target vol={TARGET_VOL*100:.0f}%")
print(f"Base ratio: {BASE_RATIO*100:.0f}% of capital for base positions")

# ── Walk-forward training ──
daily_preds = []
t0 = datetime.now()

for day_idx, pred_date in enumerate(dates):
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

    # Full ranking for base position selection
    test_df = pd.DataFrame({"code": codes_test, "pred": preds, "label": y_test.values})
    ranked = test_df.sort_values("pred", ascending=False)

    daily_preds.append({
        "date": pred_date,
        "ranked_codes": list(ranked["code"].values),
        "top_codes": list(ranked.head(TOP_N)["code"].values),
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

# ── Base position T+0 simulation ──
print(f"\n[exec] Simulating base position T+0 intraday portfolio...")

base_capital = INIT_CAPITAL * BASE_RATIO
trade_capital = INIT_CAPITAL * (1 - BASE_RATIO)

# State
capital = INIT_CAPITAL
base_holdings = {}  # code -> {"shares": int, "buy_price": float, "buy_date": date}
daily_records = []
trade_log = []
total_buy_cost = 0.0
total_sell_cost = 0.0
rejected_buy = 0
rejected_sell = 0
n_cash_days = 0
n_top3_in_base = 0
n_top3_total = 0

ic_history = []
in_cash_mode = False

def sell_stock(code, shares, price, date_str, reason):
    """Sell shares and return net proceeds."""
    gross = shares * price
    fee = max(gross * TRADE_COST, 5.0)
    net = gross - fee
    trade_log.append({
        "date": date_str, "code": code, "action": "sell",
        "shares": shares, "price": price, "gross": gross,
        "fee": fee, "net": net, "reason": reason,
    })
    return net, fee

def buy_stock(code, shares, price, date_str, reason):
    """Buy shares and return total cost."""
    gross = shares * price
    fee = max(gross * TRADE_COST, 5.0)
    total = gross + fee
    trade_log.append({
        "date": date_str, "code": code, "action": "buy",
        "shares": shares, "price": price, "gross": gross,
        "fee": fee, "net": -total, "reason": reason,
    })
    return total, fee

for i, dp in enumerate(daily_preds):
    pred_date = dp["date"]
    date_str = str(pred_date.date())
    ic_val = dp["ic"]
    ranked_codes = dp["ranked_codes"]

    # ── Model confidence timing ──
    if not np.isnan(ic_val):
        ic_history.append(ic_val)
    if len(ic_history) >= IC_WINDOW:
        recent_ic = ic_history[-IC_WINDOW:]
        if len(recent_ic) >= IC_BAD_DAYS:
            last_bad = all(ic < 0 for ic in recent_ic[-IC_BAD_DAYS:])
            if last_bad:
                in_cash_mode = True
            if in_cash_mode and np.mean(recent_ic) > 0:
                in_cash_mode = False

    # ── Volatility target position sizing ──
    pos_size = 1.0
    if pred_date in mkt_position.index:
        pos_size = float(mkt_position.loc[pred_date])
        if np.isnan(pos_size):
            pos_size = 1.0

    # ── Get top3 that are in base holdings ──
    current_base = set(base_holdings.keys())
    top3_all = [str(c) for c in dp["top_codes"]]
    top3_in_base = [c for c in top3_all if c in current_base]
    n_top3_total += len(top3_all)
    n_top3_in_base += len(top3_in_base)

    if in_cash_mode:
        n_cash_days += 1
        # Sell all base holdings at open
        sell_proceeds = 0.0
        sell_cost = 0.0
        for code in list(base_holdings.keys()):
            h = base_holdings[code]
            open_price, _ = get_ohlc(klines, code, pred_date)
            if open_price is None:
                continue
            if is_limit_down_open(klines, code, pred_date):
                rejected_sell += 1
                continue
            net, fee = sell_stock(code, h["shares"], open_price, date_str, "cash_mode_sell")
            sell_proceeds += net
            sell_cost += fee
            del base_holdings[code]
        capital += sell_proceeds
        total_sell_cost += sell_cost

        # Rebuild base at close (minimal, just to have positions for tomorrow)
        # Actually in cash mode, we don't rebuild - stay in cash
        portfolio_value = capital
        if i > 0:
            prev_value = daily_records[-1]["portfolio_value"]
            daily_ret = portfolio_value / prev_value - 1 if prev_value > 0 else 0.0
        else:
            daily_ret = 0.0
        daily_records.append({
            "date": date_str, "portfolio_value": round(portfolio_value, 2),
            "cash": round(capital, 2), "daily_ret": round(daily_ret, 6),
            "n_holdings": 0, "holdings": [],
            "top3_all": top3_all, "top3_in_base": [],
            "sell_cost": round(sell_cost, 2), "buy_cost": 0.0,
            "ic": round(ic_val, 4) if not np.isnan(ic_val) else None,
            "pos_size": 0.0, "in_cash": True,
        })
        continue

    # ═══════════════════════════════════════════════════════════
    # Step 1: Open - Sell non-top3 base holdings (T+1 legal)
    # ═══════════════════════════════════════════════════════════
    sell_proceeds = 0.0
    sell_cost_total = 0.0
    for code in list(base_holdings.keys()):
        if code not in top3_in_base:
            h = base_holdings[code]
            open_price, _ = get_ohlc(klines, code, pred_date)
            if open_price is None:
                rejected_sell += 1
                continue
            if is_limit_down_open(klines, code, pred_date):
                rejected_sell += 1
                continue
            net, fee = sell_stock(code, h["shares"], open_price, date_str, "open_sell_non_top")
            sell_proceeds += net
            sell_cost_total += fee
            del base_holdings[code]

    # ═══════════════════════════════════════════════════════════
    # Step 2: Intraday - top3 base holdings earn open->close
    # (Nothing to do, just hold)
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # Step 3: Close - Sell top3 base holdings (T+1 legal)
    # ═══════════════════════════════════════════════════════════
    for code in list(base_holdings.keys()):
        h = base_holdings[code]
        _, close_price = get_ohlc(klines, code, pred_date)
        if close_price is None:
            # Can't sell, keep overnight (unwanted but no choice)
            continue
        net, fee = sell_stock(code, h["shares"], close_price, date_str, "close_sell_top")
        sell_proceeds += net
        sell_cost_total += fee
        del base_holdings[code]

    capital += sell_proceeds
    total_sell_cost += sell_cost_total

    # ═══════════════════════════════════════════════════════════
    # Step 4: Close - Rebuild base positions for tomorrow
    # Use model's full ranking, buy top-K that we can afford
    # ═══════════════════════════════════════════════════════════
    available_capital = capital * pos_size
    buy_cost_total = 0.0

    for code in ranked_codes:
        code = str(code)
        _, close_price = get_ohlc(klines, code, pred_date)
        if close_price is None:
            continue
        if is_limit_up(klines, code, pred_date):
            rejected_buy += 1
            continue
        lot_cost = close_price * 100
        if lot_cost > available_capital:
            continue  # Can't afford even 1 lot
        shares = int(available_capital / lot_cost) * 100
        if shares <= 0:
            continue
        # Buy 1 lot per stock for base (spread capital across many stocks)
        shares = 100  # Just 1 lot for base position
        gross = shares * close_price
        fee = max(gross * TRADE_COST, 5.0)
        total_deducted = gross + fee
        if total_deducted > available_capital:
            continue
        available_capital -= total_deducted
        capital -= total_deducted
        buy_cost_total += fee
        base_holdings[code] = {"shares": shares, "buy_price": close_price, "buy_date": pred_date}
        trade_log.append({
            "date": date_str, "code": code, "action": "buy_base",
            "shares": shares, "price": close_price, "gross": gross,
            "fee": fee, "net": -total_deducted, "reason": "rebuild_base",
        })

    total_buy_cost += buy_cost_total

    # ── Portfolio value at close ──
    portfolio_value = capital
    for code, h in base_holdings.items():
        _, cp = get_ohlc(klines, code, pred_date)
        if cp is not None:
            portfolio_value += h["shares"] * cp
        else:
            portfolio_value += h["shares"] * h["buy_price"]

    if i > 0:
        prev_value = daily_records[-1]["portfolio_value"]
        daily_ret = portfolio_value / prev_value - 1 if prev_value > 0 else 0.0
    else:
        daily_ret = 0.0

    daily_records.append({
        "date": date_str, "portfolio_value": round(portfolio_value, 2),
        "cash": round(capital, 2), "daily_ret": round(daily_ret, 6),
        "n_holdings": len(base_holdings), "holdings": list(base_holdings.keys()),
        "top3_all": top3_all, "top3_in_base": top3_in_base,
        "sell_cost": round(sell_cost_total, 2), "buy_cost": round(buy_cost_total, 2),
        "ic": round(ic_val, 4) if not np.isnan(ic_val) else None,
        "pos_size": round(pos_size, 3), "in_cash": False,
    })

# Force liquidate at last day
last_date = daily_preds[-1]["date"]
final_value = capital
for code, h in base_holdings.items():
    _, cp = get_ohlc(klines, code, last_date)
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

sharpe_raw = float(rdf["daily_ret"].mean() / rdf["daily_ret"].std() * np.sqrt(252)) if rdf["daily_ret"].std() > 0 else 0
cum_series = (1 + rdf["daily_ret"]).cumprod()
max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
win_rate = float((rdf["daily_ret"] > 0).mean())
ann_ret = float((1 + rdf["daily_ret"]).prod() ** (252 / n_days) - 1) if n_days > 0 else 0
total_return = (final_value / INIT_CAPITAL - 1) * 100
total_cost = (total_buy_cost + total_sell_cost) / INIT_CAPITAL * 100

ic_vals = [r["ic"] for r in daily_records if r["ic"] is not None]
ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
ic_std = float(np.std(ic_vals)) if ic_vals else 0.0

avg_holdings = float(rdf["n_holdings"].mean())
avg_pos_size = float(rdf["pos_size"].mean())
top3_hit_rate = n_top3_in_base / n_top3_total * 100 if n_top3_total > 0 else 0

print(f"\n{'='*60}")
print(f"  v34 Base Position T+0 Intraday Results")
print(f"{'='*60}")
print(f"  Days: {n_days} (cash: {n_cash_days}, invested: {n_days - n_cash_days})")
print(f"  Features: {len(features)} (from {len(all_features)})")
print(f"  Label: {LABEL_RAW} (open->close intraday)")
print(f"  IC: mean={ic_mean:.4f} std={ic_std:.4f}")
print(f"  Portfolio: ¥{final_value:,.0f} (from ¥{INIT_CAPITAL:,})")
print(f"  Total return: {total_return:+.1f}%")
print(f"  Annualized: {ann_ret*100:+.1f}%")
print(f"  Sharpe (raw): {sharpe_raw:.2f}")
print(f"  Max DD: {max_dd*100:.1f}%")
print(f"  Win rate: {win_rate*100:.1f}%")
print(f"  Total cost: {total_cost:.1f}% of capital")
print(f"  Avg position size: {avg_pos_size:.2f}")
print(f"  Avg base holdings: {avg_holdings:.1f} stocks")
print(f"  Top3 in base hit rate: {top3_hit_rate:.1f}% ({n_top3_in_base}/{n_top3_total})")
print(f"  Cash mode days: {n_cash_days} ({n_cash_days/n_days*100:.1f}%)")
print(f"  Rejected: buy={rejected_buy} sell={rejected_sell}")
print(f"  Trades: {len(trade_log)}")
print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

# ── Save JSON ──
output = {
    "label": LABEL_RAW,
    "model": f"LightGBM DART n={LOCKED_PARAMS['n_estimators']} d={LOCKED_PARAMS['max_depth']} "
             f"lr={LOCKED_PARAMS['learning_rate']}, base position T+0 intraday",
    "features": len(features),
    "selected_features": features,
    "strategy": "base_position_t0_intraday",
    "enhancements": {
        "feature_selection": f"importance top {N_FEATURES} + corr dedup {CORR_THRESHOLD}",
        "timing": f"IC window={IC_WINDOW}, bad days={IC_BAD_DAYS}",
        "position_sizing": f"target vol={TARGET_VOL*100:.0f}%, clamp [0.3, 1.0]",
        "base_ratio": BASE_RATIO,
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
        "annualized_return_pct": round(ann_ret * 100, 1),
        "sharpe_raw": round(sharpe_raw, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_cost_pct": round(total_cost, 1),
        "avg_pos_size": round(avg_pos_size, 3),
        "avg_base_holdings": round(avg_holdings, 1),
        "top3_hit_rate_pct": round(top3_hit_rate, 1),
        "n_cash_days": n_cash_days,
        "cash_pct": round(n_cash_days / n_days * 100, 1),
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
