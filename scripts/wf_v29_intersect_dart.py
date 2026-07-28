"""
Walk-Forward v29: 交集换仓策略
标签: fwd_1d_exec_ret (日内收益, 用于选股)
执行: 收盘买, 收盘卖, T+1 制度
逻辑:
  - T-1 日训练, 预测 T 日 Top3, T 日收盘买入
  - T 日训练, 预测 T+1 日 Top3
  - T+1 日收盘: 卖掉不在新 Top3 的持仓, 买入新进 Top3 的股票
  - 交集内的股票继续持有, 不买卖
  - 持仓周期由市场决定 (可能 1 天也可能很多天)
参数: n=151, d=4, lr=0.03, n_jobs=10, boosting_type=dart
约束: 整数手, 涨跌停限制, 最低5元佣金
"""
import pandas as pd, numpy as np, json, warnings, argparse, math
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
import lightgbm as lgb

rng = np.random.default_rng(42)

parser = argparse.ArgumentParser(description="WF v29 intersection rebalance")
parser.add_argument("--test-start", type=str, default="2023-01-01")
parser.add_argument("--test-end", type=str, default="2024-12-31")
parser.add_argument("--initial-capital", type=float, default=100000.0)
parser.add_argument("--top-n", type=int, default=3)
args = parser.parse_args()

from pipeline.config import settings
DATA_DIR = settings.DATA_DIR
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v24.parquet"
KLINE_DIR = DATA_DIR / "raw" / "kline"

LABEL_RAW = "fwd_5d_ret"
LABEL = "fwd_5d_excess"
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

_tag = "v31_intersect_5d_dart"
_out = f"wf_daily_{_tag}_ts{TEST_START}_te{TEST_END}_cap{int(INIT_CAPITAL)}"
OUT_PATH = DATA_DIR / "processed" / f"{_out}.json"

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

# ── Load kline data for execution ──
_COL_MAP = {"时间": "date", "收盘价": "close", "开盘价": "open",
            "最高价": "high", "最低价": "low", "成交量": "volume"}

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

# ── Load training data ──
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

_dmask = df["date"] >= pd.Timestamp(TEST_START)
if TEST_END is not None:
    _dmask &= df["date"] <= pd.Timestamp(TEST_END)
dates = sorted(df[_dmask]["date"].unique())
if not dates:
    print(f"ERROR: No data after {TEST_START}")
    exit(1)

MIN_TRAIN_DAYS = 250
print(f"\nWalk-Forward: {len(dates)} prediction days ({dates[0].date()} ~ {dates[-1].date()})")
print(f"Strategy: intersection rebalance, top{TOP_N}, close-to-close execution")

# ── Walk-forward: generate daily predictions ──
daily_preds = []
t0 = datetime.now()

LABEL_HORIZON = 5  # fwd_5d_ret covers 5 days, must exclude from training

for day_idx, pred_date in enumerate(dates):
    # Exclude last LABEL_HORIZON days to prevent label leakage
    # fwd_5d_ret at day D covers D..D+5, so D+5 must be < pred_date => D < pred_date - 5 trading days
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
    top_preds = {str(c): float(p) for c, p in zip(top["code"].values, preds[top.index])}

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
holdings = {}  # code -> {"shares": int, "buy_price": float, "buy_date": date}
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
    
    # Determine what to sell and buy
    to_sell = current_holdings - new_top  # held but not in new top
    to_buy = new_top - current_holdings  # in new top but not held
    to_keep = current_holdings & new_top  # in both
    
    # Categorize rebalance type
    if len(to_sell) == 0 and len(to_buy) == 0:
        n_no_change += 1
    elif len(to_sell) == len(current_holdings) and len(to_buy) == len(new_top):
        n_full_turnover += 1
    else:
        n_partial_turnover += 1
    n_rebalances += 1
    
    # ── Sell: execute at close price ──
    sell_proceeds = 0.0
    sell_cost_total = 0.0
    for code in list(to_sell):
        h = holdings[code]
        close_price = get_close(klines, code, pred_date)
        if close_price is None:
            # Can't sell, keep holding
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
    
    # ── Buy: execute at close price (sell-first, then buy with available cash) ──
    available_capital = capital + sell_proceeds
    n_to_buy = len(to_buy)
    
    if n_to_buy > 0:
        # Each new stock gets equal share of available cash
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
    
    capital = available_capital
    
    # ── Calculate portfolio value at close ──
    portfolio_value = capital
    for code, h in holdings.items():
        cp = get_close(klines, code, pred_date)
        if cp is not None:
            portfolio_value += h["shares"] * cp
        else:
            portfolio_value += h["shares"] * h["buy_price"]
    
    # Daily return (close-to-close)
    if i > 0:
        prev_value = daily_records[-1]["portfolio_value"]
        daily_ret = portfolio_value / prev_value - 1 if prev_value > 0 else 0.0
    else:
        daily_ret = 0.0
    
    total_buy_cost += buy_cost_total if n_to_buy > 0 else 0
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
        "buy_cost": round(buy_cost_total if n_to_buy > 0 else 0, 2),
        "ic": round(dp["ic"], 4) if not np.isnan(dp["ic"]) else None,
    })

# ── Force liquidate at last day ──
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

# ── Calculate metrics ──
rdf = pd.DataFrame(daily_records)
rdf["date"] = pd.to_datetime(rdf["date"])
n_days = len(rdf)

# Sharpe (sampled every 5 days)
rdf_s = rdf.iloc[::5]
sharpe = float(rdf_s["daily_ret"].mean() / rdf_s["daily_ret"].std() * np.sqrt(252/5)) if rdf_s["daily_ret"].std() > 0 else 0
sharpe_raw = float(rdf["daily_ret"].mean() / rdf["daily_ret"].std() * np.sqrt(252)) if rdf["daily_ret"].std() > 0 else 0

cum_series = (1 + rdf["daily_ret"]).cumprod()
max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
win_rate = float((rdf["daily_ret"] > 0).mean())
ann_ret = float((1 + rdf["daily_ret"]).prod() ** (252/n_days) - 1) if n_days > 0 else 0
total_return = (final_value / INIT_CAPITAL - 1) * 100
total_cost = (total_buy_cost + total_sell_cost) / INIT_CAPITAL * 100

# IC stats
ic_vals = [r["ic"] for r in daily_records if r["ic"] is not None]
ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
ic_std = float(np.std(ic_vals)) if ic_vals else 0.0

# Turnover stats
avg_holdings = float(rdf["n_holdings"].mean())
avg_turnover = float((rdf["to_sell"].apply(len) + rdf["to_buy"].apply(len)).mean())

print(f"\n{'='*60}")
print(f"  v29 Intersection Rebalance DART Results")
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
             f"lr={LOCKED_PARAMS['learning_rate']}, intersection rebalance, close-to-close",
    "features": len(features),
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
