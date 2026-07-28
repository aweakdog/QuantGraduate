"""
将原始1min K线+资金流数据 → 合并到训练数据
过程：
1. 从1min K线算每日OHLCV
2. 从资金流数据取主力净流入
3. 追加到training_data_v22.parquet
"""
import pandas as pd, numpy as np, warnings, os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
BASE = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
TRAIN_PATH = BASE / "processed" / "training_data_v22.parquet"
OUT_PATH = BASE / "processed" / "training_data_v23.parquet"

print("=== 数据合并 ===")

# 1. Load existing training data
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
last_date = df["date"].max()
print(f"现有训练数据: {last_date.date()} ~ {df['date'].max().date()}, {len(df)} rows")

# 2. Find raw 1-min files after last training date
raw_dir = BASE / "raw" / "kline_1min"
raw_files = sorted(raw_dir.glob("*.parquet"))
dates_to_add = []
for f in raw_files:
    d = datetime.strptime(f.stem, "%Y%m%d")
    if d > last_date:
        dates_to_add.append(d)
print(f"待合并的日期: {[d.date() for d in dates_to_add]}")

if not dates_to_add:
    print("无新数据可合并")
    exit(0)

# 3. Get 120 pool codes
codes_120 = sorted(df["code"].unique())
print(f"120池: {len(codes_120)} 只股票")

# 4. Aggregate 1-min → daily for each date
new_rows = []
for d in dates_to_add:
    fname = d.strftime("%Y%m%d")
    fpath = raw_dir / f"{fname}.parquet"
    if not fpath.exists():
        print(f"  跳过 {fname}: 文件不存在")
        continue
    raw = pd.read_parquet(fpath)
    # raw data uses 6-char codes (e.g. "600519"), training uses 8-char ("600519.SH")
    raw["code6"] = raw["code"].str[:6]
    raw_120 = raw[raw["code6"].isin([c[:6] for c in codes_120])]
    if len(raw_120) == 0:
        print(f"  跳过 {fname}: 无120池数据")
        continue
    
    # Aggregate per stock per day
    for code_full in codes_120:
        code6 = code_full[:6]
        sub = raw_120[raw_120["code6"] == code6].sort_values("time")
        if len(sub) < 60:
            continue
        daily = {
            "date": pd.Timestamp(d),
            "code": code_full,
            "close": sub["close"].iloc[-1],
            "open": sub["open"].iloc[0],
            "high": sub["high"].max(),
            "low": sub["low"].min(),
            "vol": sub["vol"].sum(),
            "amount": sub["amount"].sum(),
        }
        new_rows.append(daily)
    
    print(f"  {fname}: {len(raw)} bars → {len(set(raw['code']))} stocks aggregated")

if not new_rows:
    print("无聚合数据")
    exit(0)

# 5. Build new dataframe with computed features
new_df = pd.DataFrame(new_rows)
new_df = new_df.sort_values(["date", "code"]).reset_index(drop=True)

# Compute returns
for code in codes_120:
    sub = new_df[new_df["code"] == code].sort_values("date")
    if len(sub) == 0:
        continue
    # Get last close from existing training data for ret_1d
    last_close = df[df["code"] == code].sort_values("date")
    prev_close = last_close["close"].iloc[-1] if len(last_close) > 0 and "close" in last_close.columns else None
    
    for i, idx in enumerate(sub.index):
        if prev_close is not None:
            new_df.loc[idx, "ret_1d"] = (sub.loc[idx, "close"] - prev_close) / prev_close
            new_df.loc[idx, "fwd_1d_ret"] = 0  # placeholder, computed in pipeline
        prev_close = sub.loc[idx, "close"]

# Compute MA features per stock
for code in codes_120:
    sub = new_df[new_df["code"] == code].sort_values("date").copy()
    for col in ["close", "vol", "amount"]:
        if col not in sub.columns:
            continue
        sub[f"{col}_ma5"] = sub[col].rolling(5).mean()
        sub[f"{col}_ma20"] = sub[col].rolling(20).mean()
    for c in sub.columns:
        if c in new_df.columns:
            new_df.loc[sub.index, c] = sub[c]

# Compute percentage features
for col in ["close"]:
    for w in [5, 20]:
        ma_col = f"{col}_ma{w}"
        if ma_col in new_df.columns:
            new_df[f"ma{w}_pct"] = (new_df[col] - new_df[ma_col]) / new_df[ma_col]

print(f"\n新数据: {len(new_df)} rows, {new_df['code'].nunique()} stocks")
print(f"  日期: {new_df['date'].min().date()} ~ {new_df['date'].max().date()}")

# 6. Merge with existing training data
combined = pd.concat([df, new_df], ignore_index=True)
combined = combined.sort_values(["date", "code"]).reset_index(drop=True)

# ffill: per-stock, carry forward all columns from last known value
num_cols = combined.select_dtypes(include=[np.number]).columns
for code in codes_120:
    mask = combined["code"] == code
    combined.loc[mask, num_cols] = combined.loc[mask, num_cols].ffill().bfill()

print(f"\n合并后 (ffill补齐): {len(combined)} rows, {combined['code'].nunique()} stocks")
print(f"  日期: {combined['date'].min().date()} ~ {combined['date'].max().date()}")

# 7. Save
combined.to_parquet(OUT_PATH, index=False)
print(f"\n保存: {OUT_PATH}")
print(f"  V16: {len(df)} rows → V17: {len(combined)} rows (+{len(new_df)} 新行)")
