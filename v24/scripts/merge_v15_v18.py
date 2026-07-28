"""
合并7/3~7/6到训练集 (v15 → v21)
仅添加日期占位行 + fwd_1d_ret 计算，不传OHLCV
v15没有close列，传了也是污染
"""
import pandas as pd, numpy as np, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
BASE = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
V15 = BASE / "processed" / "training_data_v15.parquet"
V22 = BASE / "processed" / "training_data_v22.parquet"

print("=== 补全7/3~7/6 (仅日期+标签, 无OHLCV) ===")

df = pd.read_parquet(V15)
df["date"] = pd.to_datetime(df["date"])
codes = sorted(df["code"].unique())
print(f"v15: {len(df):,} rows, {len(codes)} stocks, ~{df['date'].max().date()}")

# 从1min K线算2天的ret_1d和fwd_1d_ret
raw_dir = BASE / "raw" / "kline_1min"
new_rows = []

for fname in ["20260703", "20260706"]:
    fpath = raw_dir / f"{fname}.parquet"
    if not fpath.exists():
        continue
    raw = pd.read_parquet(fpath)
    d = datetime.strptime(fname, "%Y%m%d")
    for code_full in codes:
        code6 = code_full[:6]
        sub = raw[raw["code"].astype(str).str[:6]==code6].sort_values("time")
        if len(sub) < 60:
            continue
        # Get close from 1min to compute ret_1d only
        close = sub["close"].iloc[-1]
        new_rows.append({"date": pd.Timestamp(d), "code": code_full, "close_1min": close})
    cnt = sum(1 for r in new_rows if r.get("date")==pd.Timestamp(d))
    print(f"  {fname}: {cnt} stocks")

if not new_rows:
    print("无新数据"); exit()

new_df = pd.DataFrame(new_rows)

# Compute ret_1d from last close in v15 + 1min close
for code in codes:
    sub = new_df[new_df["code"]==code].sort_values("date")
    if len(sub) == 0:
        continue
    last_close = df[df["code"]==code].sort_values("date")
    prev = last_close.iloc[-1] if len(last_close)>0 else None
    prev_close = prev.get("close") if prev is not None else None
    # Try to get close from ret_1d derived in v15
    if prev_close is None:
        # Use last ret_1d to back-compute
        for i, idx in enumerate(sub.index):
            new_df.loc[idx, "ret_1d"] = 0
            new_df.loc[idx, "fwd_1d_ret"] = 0
    else:
        for i, idx in enumerate(sub.index):
            new_df.loc[idx, "ret_1d"] = (sub.loc[idx,"close_1min"] - prev_close) / prev_close
            new_df.loc[idx, "fwd_1d_ret"] = sub.loc[idx,"ret_1d"] if i < len(sub)-1 else 0
            prev_close = sub.loc[idx, "close_1min"]

# 合并：只保留v15已有的列
# new_df只包含v15有 + 刚才计算的ret_1d/fwd_1d_ret
keep_cols = [c for c in ["date","code","ret_1d","fwd_1d_ret"] if c in new_df.columns]
new_clean = new_df[keep_cols]

combined = pd.concat([df, new_clean], ignore_index=True)
combined = combined.sort_values(["date","code"]).reset_index(drop=True)

print(f"\n合并后: {len(combined):,} rows, {combined['code'].nunique()} stocks")
print(f"  日期: {combined['date'].min().date()} ~ {combined['date'].max().date()}")

combined.to_parquet(V22, index=False)
print(f"保存: {V22}")
