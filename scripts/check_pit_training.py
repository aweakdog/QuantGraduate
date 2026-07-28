"""检查 PIT 训练集: 成分过滤效果 + 关键特征缺失率"""
import numpy as np
import pandas as pd

COLS = ["date", "code", "mf_net_1d", "mtss_z", "ev_net_5d", "ann_5d", "pe", "ma20_pct"]

df = pd.read_parquet("data/processed/training_data_pit_v24.parquet", columns=COLS)
df["date"] = pd.to_datetime(df["date"])
u = pd.read_parquet("data/universe/universe_pit.parquet")
u["effective_date"] = pd.to_datetime(u["effective_date"])
eff = np.array(sorted(u["effective_date"].unique()))
members = {d: set(g["code"].astype(str).str.zfill(6)) for d, g in u.groupby("effective_date")}

c6 = df["code"].astype(str).str[:6]
per = np.searchsorted(eff, df["date"].values, side="right") - 1
keep = np.zeros(len(df), bool)
for i, d in enumerate(eff):
    m = per == i
    if m.any():
        keep[m] = c6[m].isin(members[pd.Timestamp(d)]).values

print(f"过滤前 {len(df):,} 行 / {df['code'].nunique()} 只"
      f" -> 过滤后 {keep.sum():,} 行 / {df[keep]['code'].nunique()} 只")
print(f"早于首生效日的行: {int((per < 0).sum()):,}")
g = df[keep]
sz = g.groupby("date").size()
print(f"每日截面股票数: 中位 {int(sz.median())} 最小 {int(sz.min())} 最大 {int(sz.max())}")
print("关键特征缺失率:")
print(g[["mf_net_1d", "mtss_z", "ev_net_5d", "ann_5d", "pe", "ma20_pct"]]
      .isna().mean().round(3).to_string())
