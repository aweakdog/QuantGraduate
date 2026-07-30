"""被当作"泄漏特征"排除掉的 ret_1d/2d/5d/21d, 真的是泄漏吗? 它们有多少 alpha?

wf_v35 里:
    LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
这四个在 feature_engine 里是 close.pct_change(n), 即
    ret_5d(t) = close(t)/close(t-5) - 1
纯回看, 与标签 fwd_5d_ret(t) = close(t+5)/close(t) - 1 无任何重叠, 不构成泄漏。

而短期反转(过去几日跌得多的未来涨)是 A 股最稳健的截面因子之一。若它们
确实有显著 IC, 那么当前是在白白丢弃 alpha。

本脚本测这几个特征(及其 MA 派生)的截面 IC, 并与已入选特征做对比。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
LABEL_RAW = "fwd_5d_ret"
CUTOFF = "2023-09-19"

print("加载数据 ...")
df = pd.read_parquet(DATA_DIR / "processed" / "training_data_pit_v24.parquet")
df["date"] = pd.to_datetime(df["date"])

u = pd.read_parquet(DATA_DIR / "universe" / "universe_pit.parquet")
u["effective_date"] = pd.to_datetime(u["effective_date"])
u["code6"] = u["code"].astype(str).str.zfill(6)
eff = pd.DatetimeIndex(sorted(pd.to_datetime(u["effective_date"].unique())))
members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
c6 = df["code"].astype(str).str[:6]
per = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
keep = np.zeros(len(df), dtype=bool)
for i, d in enumerate(eff):
    m = per == i
    if m.any():
        keep[m] = c6[m].isin(members[pd.Timestamp(d)]).values
df = df[keep].reset_index(drop=True)
df["y"] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

# 全样本 + 分期
def ic_of(col, sub):
    ics = []
    for _, g in sub.groupby("date"):
        v = g[[col, "y"]].dropna()
        if len(v) < 20 or v[col].nunique() < 3:
            continue
        r = v[col].rank().corr(v["y"].rank())
        if not np.isnan(r):
            ics.append(r)
    if len(ics) < 30:
        return None
    a = np.array(ics)
    return a.mean(), a.mean() / a.std() * np.sqrt(len(a)), len(a)


TARGETS = ["ret_1d", "ret_2d", "ret_5d", "ret_21d",
           "ret_1d_ma5", "ret_1d_ma20", "ret_5d_ma5", "ret_5d_ma20",
           "ret_21d_ma5", "ret_21d_ma20"]
TARGETS = [c for c in TARGETS if c in df.columns]

sel_now = ["atr_ma20", "atr_pct_ma20", "ma20_pct_ma20", "rsi_14", "roe", "revenue"]
sel_now = [c for c in sel_now if c in df.columns]

sub_sel = df[df["date"] < pd.Timestamp(CUTOFF)]
sub_oos = df[df["date"] >= pd.Timestamp(CUTOFF)]

print(f"  筛选期 {sub_sel['date'].nunique()} 天 | 测试期 {sub_oos['date'].nunique()} 天\n")

print("=" * 88)
print("被当作泄漏排除的特征, 其截面 IC")
print("=" * 88)
print(f"{'特征':<18} {'筛选期IC':>10} {'t':>7} | {'测试期IC':>10} {'t':>7} | {'当前状态':>14}")
print("-" * 88)
LEAK = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
for c in TARGETS:
    a = ic_of(c, sub_sel)
    b = ic_of(c, sub_oos)
    if a is None or b is None:
        continue
    st = "已排除(误判)" if c in LEAK else "候选池内"
    print(f"{c:<18} {a[0]:>+10.4f} {a[1]:>7.2f} | {b[0]:>+10.4f} {b[1]:>7.2f} | {st:>14}")

print()
print("=" * 88)
print("对照: 当前 top 特征的截面 IC")
print("=" * 88)
print(f"{'特征':<18} {'筛选期IC':>10} {'t':>7} | {'测试期IC':>10} {'t':>7}")
print("-" * 88)
for c in sel_now:
    a = ic_of(c, sub_sel)
    b = ic_of(c, sub_oos)
    if a is None or b is None:
        continue
    print(f"{c:<18} {a[0]:>+10.4f} {a[1]:>7.2f} | {b[0]:>+10.4f} {b[1]:>7.2f}")

print()
print("=" * 88)
print("判读")
print("=" * 88)
print("  ret_Nd = close(t)/close(t-N)-1, 纯回看; 标签 = close(t+5)/close(t)-1。")
print("  两者无重叠, 不构成泄漏。若其 |t| 明显高于当前入选特征, 说明")
print("  LEAKAGE_FEATS 的设定在白白丢弃 A 股最经典的短期反转 alpha。")
