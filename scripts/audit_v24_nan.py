"""排查 v24 最新日的全NaN列, 并与重建前的旧版对比"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data/processed/training_data_v24.parquet"
OLD = ROOT / "data/processed/bak_20260728/training_data_v24.parquet"

new = pd.read_parquet(NEW)
new["date"] = pd.to_datetime(new["date"])
print(f"新 v24: {len(new):,} 行, {new['code'].nunique()} 只, {len(new.columns)} 列, "
      f"{new['date'].min().date()} ~ {new['date'].max().date()}")

old = pd.read_parquet(OLD, columns=["date", "code"])
old["date"] = pd.to_datetime(old["date"])
print(f"旧 v24: {len(old):,} 行, {old['code'].nunique()} 只, "
      f"{old['date'].min().date()} ~ {old['date'].max().date()}")
print(f"新增行: {len(new) - len(old):+,}")

last = new["date"].max()
tail = new[new["date"] == last]
nan_all = [c for c in tail.columns if tail[c].isna().all()]
print(f"\n=== 最新日 {last.date()} 全NaN列: {len(nan_all)} 个 ===")

lab = [c for c in nan_all if c.startswith("fwd_")]
other = [c for c in nan_all if not c.startswith("fwd_")]
print(f"\n[标签类 {len(lab)} 个 — 正常, 最新日无未来数据]")
print("  " + ", ".join(lab))
print(f"\n[非标签 {len(other)} 个 — 需关注]")
for c in other:
    print(f"  {c}")

# 这些非标签列在倒数第 N 日的可用性, 定位断点
print(f"\n=== 非标签NaN列的最后有效日期 ===")
dates = sorted(new["date"].unique())[-40:]
sub = new[new["date"].isin(dates)]
g = sub.groupby("date")[other].apply(lambda x: x.notna().any())
for c in other:
    ok = g.index[g[c]]
    print(f"  {c:34s} 最后有效 = {pd.Timestamp(ok.max()).date() if len(ok) else '40日内均无'}")

# 检查回测测试期末 (2026-07-16) 的完整度
cut = pd.Timestamp("2026-07-16")
bt = new[new["date"] == cut]
if len(bt):
    n_nan = sum(bt[c].isna().all() for c in new.columns if not c.startswith("fwd_"))
    print(f"\n=== 回测期末 {cut.date()}: 非标签全NaN列 = {n_nan} 个 ===")

# c1_base 选中的 55 特征在最新日的可用性
res = ROOT / "data/processed"
cand = sorted(res.glob("wf_daily_c1_base*.json"))
if cand:
    j = json.loads(cand[-1].read_text())
    feats = j.get("selected_features") or j.get("features") or []
    if feats:
        miss = [f for f in feats if f in tail.columns and tail[f].isna().all()]
        absent = [f for f in feats if f not in tail.columns]
        print(f"\n=== c1_base 的 {len(feats)} 个选中特征在最新日 ===")
        print(f"  全NaN: {len(miss)} 个" + (f" -> {miss}" if miss else ""))
        print(f"  列不存在: {len(absent)} 个" + (f" -> {absent}" if absent else ""))
