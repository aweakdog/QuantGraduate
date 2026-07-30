"""只针对回测锁定的 57 个特征, 逐年量化新旧训练集的漂移

回答: 重建后模型输入是否还等价于验证过的那一套
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOCK = PROC / "wf_daily_em_t1close_s001_ts2022-09-01_te2026-07-27_cap20000.json"

sel = json.load(open(LOCK))["selected_features"]
print(f"锁定特征 {len(sel)} 个")

old = pd.read_parquet(PROC / "training_data_pit_v24.parquet")
new = pd.read_parquet(PROC / "_smoke_test.parquet")
old["date"] = pd.to_datetime(old["date"])
new["date"] = pd.to_datetime(new["date"])

missing_old = [f for f in sel if f not in old.columns]
missing_new = [f for f in sel if f not in new.columns]
print(f"旧训练集缺失的锁定特征: {missing_old or '无'}")
print(f"新训练集缺失的锁定特征: {missing_new or '无'}")

usable = [f for f in sel if f in old.columns and f in new.columns]
key = ["date", "code"]
m = old[key + usable].merge(new[key + usable], on=key, suffixes=("_o", "_n"))
m["year"] = m["date"].dt.year
print(f"重叠行 {len(m):,}\n")

rows = []
for f in usable:
    a, b = m[f"{f}_o"], m[f"{f}_n"]
    both = a.notna() & b.notna()
    if not both.any():
        rows.append((f, 0.0, 0.0, 0))
        continue
    eq = float((a[both] == b[both]).mean())
    rel = float(((a[both] - b[both]).abs()
                 / a[both].abs().clip(lower=1e-9)).max())
    rows.append((f, eq, rel, int(both.sum())))

rows.sort(key=lambda r: r[1])
print("最不一致的 8 个锁定特征:")
print(f"  {'特征':30s}{'完全相等':>10}{'最大相对偏差':>16}{'比较行数':>10}")
for f, eq, rel, n in rows[:8]:
    print(f"  {f:30s}{eq:>9.2%}{rel:>16.4g}{n:>10,}")

clean = [r for r in rows if r[1] > 0.9999]
print(f"\n完全一致(>99.99%)的锁定特征: {len(clean)}/{len(usable)}")

# 对不一致的列, 看漂移是否只在 2022 (rolling 预热边界)
sus = [r[0] for r in rows if r[1] <= 0.9999]
if sus:
    print(f"\n不一致列逐年明细 ({len(sus)} 列):")
    for f in sus:
        a_all, b_all = m[f"{f}_o"], m[f"{f}_n"]
        parts = []
        for y, g in m.groupby("year"):
            a, b = g[f"{f}_o"], g[f"{f}_n"]
            both = a.notna() & b.notna()
            if not both.any():
                continue
            parts.append(f"{y}:{(a[both] == b[both]).mean():.1%}")
        print(f"  {f:30s} " + "  ".join(parts))
