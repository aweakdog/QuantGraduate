"""校验重建出的训练集与旧训练集口径一致 (列集/历史段逐行数值)

用法: python scripts/verify_rebuild_parity.py <旧parquet> <新parquet>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

old_p = PROC / (sys.argv[1] if len(sys.argv) > 1 else "training_data_pit_v24.parquet")
new_p = PROC / (sys.argv[2] if len(sys.argv) > 2 else "_smoke_test.parquet")

old = pd.read_parquet(old_p)
new = pd.read_parquet(new_p)
old["date"] = pd.to_datetime(old["date"])
new["date"] = pd.to_datetime(new["date"])

print(f"旧 {old_p.name}: {len(old):,} 行 {old['code'].nunique()} 只 "
      f"{len(old.columns)} 列 max={old['date'].max().date()}")
print(f"新 {new_p.name}: {len(new):,} 行 {new['code'].nunique()} 只 "
      f"{len(new.columns)} 列 max={new['date'].max().date()}")

oc, nc = set(old.columns), set(new.columns)
print(f"\n旧有新无 (缺列): {sorted(oc - nc) or '无'}")
print(f"新有旧无 (新增): {sorted(nc - oc) or '无'}")

# 历史重叠段逐行比对
key = ["date", "code"]
num_cols = [c for c in (oc & nc) if c not in key
            and pd.api.types.is_numeric_dtype(old[c])]
m = old[key + num_cols].merge(new[key + num_cols], on=key, suffixes=("_o", "_n"))
print(f"\n重叠行: {len(m):,}  (对 {len(num_cols)} 个数值列逐行比对)")

worst = []
for c in num_cols:
    a, b = m[f"{c}_o"], m[f"{c}_n"]
    both = a.notna() & b.notna()
    if not both.any():
        continue
    diff = (a[both] - b[both]).abs()
    scale = a[both].abs().clip(lower=1e-9)
    rel = (diff / scale).max()
    worst.append((float(diff.max()), float(rel), c, int(both.sum())))

worst.sort(reverse=True)
print("\n偏差最大的 12 列:")
for amax, rmax, c, n in worst[:12]:
    print(f"  {c:28s} n={n:>8,}  绝对={amax:.10g}  相对={rmax:.3g}")

bad = [w for w in worst if w[1] > 1e-6 and w[0] > 1e-8]
print(f"\n相对偏差 >1e-6 的列数: {len(bad)} / {len(worst)}")
print("结论:", "口径一致 ✅" if not bad else f"存在口径漂移 ⚠️  {[w[2] for w in bad[:10]]}")
