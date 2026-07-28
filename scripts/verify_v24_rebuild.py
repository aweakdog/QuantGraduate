"""重建后一致性验证: 新旧 v24 在重叠期的特征是否吻合

预期:
  - 纯K线派生特征: 应高度一致 (新浪qfq 全量重拉, 复权基准可能整体变化 ->
    水平可能有偏移, 但【收益率类/比率类】特征应几乎相同)
  - 标签 fwd_*: 应高度一致
  - 资金流/事件类: 事件已重建(公告扩充), 允许差异
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
new = pd.read_parquet(ROOT / "data/processed/training_data_v24.parquet")
old = pd.read_parquet(ROOT / "data/processed/bak_20260728/training_data_v24.parquet")
for d in (new, old):
    d["date"] = pd.to_datetime(d["date"])

common_cols = [c for c in new.columns if c in old.columns]
print(f"新 {len(new.columns)} 列 | 旧 {len(old.columns)} 列 | 共有 {len(common_cols)} 列")
only_new = [c for c in new.columns if c not in old.columns]
only_old = [c for c in old.columns if c not in new.columns]
if only_new:
    print(f"  仅新增: {only_new[:10]}")
if only_old:
    print(f"  仅旧有: {only_old[:10]}")

key = ["date", "code"]
num = [c for c in common_cols if c not in key and
       pd.api.types.is_numeric_dtype(new[c]) and pd.api.types.is_numeric_dtype(old[c])]

m = new[key + num].merge(old[key + num], on=key, suffixes=("_n", "_o"))
print(f"\n重叠 (date,code) 行数: {len(m):,}")

rows = []
t0 = time.time()
for i, c in enumerate(num, 1):
    a, b = m[f"{c}_n"], m[f"{c}_o"]
    both = a.notna() & b.notna()
    n_both = int(both.sum())
    if n_both < 100:
        rows.append((c, n_both, np.nan, np.nan))
        continue
    x, y = a[both], b[both]
    corr = x.corr(y)
    scale = max(abs(y).median(), 1e-9)
    rel = (x - y).abs().median() / scale
    rows.append((c, n_both, corr, rel))
    if i % 100 == 0:
        k = int(28 * i / len(num))
        print(f"\r  [{'#'*k}{'-'*(28-k)}] {i}/{len(num)} | {time.time()-t0:.0f}s",
              end="", flush=True)
print(f"\r  [{'#'*28}] {len(num)}/{len(num)} | {time.time()-t0:.0f}s\n")

r = pd.DataFrame(rows, columns=["feat", "n", "corr", "rel_dev"])


def bucket(c):
    if c.startswith("fwd_"):
        return "标签"
    if c.startswith(("mf_", "dde_", "mtss_", "fund_flow")):
        return "资金流"
    if c.startswith(("ev_", "tev_", "ann_")):
        return "事件/公告"
    if c.startswith("con_"):
        return "概念板块"
    if c.startswith(("cn_", "us_", "usd", "sox", "sp_", "dj_", "nq_", "a50")):
        return "宏观/外盘"
    if c in ("pe", "pb", "mcap", "revenue", "profit", "eps", "bps", "roe",
             "total_assets", "debt_ratio", "gross_margin", "operate_cf") or \
       any(c.startswith(p + "_ma") for p in ("pe", "pb", "eps", "bps", "roe")):
        return "基本面"
    return "K线技术面"


r["域"] = r["feat"].map(bucket)
print("=" * 70)
print(f"{'域':12s} {'列数':>5s} {'中位corr':>9s} {'corr<0.9':>9s} "
      f"{'中位相对偏差':>12s} {'偏差>5%':>8s}")
print("-" * 70)
for dm, g in r.groupby("域"):
    v = g.dropna(subset=["corr"])
    if not len(v):
        print(f"{dm:12s} {len(g):>5d}   (无可比数据)")
        continue
    print(f"{dm:12s} {len(g):>5d} {v['corr'].median():>9.4f} "
          f"{int((v['corr']<0.9).sum()):>9d} {v['rel_dev'].median():>12.5f} "
          f"{int((v['rel_dev']>0.05).sum()):>8d}")
print("-" * 70)

bad = r.dropna(subset=["corr"]).query("corr < 0.9").sort_values("corr")
print(f"\n相关性 <0.9 的特征 {len(bad)} 个 (TOP15):")
for _, x in bad.head(15).iterrows():
    print(f"  {x['feat']:36s} [{x['域']:8s}] corr={x['corr']:+.4f} "
          f"相对偏差={x['rel_dev']:.4f}  n={int(x['n']):,}")

print("\n=== 标签一致性 (最关键) ===")
for c in ["fwd_1d_ret", "fwd_5d_ret", "fwd_1d_t1_open_ret", "fwd_1d_exec_ret"]:
    row = r[r["feat"] == c]
    if len(row):
        x = row.iloc[0]
        print(f"  {c:22s} corr={x['corr']:.6f}  相对偏差={x['rel_dev']:.6f}  n={int(x['n']):,}")

r.to_csv(ROOT / "data/processed/verify_v24_rebuild.csv", index=False)
print(f"\n明细: data/processed/verify_v24_rebuild.csv")
