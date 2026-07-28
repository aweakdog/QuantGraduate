"""验证宏观序列拼接质量: 接缝处日收益是否为异常离群值

原理: 若新旧口径不一致, 接缝当日会出现一个不属于历史分布的假跳变。
判据: 接缝日收益的 |z-score| (相对该序列历史日收益分布) 应 < 4
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "raw" / "macro"
BAK = ROOT / "data" / "raw" / "macro.bak_20260728"

print("=== 宏观序列拼接质量校验 ===\n")
print(f"{'序列':22s} {'旧末日':11s} {'新末日':11s} {'新增':>4s} "
      f"{'接缝收益':>9s} {'z值':>7s}  判定")
print("-" * 78)

rows = []
for p in sorted(MACRO.glob("*.parquet")):
    b = BAK / p.name
    if not b.exists():
        continue
    try:
        new = pd.read_parquet(p)
        old = pd.read_parquet(b)
    except Exception:
        continue
    if "日期" not in new.columns or "最新值" not in new.columns:
        continue
    for d in (new, old):
        d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
        d["v"] = pd.to_numeric(d["最新值"], errors="coerce")
    new = new.dropna(subset=["日期", "v"]).sort_values("日期")
    old = old.dropna(subset=["日期", "v"]).sort_values("日期")
    if not len(old) or not len(new):
        continue
    old_max = old["日期"].max()
    added = int((new["日期"] > old_max).sum())
    if added == 0:
        continue

    r = new.set_index("日期")["v"].pct_change().dropna()
    if len(r) < 60:
        continue
    # 接缝日 = 新增段第一天
    seam_date = new[new["日期"] > old_max]["日期"].min()
    if seam_date not in r.index:
        continue
    seam_r = r.loc[seam_date]
    hist = r[r.index < seam_date]
    mu, sd = hist.mean(), hist.std()
    z = (seam_r - mu) / sd if sd else float("nan")
    verdict = "OK" if abs(z) < 4 else ("可疑" if abs(z) < 6 else "!! 假跳变")
    rows.append((p.stem, z, verdict))
    print(f"{p.stem:22s} {str(old_max.date()):11s} "
          f"{str(new['日期'].max().date()):11s} {added:>4d} "
          f"{seam_r*100:>8.2f}% {z:>7.2f}  {verdict}")

print("-" * 78)
bad = [r for r in rows if r[2] != "OK"]
print(f"共校验 {len(rows)} 条更新序列 | 正常 {len(rows)-len(bad)} | 异常 {len(bad)}")
if bad:
    print("异常序列: " + ", ".join(f"{n}(z={z:.1f})" for n, z, _ in bad))
else:
    print("全部接缝平滑, 无口径断裂")
