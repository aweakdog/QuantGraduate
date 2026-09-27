"""N4 机制检验(离线, 只读预测缓存与K线): purge 是否降低了模型的追涨暴露。

对每个种子、每个预测日 D:
  mom5   = C[D]/C[D-5]-1                       (信号日可知)
  fwd5   = C[D+6]/C[D+1]-1                     (t1close 执行口径的真实持有期收益, 只做诊断)
  expo   = spearman(pred, mom5)                (模型对近 5 日涨幅的暴露)
  top_mom= 前 5 名的 mom5 截面分位均值
  rev_ic = spearman(mom5, fwd5)                (市场本身: <0 = 5 日反转)
比较 PURGEk 与同种子基线的配对差; 并按 rev_ic 分组看差异是否集中在反转强的日子。
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/home/yliog/quant-research-20260912-n2")
proc = root / "data/processed"
SEEDS = [1, 42, 123, 888, 2024, 7, 31337, 2, 3, 5, 11, 17, 23, 55, 77, 99, 202, 314, 512, 1234]
model = sys.argv[1] if len(sys.argv) > 1 else "B"


def load(tag, seed):
    p = proc / f"preds_{tag}_s{seed}.pkl"
    if not p.exists():
        return None
    return {pd.Timestamp(r["date"]): pd.Series(r["pred_vals"], index=[str(c)[:6] for c in r["ranked"]])
            for r in pickle.load(p.open("rb"))["preds"]}


any_preds = load(f"N2_L_{model}_CURRENT", 1)
codes = sorted({c for s in any_preds.values() for c in s.index})
close = {}
for c in codes:
    f = root / "data/raw/kline" / f"{c}.parquet"
    if f.exists():
        k = pd.read_parquet(f, columns=["date", "close"])
        k["date"] = pd.to_datetime(k["date"])
        close[c] = k.drop_duplicates("date").set_index("date")["close"].astype(float)
px = pd.DataFrame(close).sort_index()
px = px[px.index <= pd.Timestamp(json.loads((root / "snapshot_manifest.json").read_text())["test_end"])]
mom5 = px / px.shift(5) - 1
fwd5 = px.shift(-6) / px.shift(-1) - 1


def day_stats(preds):
    rows = {}
    for d, s in preds.items():
        if d not in mom5.index:
            continue
        m = mom5.loc[d].reindex(s.index)
        f = fwd5.loc[d].reindex(s.index)
        ok = m.notna()
        if ok.sum() < 30:
            continue
        mr = m[ok].rank(pct=True)
        top = s[ok].sort_values(ascending=False).index[:5]
        rows[d] = {"expo": s[ok].rank().corr(m[ok].rank()),
                   "top_mom_pct": float(mr.loc[top].mean()),
                   "rev_ic": m[ok & f.notna()].rank().corr(f[ok & f.notna()].rank()) if (ok & f.notna()).sum() >= 30 else np.nan}
    return pd.DataFrame(rows).T


def summary(arm, base_tag, seeds, fallback=None):
    out = []
    for seed in seeds:
        a = load(f"N2_L_{model}_{arm}", seed)
        b = load(f"N2_L_{model}_{base_tag}", seed) or (load(fallback, seed) if fallback else None)
        if a is None or b is None:
            continue
        da, db = day_stats(a), day_stats(b)
        common = da.index.intersection(db.index)
        da, db = da.loc[common], db.loc[common]
        diff = da - db
        strong_rev = db["rev_ic"] < db["rev_ic"].median()
        out.append({"seed": seed, "days": len(common),
                    "expo_base": db.expo.mean(), "expo_arm": da.expo.mean(),
                    "top_mom_base": db.top_mom_pct.mean(), "top_mom_arm": da.top_mom_pct.mean(),
                    "d_expo": diff.expo.mean(), "d_top": diff.top_mom_pct.mean(),
                    "d_expo_strongrev": diff.expo[strong_rev].mean(), "d_expo_weakrev": diff.expo[~strong_rev].mean(),
                    "rev_ic": db.rev_ic.mean()})
    return pd.DataFrame(out)


pd.set_option("display.width", 200)
for arm, seeds in [("PURGE6", SEEDS), ("PURGE7", SEEDS[:10]), ("PURGE8", SEEDS[:10]), ("COMMON5", SEEDS[:10])]:
    t = summary(arm, "CURRENT", seeds, fallback=f"N2_F_{model}_FULL")
    if t.empty:
        continue
    print(f"\n=== {model} {arm} vs CURRENT  seeds {len(t)}  days/seed {int(t.days.median())}")
    print(f"  expo(pred~mom5)  base {t.expo_base.mean():+.4f}  arm {t.expo_arm.mean():+.4f}  "
          f"diff med {t.d_expo.median():+.4f}  lower in {int((t.d_expo < 0).sum())}/{len(t)}")
    print(f"  top5 mom5 pct    base {t.top_mom_base.mean():.3f}   arm {t.top_mom_arm.mean():.3f}   "
          f"diff med {t.d_top.median():+.4f}  lower in {int((t.d_top < 0).sum())}/{len(t)}")
    print(f"  expo diff on strong-reversal days {t.d_expo_strongrev.median():+.4f} vs weak {t.d_expo_weakrev.median():+.4f}")
    print(f"  market rev_ic(mom5~fwd5) mean {t.rev_ic.mean():+.4f}")
