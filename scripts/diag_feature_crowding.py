"""诊断: 基本面特征修复后, 为什么挤掉了 18 个旧特征?

select_features 的机制是 imp.head(80) 硬截断 -> 去相关。
top80 是零和的: 新进 8 个, 必然挤出 8 个。
关键问题: 被挤掉的特征原本排第几? 如果都在 80 名边界附近徘徊,
那么"换血"只是噪声抖动, 不代表新特征更好。
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"

OLD_JSON = PROC / "wf_daily_em_t1close_s001_ts2022-09-01_te2026-07-27_cap20000.json"
NEW_JSON = PROC / "wf_daily_em_t1close_s001_fundfix_ts2022-09-01_te2026-07-27_cap20000.json"

old_feats = json.load(open(OLD_JSON, encoding="utf-8"))["selected_features"]
new_feats = json.load(open(NEW_JSON, encoding="utf-8"))["selected_features"]

both = [f for f in old_feats if f in new_feats]
only_old = [f for f in old_feats if f not in new_feats]
only_new = [f for f in new_feats if f not in old_feats]

print(f"旧 {len(old_feats)} 个 | 新 {len(new_feats)} 个 | 共有 {len(both)} 个")
print(f"被挤掉 {len(only_old)} 个 | 新进 {len(only_new)} 个\n")

FUND_PREFIX = ("revenue", "profit", "roe", "eps", "bps", "total_assets",
               "debt_", "gross_", "operate_cf", "pe", "pb")
def is_fund(f):
    return f.startswith(FUND_PREFIX)

print("新进的特征里, 基本面类:")
for f in only_new:
    print(f"    {'[基本面]' if is_fund(f) else '        '} {f}")

# ── 复现新数据上的重要度排名 ──
print("\n复现特征重要度排名 (与 select_features 完全一致的口径)...")
from pipeline.config import settings
DATA_DIR = settings.DATA_DIR

sys.argv = ["x", "--train-file", "training_data_pit_v24.parquet"]
LOCKED = dict(n_estimators=50, learning_rate=0.05, num_leaves=31,
              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=0.1, random_state=42,
              n_jobs=-1, verbose=-1, boosting_type="gbdt")

df = pd.read_parquet(DATA_DIR / "processed" / "training_data_pit_v24.parquet")
df["date"] = pd.to_datetime(df["date"])
u = pd.read_parquet(DATA_DIR / "universe" / "universe_pit.parquet")
u["effective_date"] = pd.to_datetime(u["effective_date"])
eff = pd.DatetimeIndex(sorted(pd.to_datetime(u["effective_date"].unique())))
members = {d: set(g["code"].astype(str).str.zfill(6)) for d, g in u.groupby("effective_date")}
c6 = df["code"].astype(str).str[:6]
per = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
keep = np.zeros(len(df), bool)
for i, d in enumerate(eff):
    m = per == i
    if m.any():
        keep[m] = c6[m].isin(members[pd.Timestamp(d)]).values
df = df[keep].reset_index(drop=True)

LABEL_RAW = "fwd_5d_ret"
df["y_target"] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

drop = {"date", "code", "y_target", "fwd_1d_ret", "fwd_5d_ret", "fwd_10d_ret",
        "fwd_20d_ret", "name", "industry"}
cands = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

CUTOFF = json.load(open(NEW_JSON, encoding="utf-8"))["feat_select_cutoff"]
s = df[(df["date"] < pd.Timestamp(CUTOFF)) & df["y_target"].notna()]
print(f"  筛选样本: {len(s):,} 行 (< {CUTOFF}), 候选 {len(cands)} 个")

X = s.groupby("code")[cands].transform(lambda c: c.ffill().fillna(0))
mdl = lgb.LGBMRegressor(**LOCKED).fit(X, s["y_target"])
imp = (pd.DataFrame({"feature": cands, "importance": mdl.feature_importances_})
       .sort_values("importance", ascending=False).reset_index(drop=True))
imp["rank"] = imp.index + 1
rk = dict(zip(imp["feature"], imp["rank"]))
iv = dict(zip(imp["feature"], imp["importance"]))

print(f"\n{'':2}top80 截断线附近的重要度 (第70~90名):")
for _, r in imp.iloc[69:90].iterrows():
    mark = "  <-- 第80名截断线" if r["rank"] == 80 else ""
    tag = "[基本面]" if is_fund(r["feature"]) else "        "
    print(f"    #{r['rank']:>3} {tag} {r['feature']:<30} 重要度 {r['importance']:>5}{mark}")

print(f"\n被挤掉的 {len(only_old)} 个旧特征, 在新数据里排名:")
rows = sorted([(rk.get(f, 999), iv.get(f, -1), f) for f in only_old])
for r, v, f in rows:
    where = "仍在top80内(被去相关剔除)" if r <= 80 else f"掉出top80"
    print(f"    #{r:>3} 重要度{v:>5}  {f:<30} {where}")

print(f"\n新进的 {len(only_new)} 个特征, 排名:")
for f in only_new:
    tag = "[基本面]" if is_fund(f) else "        "
    print(f"    #{rk.get(f,999):>3} 重要度{iv.get(f,-1):>5}  {tag} {f}")

nf = [f for f in new_feats if is_fund(f)]
print(f"\n入选的基本面特征 {len(nf)} 个: {nf}")
print(f"  它们的排名: {[rk.get(f) for f in nf]}")
print(f"\n第1名重要度 {imp['importance'].iloc[0]} | 第80名 {imp['importance'].iloc[79]} "
      f"| 第120名 {imp['importance'].iloc[119] if len(imp)>119 else 'NA'}")
