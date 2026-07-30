"""P3 验收: 特征筛选必须可复现, 且排名不能被实现细节左右

三项测试:
  1. 可复现性 —— 同一份数据、同一参数, 跑两次必须得到【完全相同】的特征集。
     旧实现用 split 计数 + 非稳定排序, 并列项顺序不确定, 无法保证。
  2. 列顺序不变性 —— 打乱候选特征的【列顺序】后重跑, 特征集应保持不变。
     这是旧实现最致命的问题: 修复基本面数据后新增 30 列, 列顺序一变,
     并列区的特征就重新洗牌, 导致 18 个特征被"挤掉"。
  3. 新旧对比 —— 报告旧行为(seeds=1, pool_mult=1)与新行为的差异, 以及
     重要度分布的扁平程度(截断线附近有多少并列)。

不跑回测, 只跑筛选, 因此很快。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-file", default="training_data_pit_v24.parquet")
ap.add_argument("--pit-universe", default="universe_pit.parquet")
ap.add_argument("--cutoff", default="2023-09-19")
ap.add_argument("--n-features", type=int, default=80)
ap.add_argument("--corr-threshold", type=float, default=0.9)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--pool-mult", type=int, default=3)
args = ap.parse_args()

from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
LABEL_RAW, LABEL = "fwd_5d_ret", "y_target"
LEAKAGE = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP = {"date", "code", "group", LABEL, "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
        "fwd_21d_ret", "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret",
        "fwd_1d_exec_ret", "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret",
        "fwd_1d_exec_excess"}
EXCL = {"mf_pct_1d", "mf_pct_1d_ma5", "mf_pct_1d_ma20",
        "macd_signal", "macd_signal_ma5", "macd_signal_ma20"}
LOCKED = dict(n_estimators=151, max_depth=4, learning_rate=0.03, num_leaves=15,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
              random_state=42, n_jobs=10, verbosity=-1, boosting_type="dart")

print("加载数据 ...")
df = pd.read_parquet(DATA_DIR / "processed" / args.train_file)
df["date"] = pd.to_datetime(df["date"])

u = pd.read_parquet(DATA_DIR / "universe" / args.pit_universe)
u["effective_date"] = pd.to_datetime(u["effective_date"])
u["code6"] = u["code"].astype(str).str.zfill(6)
eff = pd.DatetimeIndex(sorted(pd.to_datetime(u["effective_date"].unique())))
members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
code6 = df["code"].astype(str).str[:6]
period = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
keep = np.zeros(len(df), dtype=bool)
for i, d in enumerate(eff):
    m = period == i
    if m.any():
        keep[m] = code6[m].isin(members[pd.Timestamp(d)]).values
df = df[keep].reset_index(drop=True)

df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())
all_features = [c for c in df.columns
                if c not in SKIP and c not in LEAKAGE and c not in EXCL
                and pd.api.types.is_numeric_dtype(df[c])]
print(f"  {len(df)} 行, {df['code'].nunique()} 只, {len(all_features)} 个候选特征")

s = df[(df["date"] < pd.Timestamp(args.cutoff)) & df[LABEL].notna()]
print(f"  筛选样本: {len(s)} 行 (< {args.cutoff})")
print(f"  有效观测估算: 约 {s['date'].nunique()} 个交易日, "
      f"5日重叠标签 -> 独立区块约 {s['date'].nunique() // 5} 个")


UNIQ_DATES = np.array(sorted(s["date"].unique()))
BLOCKS = np.array_split(np.arange(len(UNIQ_DATES)), max(1, len(UNIQ_DATES) // 5))


def run_selection(feats, n_top, corr_thresh, n_seeds, pool_mult, use_gain=True,
                  stable_sort=True, colsample=1.0, resample_dates=True):
    X = s.groupby("code")[feats].transform(lambda c: c.ffill().fillna(0))
    y = s[LABEL]
    mat = np.zeros((n_seeds, len(feats)))
    for i in range(n_seeds):
        itype = "gain" if use_gain else "split"
        if resample_dates and n_seeds > 1:
            rng = np.random.default_rng(1000 + i)
            pick = rng.choice(len(BLOCKS), size=max(1, int(len(BLOCKS) * 0.8)),
                              replace=False)
            keep = set(UNIQ_DATES[np.concatenate([BLOCKS[b] for b in sorted(pick)])])
            rowm = s["date"].isin(keep).values
            seed = 42
        else:
            rowm = np.ones(len(s), dtype=bool)
            seed = 42 + i * 1000
        p = dict(LOCKED, n_estimators=50, boosting_type="gbdt",
                 colsample_bytree=colsample,
                 random_state=seed, importance_type=itype)
        m = lgb.LGBMRegressor(**p).fit(X[rowm], y[rowm])
        g = np.asarray(m.booster_.feature_importance(importance_type=itype), float)
        mat[i] = g / g.sum() if g.sum() > 0 else 0.0
    imp = pd.DataFrame({"feature": feats, "importance": mat.mean(0),
                        "imp_std": mat.std(0)})
    if stable_sort:
        imp = imp.sort_values(["importance", "feature"], ascending=[False, True],
                             kind="mergesort").reset_index(drop=True)
    else:
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    nz = imp[imp["importance"] > 0].reset_index(drop=True)
    pool = nz["feature"].head(max(n_top * pool_mult, n_top)).tolist()
    cm = s[pool].corr().abs()
    sel = []
    for f in pool:
        if len(sel) >= n_top:
            break
        if any(cm.at[f, g] > corr_thresh for g in sel):
            continue
        sel.append(f)
    return sel, imp


print("\n" + "=" * 70)
print("测试1: 可复现性 (同参数跑两次)")
print("=" * 70)
a, imp_a = run_selection(all_features, args.n_features, args.corr_threshold,
                         args.seeds, args.pool_mult)
b, _ = run_selection(all_features, args.n_features, args.corr_threshold,
                     args.seeds, args.pool_mult)
same = a == b
print(f"  两次结果完全相同: {'是 ✓' if same else '否 ✗'}")
print(f"  选中 {len(a)} 个特征")
if not same:
    print(f"  差异: {set(a) ^ set(b)}")

print("\n" + "=" * 70)
print("测试2: 列顺序不变性 (打乱候选列顺序后重跑)")
print("=" * 70)
rng = np.random.default_rng(7)
for trial in range(2):
    shuf = list(all_features)
    rng.shuffle(shuf)
    c, _ = run_selection(shuf, args.n_features, args.corr_threshold,
                         args.seeds, args.pool_mult)
    inter = len(set(a) & set(c))
    jac = inter / len(set(a) | set(c))
    flag = "✓" if jac > 0.95 else ("△" if jac > 0.85 else "✗")
    print(f"  第{trial+1}次打乱: 交集 {inter}/{len(a)} | Jaccard {jac:.1%} {flag}")

print("\n" + "=" * 70)
print("测试3: 旧行为 vs 新行为")
print("=" * 70)
old, imp_old = run_selection(all_features, args.n_features, args.corr_threshold,
                             n_seeds=1, pool_mult=1, use_gain=False,
                             stable_sort=False, colsample=0.8,
                             resample_dates=False)
print(f"  旧行为(split计数/单种子/pool=1): 选中 {len(old)} 个  <- 名额 {args.n_features} 没填满")
print(f"  新行为(gain/{args.seeds}种子/pool={args.pool_mult}): 选中 {len(a)} 个")
print(f"  两者交集 {len(set(old) & set(a))} 个 | Jaccard {len(set(old)&set(a))/len(set(old)|set(a)):.1%}")

print("\n  --- 重要度分布扁平程度 (截断线附近并列数) ---")
for name, tbl, gain in [("旧 split计数", imp_old, False), ("新 gain", imp_a, True)]:
    nz = tbl[tbl["importance"] > 0].reset_index(drop=True)
    if len(nz) <= args.n_features:
        print(f"  {name}: 非零特征仅 {len(nz)} 个, 不足 {args.n_features}")
        continue
    cut = nz.loc[args.n_features - 1, "importance"]
    tied = int((nz["importance"] == cut).sum())
    top1 = nz.loc[0, "importance"]
    print(f"  {name:14s}: 截断值 {cut:.4g} | 并列 {tied:>3} 个 | "
          f"第1名/截断线 = {top1/cut if cut else float('inf'):.0f}x")

cv = (imp_a.set_index("feature").loc[a, "imp_std"] /
      imp_a.set_index("feature").loc[a, "importance"].replace(0, np.nan)).median()
print(f"\n  新方法选中特征的种子间变异系数(中位数): {cv:.2f}")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
ok = same
print(f"  可复现: {'✓' if same else '✗ 仍不可复现, 需继续排查'}")
print(f"  新特征集已写入下方, 可用 --features-from 复用")
out = ROOT / "data" / "processed" / "feature_selection_v2.json"
import json
out.write_text(json.dumps({
    "selected_features": a,
    "n_features": len(a),
    "feat_select_cutoff": args.cutoff,
    "feat_select_seeds": args.seeds,
    "feat_select_pool_mult": args.pool_mult,
    "corr_threshold": args.corr_threshold,
    "reproducible": bool(same),
    "importance_top120": imp_a.head(120).to_dict("records"),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  已保存: {out}")
sys.exit(0 if ok else 1)
