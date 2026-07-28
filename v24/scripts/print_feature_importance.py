"""
打印 v22 全量训练模型的 LightGBM 特征权重 (gain + split)
复刻 wf_daily_expanding.py 的: 特征筛选 / 逐股ffill预处理 / LOCKED_PARAMS
仅用于诊断"特征权重是否合理", 不跑回测。
"""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import lightgbm as lgb

DATA_DIR = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")
TRAIN_PATH = DATA_DIR / "processed" / "training_data_v22.parquet"
OUT_PATH = DATA_DIR / "processed" / "feature_importance_v22.json"

LABEL_RAW = "fwd_1d_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess"}
LOCKED_PARAMS = dict(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=50, random_state=42, n_jobs=64, verbosity=-1
)

def is_valid_feat(f):
    return "_21d" not in f and not f.endswith("_cross")

print("Loading...")
df = pd.read_parquet(TRAIN_PATH)
df["date"] = pd.to_datetime(df["date"])
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[LABEL_RAW])
df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

all_cols = [c for c in df.columns if c not in SKIP_COLS and is_valid_feat(c)]
features = [f for f in all_cols if f not in LEAKAGE_FEATS]
print(f"  {len(df):,} rows, {df['code'].nunique()} codes, {len(features)} features")

X = df.groupby("code")[features].transform(lambda s: s.ffill().bfill().fillna(0))
y = df[LABEL]
print("Training representative model (full data, LOCKED_PARAMS)...")
model = lgb.LGBMRegressor(**LOCKED_PARAMS)
model.fit(X, y)

gain = model.booster_.feature_importance(importance_type="gain")
split = model.booster_.feature_importance(importance_type="split")
imp = pd.DataFrame({"feature": features, "gain": gain, "split": split})
imp["gain_share"] = imp["gain"] / imp["gain"].sum() * 100
imp["split_share"] = imp["split"] / imp["split"].sum() * 100
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

# ── 集中度 ──
n_feat = len(imp)
n_zero = int((imp["gain"] == 0).sum())
top10_share = imp.head(10)["gain_share"].sum()
top20_share = imp.head(20)["gain_share"].sum()

# ── 按前缀分组 (首 _ 之前) ──
imp["group"] = imp["feature"].str.split("_").str[0]
grp = imp.groupby("group").agg(
    n=("feature", "size"),
    gain_share=("gain_share", "sum"),
    top_feat=("feature", "first"),
).sort_values("gain_share", ascending=False)

print("\n" + "="*70)
print("特征权重诊断 (v22 全量训练, gain 口径)")
print("="*70)
print(f"特征总数: {n_feat} | 零权重特征: {n_zero} ({n_zero/n_feat*100:.1f}%)")
print(f"Top10 集中度: {top10_share:.1f}% | Top20 集中度: {top20_share:.1f}%")
print(f"\n--- Top 40 (gain) ---")
for i, r in imp.head(40).iterrows():
    print(f"  {i+1:>2}. {r['feature']:<34} gain={r['gain_share']:6.2f}%  split={r['split_share']:5.2f}%")
print(f"\n--- Bottom 25 (gain, 含零权重) ---")
for i, r in imp.tail(25).iterrows():
    print(f"  {r['feature']:<34} gain={r['gain_share']:6.2f}%  split={r['split_share']:5.2f}%")

print(f"\n--- 按前缀分组 (gain_share 求和) ---")
for g, row in grp.iterrows():
    print(f"  {g:<14} n={int(row['n']):>3}  gain_share={row['gain_share']:6.2f}%  例:{row['top_feat']}")

# 链主特征专项
chain = imp[imp["feature"].str.contains("chain", case=False) | imp["feature"].str.contains("master", case=False)]
if len(chain):
    print(f"\n--- 链主相关特征: {len(chain)} 个, 总gain_share={chain['gain_share'].sum():.2f}% ---")
    for i, r in chain.head(15).iterrows():
        print(f"  {r['feature']:<34} gain={r['gain_share']:6.2f}%")

# 保存
out = {
    "n_features": n_feat, "n_zero": n_zero,
    "top10_share": round(top10_share, 2), "top20_share": round(top20_share, 2),
    "ranked": imp[["feature", "gain", "gain_share", "split", "split_share"]].to_dict(orient="records"),
    "by_group": {g: {"n": int(row["n"]), "gain_share": round(row["gain_share"], 2),
                     "top_feat": row["top_feat"]} for g, row in grp.iterrows()},
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT_PATH}")
