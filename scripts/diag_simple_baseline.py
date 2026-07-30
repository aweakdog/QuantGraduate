"""简单因子基线 vs 80特征 LightGBM

发现: 单个 atr_pct_ma20 (低波动因子) 测试期 IC -0.0448 (t=-4.15),
绝对值超过整个 80 特征模型的 +0.0275 (t=3.20)。

本脚本严格检验这是否偶然:
  1. 只用【筛选期】(< cutoff) 数据挑因子 —— 与模型享有完全相同的信息,
     不引入任何前视
  2. 剔除市场级特征(截面无变异, 对 demean 标签零信息)
  3. 挑 |t| >= 阈值 的因子, 按筛选期 IC 符号定方向
  4. 等权合成: 每日截面 z-score 后取平均
  5. 在【测试期】评估 IC / 分层单调性 / 多空价差, 与 ML 模型对比

同时报告有多少因子在测试期符号翻转 —— 这是衡量"因子稳定性"的直接证据,
也是解释 ML 模型为何跑不赢简单基线的关键。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", default="2023-09-19")
ap.add_argument("--min-t", type=float, default=3.0, help="筛选期 |t| 门槛")
ap.add_argument("--max-factors", type=int, default=12)
ap.add_argument("--corr-max", type=float, default=0.7)
args = ap.parse_args()

from pipeline.config import settings  # noqa: E402

DATA_DIR = settings.DATA_DIR
LABEL_RAW = "fwd_5d_ret"
CUTOFF = pd.Timestamp(args.cutoff)
LEAK = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP = {"date", "code", "group", "y", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
        "fwd_21d_ret", "fwd_1d_excess", "fwd_5d_excess", "fwd_1d_open_ret",
        "fwd_1d_exec_ret", "fwd_1d_t1_open_ret", "fwd_1d_t1_close_ret",
        "fwd_1d_exec_excess"}

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

feats = [c for c in df.columns if c not in SKIP and c not in LEAK
         and pd.api.types.is_numeric_dtype(df[c])]

sel = df[(df["date"] < CUTOFF) & df["y"].notna()]
oos = df[(df["date"] >= CUTOFF) & df["y"].notna()]
print(f"  筛选期 {sel['date'].nunique()} 天 | 测试期 {oos['date'].nunique()} 天")

# ── 剔除市场级特征(含其 MA 派生) ──
overall = sel[feats].std().replace(0, np.nan)
cs = sel.groupby("date")[feats].std().mean()
ratio = (cs / overall).fillna(0)


def base_of(f):
    for s_ in ("_ma5", "_ma20"):
        if f.endswith(s_):
            return f[: -len(s_)]
    return f


feats = [f for f in feats
         if ratio.get(base_of(f), ratio.get(f, 0)) >= 0.01]
print(f"  剔除市场级特征后候选: {len(feats)} 个\n")


def ic_series(col, sub):
    out = []
    for d, g in sub.groupby("date"):
        v = g[[col, "y"]].dropna()
        if len(v) < 20 or v[col].nunique() < 3:
            continue
        r = v[col].rank().corr(v["y"].rank())
        if not np.isnan(r):
            out.append(r)
    return np.array(out)


print("在【筛选期】计算每个候选因子的 IC (与模型信息完全一致) ...")
stats = {}
for c in feats:
    a = ic_series(c, sel)
    if len(a) < 100 or a.std() == 0:
        continue
    stats[c] = (a.mean(), a.mean() / a.std() * np.sqrt(len(a)))
tab = pd.DataFrame(stats, index=["ic", "t"]).T
tab["abs_t"] = tab["t"].abs()
tab = tab.sort_values("abs_t", ascending=False)
print(f"  {len(tab)} 个因子算出 IC, 其中 |t|>={args.min_t} 的有"
      f" {(tab['abs_t'] >= args.min_t).sum()} 个\n")

# ── 贪心选因子: |t| 高优先, 且与已选相关性 < corr_max ──
cand = tab[tab["abs_t"] >= args.min_t].index.tolist()
cm = sel[cand].corr().abs() if len(cand) > 1 else None
chosen = []
for c in cand:
    if len(chosen) >= args.max_factors:
        break
    if cm is not None and any(cm.at[c, g] > args.corr_max for g in chosen):
        continue
    chosen.append(c)

print("=" * 92)
print(f"选中 {len(chosen)} 个因子 (仅用筛选期信息, 方向按筛选期 IC 符号)")
print("=" * 92)
print(f"{'因子':<26} {'筛选期IC':>10} {'t':>7} {'方向':>6} | "
      f"{'测试期IC':>10} {'t':>7} {'符号':>8}")
print("-" * 92)
flip = 0
for c in chosen:
    ic_s, t_s = tab.loc[c, "ic"], tab.loc[c, "t"]
    a = ic_series(c, oos)
    ic_o = a.mean()
    t_o = a.mean() / a.std() * np.sqrt(len(a)) if a.std() else np.nan
    same = np.sign(ic_s) == np.sign(ic_o)
    if not same:
        flip += 1
    print(f"{c:<26} {ic_s:>+10.4f} {t_s:>7.2f} {'正' if ic_s>0 else '负':>6} | "
          f"{ic_o:>+10.4f} {t_o:>7.2f} {'一致' if same else '翻转!':>8}")
print("-" * 92)
print(f"  测试期符号翻转: {flip}/{len(chosen)} 个 ({flip/max(1,len(chosen)):.0%})")

# ── 合成 ──
print()
print("=" * 92)
print("等权 z-score 合成因子 在测试期的表现")
print("=" * 92)


def zscore_day(g, cols, signs):
    z = pd.DataFrame(index=g.index)
    for c, sg in zip(cols, signs):
        v = g[c]
        mu, sd = v.mean(), v.std()
        z[c] = ((v - mu) / sd).clip(-3, 3) * sg if sd and sd == sd else 0.0
    return z.mean(axis=1)


signs = [np.sign(tab.loc[c, "ic"]) for c in chosen]
rows = []
for d, g in oos.groupby("date"):
    gg = g[chosen + ["y", "code"]].copy()
    gg[chosen] = gg[chosen].apply(lambda s: s.fillna(s.median()))
    score = zscore_day(gg, chosen, signs)
    if score.notna().sum() < 20:
        continue
    rows.append(pd.DataFrame({"date": d, "score": score, "y": gg["y"]}))
panel = pd.concat(rows, ignore_index=True).dropna()

ics = []
for d, g in panel.groupby("date"):
    if len(g) < 20:
        continue
    r = g["score"].rank().corr(g["y"].rank())
    if not np.isnan(r):
        ics.append(r)
ics = np.array(ics)
t_ic = ics.mean() / ics.std() * np.sqrt(len(ics))
print(f"  合成因子 IC : {ics.mean():+.4f}  (t={t_ic:.2f}, {len(ics)} 天, "
      f"IC>0 占 {(ics>0).mean():.1%})")
print(f"  80特征 LGBM : +0.0275  (t=3.20)   <- 同期对照")
print(f"  单因子 atr_pct_ma20: 见上表")

# 分层
G = 10
deciles = []
for d, g in panel.groupby("date"):
    if len(g) < G * 3:
        continue
    q = pd.qcut(g["score"].rank(method="first"), G, labels=False)
    deciles.append(g.assign(grp=G - 1 - q).groupby("grp")["y"].mean())
dec = pd.DataFrame(deciles)
print(f"\n  分层平均收益(第1层=合成分最高, 相对当日截面均值):")
for gidx in range(G):
    if gidx in dec.columns:
        v = dec[gidx].dropna()
        tt = v.mean() / v.std() * np.sqrt(len(v)) if v.std() else np.nan
        print(f"    第{gidx+1:>2}层 {v.mean():>+8.3%}  t={tt:>6.2f}")
ls = (dec[0] - dec[G - 1]).dropna()
print(f"  多空价差: {ls.mean():+.3%}/期, t={ls.mean()/ls.std()*np.sqrt(len(ls)):.2f}, "
      f"胜率 {(ls>0).mean():.1%}")

print()
print("=" * 92)
print("判读")
print("=" * 92)
print("  若合成因子 IC 的 t 值 >= LGBM 的 3.20, 说明当前 ML 框架没有产生增量价值,")
print("  应先把简单因子基线做扎实, 再考虑用 ML 去捕捉非线性/交互。")
print("  符号翻转比例高则说明因子本身不稳定 —— 这是比换模型更根本的问题。")
