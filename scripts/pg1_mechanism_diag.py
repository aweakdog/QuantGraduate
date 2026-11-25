"""E4: PG1 (purge6) 机制诊断 —— 检验 N6 留下的「上周奏效因子组合过拟合」假说 (纯离线只读)。

原理:
  对于每个预测日 D (位置 gpos), 最新被 purge6 丢弃的那个已闭合样本日是 D_last = all_dates[gpos - 6],
  它的 5 日标签窗恰好覆盖 [D_last -> D-1] (即刚刚结束的一周)。
  在该日 D_last 上, 我们可以计算每个特征 f 与当期 5 日超额收益 y 的截面相关系数向量 g(D_last) ∈ R^84
  —— 即「最近一周刚奏效的因子收益方向」。
  然后在预测日 D 上, 用这组上周奏效方向对当日股票打分:
      factor_mom_score(i, D) = sum_f z_rank(X_{i,f,D}) * g_f(D_last)
  再比较 CURRENT (含 D_last) 与 PURGE6 (剔除 D_last) 在 40 个同种子上的:
      expo_f_mom = spearman(pred(D), factor_mom_score(D))
  以及同时测「去掉 D_last 这 1 天是否降低了预测换手/极值率/第 1 名追高」。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proctitle  # noqa: E402
import run_overnight_research as ro  # noqa: E402


def main(argv=None):
    proctitle.lowkey("mltask/task")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="N2/N4/N5 研究根目录 (含 40 对 CURRENT/PURGE6 缓存)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    mat_path = root / "data/processed/training_data_pit_v24_tick1.parquet"
    uni_path = root / "data/universe/universe_pit.parquet"
    feat_path = root / "data/processed/features_V24PUT_T1B.json"
    feats_all = json.loads(feat_path.read_text(encoding="utf-8"))["selected_features"]

    df = pd.read_parquet(mat_path)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    # 主板过滤
    df = df[~df["code"].str.startswith(("30", "688"))]
    # PIT 过滤
    u = pd.read_parquet(uni_path)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    u["code6"] = u["code"].astype(str).str.zfill(6)
    eff = pd.DatetimeIndex(sorted(u["effective_date"].unique()))
    members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
    code6 = df["code"].str[:6]
    period = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
    keep = np.zeros(len(df), dtype=bool)
    for i, d in enumerate(eff):
        m = period == i
        if m.any():
            keep[m] = code6[m].isin(members[pd.Timestamp(d)]).values
    df = df[keep].reset_index(drop=True)

    feats = [f for f in feats_all if f in df.columns]
    df["y"] = df.groupby("date")["fwd_5d_ret"].transform(lambda x: x - x.mean())
    all_dates = sorted(df["date"].unique())
    date_pos = {d: i for i, d in enumerate(all_dates)}

    # 预计算每日截面秩标准化的特征矩阵 (仅对有非零方差的股票特征, 排除截面常数列)
    print(f"加载矩阵 {len(df):,} 行, 匹配到 {len(feats)} 个特征, 预计算每日因子方向...", flush=True)
    by_day = {}
    day_factor_ic = {}
    for d, g in df.groupby("date", sort=True):
        X = g[feats].rank(pct=True) - 0.5
        X = X.fillna(0.0).to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)
        codes = g["code"].to_numpy()
        by_day[d] = (codes, X, y)
        m = np.isfinite(y)
        if m.sum() > 20:
            yc = y[m] - y[m].mean()
            yn = np.linalg.norm(yc)
            Xm = X[m]
            Xm = Xm - Xm.mean(axis=0, keepdims=True)
            xn = np.linalg.norm(Xm, axis=0)
            ic_vec = np.where((xn > 1e-12) & (yn > 1e-12), (Xm.T @ yc) / (xn * yn + 1e-12), 0.0)
            day_factor_ic[d] = ic_vec

    # 预计算每个预测日 D 上, 按 D_last = all_dates[gpos - 6] 的因子方向打出的 factor_mom_score
    fmom_score = {}
    for gpos, d in enumerate(all_dates):
        if gpos < 6:
            continue
        d_last = all_dates[gpos - 6]
        g_vec = day_factor_ic.get(d_last)
        if g_vec is None:
            continue
        codes, X, _ = by_day[d]
        sc = X @ g_vec
        fmom_score[d] = dict(zip(codes, sc))

    rows = []
    for seed in ro.N6_SEEDS:
        leg_name, pg_name, _ = ro.n6_caches(seed)
        p_leg = root / "data/processed" / leg_name
        p_pg = root / "data/processed" / pg_name
        if not (p_leg.exists() and p_pg.exists()):
            continue
        leg = pickle.load(open(p_leg, "rb"))["preds"]
        pg = pickle.load(open(p_pg, "rb"))["preds"]
        leg_d = {dp["date"]: dp for dp in leg}
        pg_d = {dp["date"]: dp for dp in pg}
        e_leg, e_pg = [], []
        top5_ov_leg, top5_ov_pg = [], []
        prev_leg, prev_pg = None, None
        for d in sorted(set(leg_d) & set(pg_d)):
            fm = fmom_score.get(d)
            if fm is None:
                continue
            dl, dp = leg_d[d], pg_d[d]
            # 1. 对上周奏效因子方向的暴露
            for dp_i, out_list in ((dl, e_leg), (dp, e_pg)):
                pv = dp_i.get("pred_vals")
                if not pv:
                    continue
                pairs = [(v, fm[c]) for c, v in zip(dp_i["ranked"], pv) if c in fm]
                if len(pairs) > 20:
                    c_val = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])[0]
                    if np.isfinite(c_val):
                        out_list.append(float(c_val))
            # 2. 相邻日 top5 重叠度 (预测稳定性 / 换手倾向)
            t5l, t5p = set(dl["ranked"][:5]), set(dp["ranked"][:5])
            if prev_leg is not None:
                top5_ov_leg.append(len(t5l & prev_leg) / 5.0)
                top5_ov_pg.append(len(t5p & prev_pg) / 5.0)
            prev_leg, prev_pg = t5l, t5p
        rows.append({
            "seed": seed,
            "fmom_leg": float(np.mean(e_leg)),
            "fmom_pg": float(np.mean(e_pg)),
            "d_fmom": float(np.mean(e_pg) - np.mean(e_leg)),
            "top5_ov_leg": float(np.mean(top5_ov_leg)),
            "top5_ov_pg": float(np.mean(top5_ov_pg)),
        })

    t = pd.DataFrame(rows)
    res = {
        "n_seeds": len(t),
        "fmom_leg_mean": round(float(t["fmom_leg"].mean()), 4),
        "fmom_pg_mean": round(float(t["fmom_pg"].mean()), 4),
        "d_fmom_median": round(float(t["d_fmom"].median()), 4),
        "fmom_lower_seeds": int((t["d_fmom"] < 0).sum()),
        "top5_overlap_leg_mean": round(float(t["top5_ov_leg"].mean()), 4),
        "top5_overlap_pg_mean": round(float(t["top5_ov_pg"].mean()), 4),
        "top5_more_stable_seeds": int((t["top5_ov_pg"] > t["top5_ov_leg"]).sum()),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (root / "summary_pg1_mech.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
