"""异质线性模型 Ridge walk-forward 预测器 (E3, 2026-11-23)。

读取 wf_v35 --export-matrix 导出的冻结矩阵 (+ .meta.json), 严守相同 PIT/板块/purge6 截断口径:
  1. 每日截面将全部特征做百分比秩标准化到 [-0.5, +0.5], 缺失填 0 (零跨日泄漏)
  2. 每个预测日 D 只用 date < all_dates[gpos - label_horizon] 的全部历史样本拟合 Ridge(alpha)
     使用增量 XtX / Xty 累加 (O(p^2) Cholesky), 760 个预测日全量只需 ~10 秒!
  3. 输出标准 wf_v35 --load-preds 兼容的 v2 预测缓存 (meta 带 "model": "ridge_a<alpha>"),
     并打印日均 IC 与排名分层 5 日超额。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proctitle  # noqa: E402


def main(argv=None):
    proctitle.lowkey("mltask/worker")
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="wf_v35 --export-matrix 生成的 parquet 路径")
    ap.add_argument("--alphas", default="300,1000,3000,10000", help="逗号分隔的 Ridge 正则强度")
    ap.add_argument("--out-dir", required=True, help="预测缓存输出目录 (如 data/processed)")
    ap.add_argument("--label-alignment", default="purge6")
    ap.add_argument("--winsor", type=float, default=0.02, help="训练标签截面缩尾分位 (0=不缩尾)")
    a = ap.parse_args(argv)

    t0 = datetime.now()
    mp = Path(a.matrix)
    meta_in = json.loads(Path(str(mp) + ".meta.json").read_text(encoding="utf-8"))
    features = meta_in["features"]
    label_col = meta_in["label"]
    label_raw = meta_in["label_raw"]
    horizon = int(meta_in["label_horizon"])
    min_days = int(meta_in["min_train_days"])
    test_start = meta_in["test_start"]
    test_end = meta_in["test_end"]
    first_pred = meta_in["first_pred"]
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]

    df = pd.read_parquet(mp)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    df = df.sort_values(["date", "code"]).reset_index(drop=True)

    # 每日截面特征秩标准化到 [-0.5, +0.5], 缺失填 0 (仅用当日截面, 严守 PIT)
    print(f"加载 {mp.name}: {len(df):,} 行 × {len(features)} 特征, 截面秩标准化...", flush=True)
    X_rank = df.groupby("date")[features].rank(pct=True) - 0.5
    X_arr = X_rank.fillna(0.0).to_numpy(dtype=np.float64)
    # 加截距列
    X_arr = np.hstack([X_arr, np.ones((len(df), 1), dtype=np.float64)])
    p = X_arr.shape[1]

    if a.winsor > 0:
        from research_port_opt import winsorize_demean
        y_train_ser = winsorize_demean(df, label_raw, a.winsor)
    else:
        y_train_ser = df[label_col]
    y_train = y_train_ser.to_numpy(dtype=np.float64)
    y_eval = df[label_col].to_numpy(dtype=np.float64)

    all_dates = sorted(df["date"].unique())
    date_pos = {d: i for i, d in enumerate(all_dates)}
    # 预建每日切片索引与每日 XtX / Xty (仅有效标签行)
    day_slices = {}
    day_xtx = {}
    day_xty = {}
    day_valid = {}
    codes_all = df["code"].to_numpy()
    dates_col = df["date"].to_numpy()
    # group indices
    boundaries = np.flatnonzero(np.concatenate([[True], dates_col[1:] != dates_col[:-1], [True]]))
    for idx in range(len(boundaries) - 1):
        s, e = int(boundaries[idx]), int(boundaries[idx + 1])
        d = pd.Timestamp(dates_col[s])
        day_slices[d] = (s, e)
        ym = np.isfinite(y_train[s:e])
        if ym.any():
            Xd = X_arr[s:e][ym]
            yd = y_train[s:e][ym]
            day_xtx[d] = Xd.T @ Xd
            day_xty[d] = Xd.T @ yd
            day_valid[d] = True
        else:
            day_valid[d] = False

    test_dates = [d for d in all_dates if pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end)]
    # 增量累加 XtX / Xty: 随着 pred_date 单调推进, cutoff 单调推进
    cum_xtx = np.zeros((p, p), dtype=np.float64)
    cum_xty = np.zeros(p, dtype=np.float64)
    added_up_to = -1
    valid_train_days = 0

    preds_by_alpha = {al: [] for al in alphas}
    reg_eye = np.eye(p, dtype=np.float64)
    reg_eye[-1, -1] = 0.0  # 截距不惩罚

    for pred_date in test_dates:
        gpos = date_pos[pred_date]
        if gpos - horizon < 0:
            continue
        cutoff_idx = gpos - horizon
        # 累加 [added_up_to+1 .. cutoff_idx-1] (对应 date < all_dates[cutoff_idx])
        while added_up_to + 1 < cutoff_idx:
            added_up_to += 1
            d_add = all_dates[added_up_to]
            if day_valid.get(d_add):
                cum_xtx += day_xtx[d_add]
                cum_xty += day_xty[d_add]
                valid_train_days += 1
        if valid_train_days < min_days:
            continue
        s, e = day_slices[pred_date]
        Xt = X_arr[s:e]
        yt = y_eval[s:e]
        ct = codes_all[s:e]
        m = np.isfinite(yt)
        for al in alphas:
            w = np.linalg.solve(cum_xtx + al * reg_eye, cum_xty)
            pr = Xt @ w
            ic = float(spearmanr(pr[m], yt[m])[0]) if m.sum() > 5 else np.nan
            order = np.argsort(-pr, kind="mergesort")
            preds_by_alpha[al].append({
                "date": pred_date,
                "ranked": [str(ct[k]) for k in order],
                "pred_vals": [round(float(pr[k]), 6) for k in order],
                "ic": ic,
            })

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for al in alphas:
        dps = preds_by_alpha[al]
        tag = f"RIDGE_a{int(al)}"
        cache_path = out_dir / f"preds_{tag}.pkl"
        meta_out = {
            "model": f"ridge_a{int(al)}_w{a.winsor:g}",
            "train_file": meta_in["train_file"],
            "pit_universe": meta_in["pit_universe"],
            "label": "5d",
            "objective": "l2",
            "label_alignment": a.label_alignment,
            "label_horizon": horizon,
            "label_panel_sha256": None,
            "selected_features": features,
            "train_start": None,
            "test_start": test_start,
            "test_end": test_end,
            "neutralize_style": False,
            "n_features": 80,
            "exclude_feats": "",
            "n_selected": len(features),
            "feat_cutoff": first_pred,
        }
        with open(cache_path, "wb") as fh:
            pickle.dump({"meta": meta_out, "preds": dps}, fh)
        ics = [dp["ic"] for dp in dps if dp["ic"] is not None and np.isfinite(dp["ic"])]
        # 排名分层诊断 (第 1 名 vs 2~5 vs 6~8)
        y_lookup = dict(zip(zip(df["date"], df["code"]), df[label_col]))
        r1, r25, r68 = [], [], []
        for dp in dps:
            d = dp["date"]
            ys = [y_lookup.get((d, c), np.nan) for c in dp["ranked"][:8]]
            if len(ys) >= 8:
                if np.isfinite(ys[0]):
                    r1.append(ys[0])
                v25 = [x for x in ys[1:5] if np.isfinite(x)]
                if v25:
                    r25.append(float(np.mean(v25)))
                v68 = [x for x in ys[5:8] if np.isfinite(x)]
                if v68:
                    r68.append(float(np.mean(v68)))
        ic_m = float(np.mean(ics)) if ics else 0.0
        ic_t = float(ic_m / (np.std(ics) / np.sqrt(len(ics)))) if len(ics) > 1 and np.std(ics) > 0 else 0.0
        summary[tag] = {
            "days": len(dps),
            "ic_mean": round(ic_m, 4),
            "ic_tstat": round(ic_t, 2),
            "r1_excess_pct": round(float(np.mean(r1)) * 100, 3) if r1 else None,
            "r2_5_excess_pct": round(float(np.mean(r25)) * 100, 3) if r25 else None,
            "r6_8_excess_pct": round(float(np.mean(r68)) * 100, 3) if r68 else None,
            "cache": str(cache_path),
        }
        print(f"  {tag:14s}: days={len(dps)} IC={ic_m:+.4f} (t={ic_t:.2f}) | r1={summary[tag]['r1_excess_pct']:+.3f}% r2-5={summary[tag]['r2_5_excess_pct']:+.3f}% r6-8={summary[tag]['r6_8_excess_pct']:+.3f}%", flush=True)
    print(f"完成, 耗时 {(datetime.now() - t0).total_seconds():.1f}s")
    (out_dir / "ridge_train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
