"""异质线性 Ridge 模型与仿射尺度对齐混合 (wf_v35 回测与 live_signal 实盘共用纯函数)。

设计目标:
  1. 单日实盘预测 ridge_predict_day() 与回测增量走窗 walkforward_ridge_preds() 数学逐位等价 (<1e-10),
     确保线上与回测口径零漂移。
  2. 严守 PIT: 特征只在每日截面内做 rank(pct=True) - 0.5 居中标准化(缺失填 0);
     训练标签只在每日截面内做 [winsor, 1-winsor] 缩尾后再 demean; 截距项不加 L2 惩罚。
  3. 正则强度支持固定 alpha (如 10000) 或样本数自适应 alpha = alpha_ratio * N_train
     (防止随着 expanding 训练窗逐年增长导致正则强度悄悄衰减)。
  4. 仿射对齐 blend_affine(): 把 Ridge 单日预测按有效股票对齐到 DART 当日均值/标准差再加权,
     保留 --min-pred 0.002 门槛语义与空仓/部署率结构不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_section_rank_center(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """每日截面内将特征转成 [-0.5, +0.5] 百分比秩, 缺失填 0, 末尾拼截距列 1。"""
    if df.empty:
        return np.empty((0, len(features) + 1), dtype=np.float64)
    if "date" in df.columns and df["date"].nunique() > 1:
        ranked = df.groupby("date", sort=False)[features].rank(pct=True) - 0.5
    else:
        ranked = df[features].rank(pct=True) - 0.5
    arr = ranked.fillna(0.0).to_numpy(dtype=np.float64)
    return np.hstack([arr, np.ones((len(df), 1), dtype=np.float64)])


def winsorize_demean_series(df: pd.DataFrame, label_raw: str, winsor: float = 0.02) -> np.ndarray:
    """按每日截面 [winsor, 1-winsor] 缩尾后再减当日均值。"""
    s = pd.to_numeric(df[label_raw], errors="coerce").astype(float)
    if winsor <= 0:
        return s.groupby(df["date"], sort=False).transform(lambda x: x - x.mean()).to_numpy(dtype=np.float64)

    def _one(x: pd.Series) -> pd.Series:
        v = x.dropna()
        if len(v) < 5:
            return x - float(v.mean()) if len(v) else x
        lo, hi = float(v.quantile(winsor)), float(v.quantile(1.0 - winsor))
        c = x.clip(lower=lo, upper=hi)
        return c - float(c.mean())

    return s.groupby(df["date"], sort=False).transform(_one).to_numpy(dtype=np.float64)


def solve_ridge(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    """解 (X^T X + lam * I_0) w = X^T y, 末列截距不惩罚。"""
    p = xtx.shape[0]
    reg = np.eye(p, dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]


def effective_lambda(n_samples: int, alpha: float = 10000.0, alpha_ratio: float = 0.0) -> float:
    if alpha_ratio > 0:
        return float(alpha_ratio) * float(n_samples)
    return float(alpha)


def ridge_predict_day(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    label_raw: str,
    *,
    alpha: float = 10000.0,
    alpha_ratio: float = 0.0,
    winsor: float = 0.02,
) -> np.ndarray:
    """实盘单日入口: 用已截断的 train_df 拟合 Ridge, 对 test_df (单信号日) 输出线性预测。"""
    if test_df.empty:
        return np.empty(0, dtype=np.float64)
    y_tr = winsorize_demean_series(train_df, label_raw, winsor=winsor)
    valid = np.isfinite(y_tr)
    if int(valid.sum()) < 20:
        return np.zeros(len(test_df), dtype=np.float64)
    X_tr = cross_section_rank_center(train_df, features)[valid]
    yv = y_tr[valid]
    lam = effective_lambda(len(yv), alpha=alpha, alpha_ratio=alpha_ratio)
    w = solve_ridge(X_tr.T @ X_tr, X_tr.T @ yv, lam)
    X_te = cross_section_rank_center(test_df, features)
    return X_te @ w


def blend_affine(main_preds: np.ndarray, aux_preds: np.ndarray, weight: float) -> np.ndarray:
    """将 aux_preds 按有效股票仿射对齐到 main_preds 的均值与标准差后再按 (1-w)*main + w*aux 混合。"""
    w = float(weight)
    pm = np.asarray(main_preds, dtype=np.float64)
    pa = np.asarray(aux_preds, dtype=np.float64)
    if w <= 0 or len(pm) == 0:
        return pm.copy()
    valid = np.isfinite(pm) & np.isfinite(pa)
    if int(valid.sum()) <= 5:
        return pm.copy()
    m_mu, m_sd = float(pm[valid].mean()), float(pm[valid].std())
    a_mu, a_sd = float(pa[valid].mean()), float(pa[valid].std())
    if a_sd <= 1e-12 or m_sd <= 1e-12:
        return pm.copy()
    aligned = (pa - a_mu) / a_sd * m_sd + m_mu
    aligned = np.where(np.isfinite(aligned), aligned, pm)
    return (1.0 - w) * pm + w * aligned


def walkforward_ridge_preds(
    df: pd.DataFrame,
    features: list[str],
    label_raw: str,
    label_eval: str,
    test_start: str,
    test_end: str,
    horizon: int,
    min_days: int,
    configs: list[tuple[str, float, float]],
    *,
    winsor: float = 0.02,
) -> dict[str, list[dict]]:
    """增量走窗 Ridge 回测预测器, 与逐日调用 ridge_predict_day() 数学逐位一致。

    configs: [(tag, alpha, alpha_ratio), ...]
    """
    work = df.sort_values(["date", "code"]).reset_index(drop=True)
    X_arr = cross_section_rank_center(work, features)
    p = X_arr.shape[1]
    y_tr = winsorize_demean_series(work, label_raw, winsor=winsor)
    y_ev = pd.to_numeric(work[label_eval], errors="coerce").to_numpy(dtype=np.float64)
    codes_all = work["code"].astype(str).to_numpy()
    dates_col = pd.to_datetime(work["date"]).to_numpy()

    all_dates = sorted(pd.to_datetime(work["date"]).unique())
    date_pos = {d: i for i, d in enumerate(all_dates)}
    day_slices, day_xtx, day_xty, day_n, day_valid = {}, {}, {}, {}, {}
    bounds = np.flatnonzero(np.concatenate([[True], dates_col[1:] != dates_col[:-1], [True]]))
    for idx in range(len(bounds) - 1):
        s, e = int(bounds[idx]), int(bounds[idx + 1])
        d = pd.Timestamp(dates_col[s])
        day_slices[d] = (s, e)
        ym = np.isfinite(y_tr[s:e])
        if ym.any():
            Xd = X_arr[s:e][ym]
            yd = y_tr[s:e][ym]
            day_xtx[d] = Xd.T @ Xd
            day_xty[d] = Xd.T @ yd
            day_n[d] = int(ym.sum())
            day_valid[d] = True
        else:
            day_valid[d] = False

    ts0, ts1 = pd.Timestamp(test_start), pd.Timestamp(test_end)
    test_dates = [d for d in all_dates if ts0 <= d <= ts1]
    cum_xtx = np.zeros((p, p), dtype=np.float64)
    cum_xty = np.zeros(p, dtype=np.float64)
    cum_n = 0
    added_up_to = -1
    valid_days = 0

    out = {tag: [] for tag, _, _ in configs}
    for pred_date in test_dates:
        gpos = date_pos[pred_date]
        if gpos - horizon < 0:
            continue
        cutoff_idx = gpos - horizon
        while added_up_to + 1 < cutoff_idx:
            added_up_to += 1
            d_add = all_dates[added_up_to]
            if day_valid.get(d_add):
                cum_xtx += day_xtx[d_add]
                cum_xty += day_xty[d_add]
                cum_n += day_n[d_add]
                valid_days += 1
        if valid_days < min_days:
            continue
        s, e = day_slices[pred_date]
        Xt, yt, ct = X_arr[s:e], y_ev[s:e], codes_all[s:e]
        m = np.isfinite(yt)
        for tag, al, ar in configs:
            lam = effective_lambda(cum_n, alpha=al, alpha_ratio=ar)
            w = solve_ridge(cum_xtx, cum_xty, lam)
            pr = Xt @ w
            ic = float(spearmanr(pr[m], yt[m])[0]) if m.sum() > 5 else float("nan")
            order = np.argsort(-pr, kind="mergesort")
            out[tag].append({
                "date": pred_date,
                "ranked": [str(ct[k]) for k in order],
                "pred_vals": [round(float(pr[k]), 6) for k in order],
                "ic": ic,
            })
    return out
