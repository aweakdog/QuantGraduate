"""组合层去极值 / 波动率倒数加权 & 截尾标签辅助模块 (2026-11-23 研究, 严守 PIT 口径)。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_demean(df: pd.DataFrame, col: str, q: float, valid_mask=None) -> pd.Series:
    """按每日截面 [q, 1-q] 分位缩尾后再减当日均值, 保持原始收益量纲。"""
    if not 0.0 < q < 0.5:
        raise ValueError(f"winsorize quantile q must be in (0, 0.5), got {q}")
    s = df[col].where(valid_mask) if valid_mask is not None else df[col]

    def _one(x: pd.Series) -> pd.Series:
        v = x.dropna()
        if len(v) < 5:
            return x - v.mean() if len(v) else x
        lo, hi = float(v.quantile(q)), float(v.quantile(1.0 - q))
        clipped = x.clip(lower=lo, upper=hi)
        return clipped - float(clipped.mean())

    return s.groupby(df["date"]).transform(_one)


class TrailingVolLookup:
    """严格 PIT 的个股近期波动率查询表: 只用 <= signal_date 的收盘日收益。"""

    def __init__(self, klines: dict, window: int = 20, min_periods: int = 10):
        self.window = int(window)
        self.min_periods = int(min_periods)
        self._dates = {}
        self._vols = {}
        for code, kl in klines.items():
            if kl is None or len(kl) < self.min_periods + 1:
                continue
            d = pd.to_datetime(kl["date"]).values
            c = pd.to_numeric(kl["close"], errors="coerce").astype(float)
            ret = c.pct_change()
            vol = ret.rolling(self.window, min_periods=self.min_periods).std().values
            self._dates[str(code)[:6]] = d
            self._vols[str(code)[:6]] = vol

    def get(self, code: str, sig_date) -> float | None:
        c6 = str(code)[:6]
        d = self._dates.get(c6)
        if d is None or len(d) == 0:
            return None
        ts = np.datetime64(pd.Timestamp(sig_date))
        idx = int(np.searchsorted(d, ts, side="right")) - 1
        if idx < 0:
            return None
        v = float(self._vols[c6][idx])
        return v if np.isfinite(v) and v > 1e-8 else None


def inv_vol_weights(codes, sig_date, vol_lookup: TrailingVolLookup,
                    lo: float = 0.5, hi: float = 2.0) -> list[float]:
    """按 1/sigma 分配候选股预算权重, 相对均值的倍率裁剪到 [lo, hi], 返回和为 1 的列表。"""
    n = len(codes)
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    raw = [vol_lookup.get(c, sig_date) for c in codes]
    valid = [v for v in raw if v is not None]
    if not valid:
        return [1.0 / n] * n
    med = float(np.median(valid))
    inv = np.array([1.0 / (v if v is not None else med) for v in raw], dtype=float)
    rel = inv / float(inv.mean())
    clipped = np.clip(rel, lo, hi)
    return [float(x) for x in (clipped / float(clipped.sum()))]
