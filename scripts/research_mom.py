"""MN1 研究口径 (2026-10-01): 排序前压低模型对近 5 日涨幅的追涨暴露。

N4 机制检验发现 PG1(训练多截断 1 天)的效果伴随追涨暴露
spearman(pred, mom5) 下降约 34% (+0.083 -> +0.055, 40/40 种子)。MN1 用来检验因果:
不改训练, 只在每个信号日把预测值里与 mom5 线性相关的部分去掉比例 k, 看能否复现收益。

  mom5[D]  = C[D] / C[D-5] - 1        (信号日收盘可知, 无未来函数)
  m        = rank_pct(mom5) - 0.5      (截面秩中心化; 缺 mom5 的股记 0 = 中性)
  beta     = cov(pred, m) / var(m)     (当日截面 OLS 斜率)
  pred'    = pred - k * beta * m       (k=1 完全去掉线性载荷, k=0 原样)

预测值的量纲不变(仍是预期 5 日超额收益), 下游的 min-pred 门槛、续持排名、补买排名
全部直接用 pred'。缓存里的 ic 字段不重算(只有 --ic-timing 用, 研究口径不开)。
"""
import numpy as np
import pandas as pd


def mom5_panel(klines, codes):
    """codes 的收盘价按各自 K 线日期对齐成面板, 返回 C/C.shift(5)-1 (行=日期, 列=6 位代码)"""
    close = {}
    for c in sorted({str(x)[:6] for x in codes}):
        kl = klines.get(c)
        if kl is not None and len(kl):
            close[c] = pd.Series(pd.to_numeric(kl["close"], errors="coerce").to_numpy(),
                                 index=pd.to_datetime(kl["date"])).where(lambda s: s > 0)
    if not close:
        return pd.DataFrame()
    px = pd.DataFrame(close).sort_index()
    return px / px.shift(5) - 1


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def neutralize_day(ranked, pred_vals, mom_row, k):
    """返回 (新排名代码, 新预测值, 处理前暴露, 处理后暴露)。

    ranked/pred_vals 一一对应; mom_row 是该信号日的 mom5 (index=6 位代码), 可为 None。
    k=0 时原样返回(逐位不变), 保证开关关闭即回到原口径。"""
    if not 0 <= k <= 1:
        raise ValueError("mom-neutral strength must be within [0, 1]")
    codes = list(ranked)
    vals = np.asarray(pred_vals, dtype=float)
    if len(codes) != len(vals):
        raise ValueError("ranked and pred_vals length differ")
    raw = np.full(len(codes), np.nan) if mom_row is None else \
        mom_row.reindex([str(c)[:6] for c in codes]).to_numpy(dtype=float)
    before = _spearman(vals, raw)
    if k == 0:
        return codes, list(pred_vals), before, before
    have = np.isfinite(raw)
    m = np.zeros(len(codes))
    if have.sum() >= 3:
        m[have] = pd.Series(raw[have]).rank(pct=True).to_numpy() - 0.5
    ok = np.isfinite(vals)
    var = float(np.var(m[ok]))
    beta = float(np.cov(vals[ok], m[ok], bias=True)[0, 1] / var) if var > 0 and ok.sum() >= 3 else 0.0
    adj = vals - k * beta * m
    order = np.argsort(-adj, kind="stable")
    return [codes[i] for i in order], [float(adj[i]) for i in order], before, _spearman(adj, raw)
