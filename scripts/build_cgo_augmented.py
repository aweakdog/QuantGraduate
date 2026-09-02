"""CGO 行为特征列增广 —— GF1 门控融合的数据层 (2026-08-22)

把 6 个 CGO(资本利得悬垂)列加到线上训练矩阵上, 供 live_signal --gate-ma60
在市场弱势态切换 CGO 模型使用:
    cgo_rec       递推参考成本悬垂: (P - RP) / P, RP 用换手率递推
    cgo_60        截断 60 日窗 turnover 加权参考成本版
    cgo_250       截断 250 日窗版
    ovh_120       头顶套牢量: 120 日内成交于当前价上方的量占比
    cgo_delta_5d  cgo_rec 5 日变化 (接近解套的速度)
    mkt_cgo       全池 cgo_rec 日均 (市场温度)

算法与 2026-08-20 研究构建 (/tmp/build_senti13.py, 产出 v2s 矩阵) 逐字一致 ——
这是回测证据 (GF1-G1 20 种子过 gate, 见 experiment_board 2026-08-22) 的口径,
任何"顺手优化"都会让线上模型脱离已验证的构造。特别地:
  * phi = turnover.clip(0, 1): 与研究版同, 单位跟着 kline 文件走, 不做换算
  * burn-in 260 行: 距上市不足一年的股票 CGO 留 NaN (LightGBM 原生处理)
  * mkt_cgo 在"矩阵里出现过的股票"上取日均, 池子跟矩阵走

用法:
    python scripts/build_cgo_augmented.py                  # 原地增广线上 v24 矩阵
    python scripts/build_cgo_augmented.py --source X.parquet --output Y.parquet

daily_rebuild §4.2 每晚在 feature_engine 重建矩阵之后调用 (重建会丢掉旧增广列,
所以必须每晚重加)。幂等: 已有 CGO 列会先剔除再重算。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
KLINE_DIR = ROOT / "data" / "raw" / "kline"

CGO_COLS = ["cgo_rec", "cgo_60", "cgo_250", "ovh_120", "cgo_delta_5d", "mkt_cgo"]
BURN_IN = 260          # 与研究构建一致: 首年不出值
MIN_KLINE_ROWS = 300   # K线太短的股票整只留 NaN (与研究构建一致)

# K线列名: 服务器上已是英文列; 兜住可能出现的中文列老文件
_COL_MAP = {"时间": "date", "开盘价": "open", "最高价": "high",
            "最低价": "low", "收盘价": "close", "成交量": "volume",
            "换手率": "turnover"}


def win_rp(P, phi, W):
    """截断窗 turnover 加权参考成本: w_i = phi_i * prod_{j>i}(1-phi_j), 归一化"""
    n = len(P)
    out = np.full(n, np.nan)
    if n < W:
        return out
    q = np.lib.stride_tricks.sliding_window_view(1.0 - phi, W).copy()
    cr = np.cumprod(q[:, ::-1], axis=1)[:, ::-1]
    c = np.ones_like(q)
    c[:, :-1] = cr[:, 1:]
    w = np.lib.stride_tricks.sliding_window_view(phi, W) * c
    pw = np.lib.stride_tricks.sliding_window_view(P, W)
    tot = w.sum(axis=1)
    out[W - 1:] = (w * pw).sum(axis=1) / np.where(tot == 0, np.nan, tot)
    return out


def cgo_for_code(kline_path):
    """单只股票的 5 个个股级 CGO 列 (mkt_cgo 之后在全池聚合)"""
    k = pd.read_parquet(kline_path)
    k = k.rename(columns=_COL_MAP)
    need = {"date", "high", "low", "close", "volume", "turnover"}
    if not need.issubset(k.columns) or len(k) < MIN_KLINE_ROWS:
        return None
    k = k[["date", "high", "low", "close", "volume", "turnover"]].copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date").reset_index(drop=True)
    P = k["close"].values.astype(float)
    phi = k["turnover"].clip(0, 1).fillna(0).values.astype(float)
    # 递推 RP (逐字复刻研究版)
    rp = np.empty(len(k))
    rp[0] = P[0]
    for i in range(1, len(k)):
        rp[i] = phi[i] * P[i] + (1 - phi[i]) * rp[i - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cgo_rec = (P - rp) / np.where(P == 0, np.nan, P)
        cgo_60 = (P - win_rp(P, phi, 60)) / np.where(P == 0, np.nan, P)
        cgo_250 = (P - win_rp(P, phi, 250)) / np.where(P == 0, np.nan, P)
    # 头顶套牢量
    typ = (k["high"] + k["low"] + k["close"]).values / 3.0
    vol = k["volume"].fillna(0).values.astype(float)
    n, W = len(k), 120
    ovh = np.full(n, np.nan)
    if n > W:
        sw_t = np.lib.stride_tricks.sliding_window_view(typ, W)
        sw_v = np.lib.stride_tricks.sliding_window_view(vol, W)
        above = (sw_t > P[W - 1:, None]) * sw_v[: n - W + 1]
        tot = sw_v[: n - W + 1].sum(axis=1)
        ovh[W - 1:] = above.sum(axis=1) / np.where(tot == 0, np.nan, tot)
    d = pd.DataFrame({"date": k["date"], "cgo_rec": cgo_rec, "cgo_60": cgo_60,
                      "cgo_250": cgo_250, "ovh_120": ovh})
    d["cgo_delta_5d"] = d["cgo_rec"] - d["cgo_rec"].shift(5)
    return d.iloc[BURN_IN:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="training_data_pit_v24.parquet")
    ap.add_argument("--output", default=None,
                    help="默认原地增广 (tmp + 原子替换)")
    a = ap.parse_args()
    t0 = time.time()

    src = PROC / a.source
    out_path = PROC / (a.output or a.source)
    mat = pd.read_parquet(src)
    mat["date"] = pd.to_datetime(mat["date"])
    # 幂等: 剔掉旧 CGO 列再重算
    mat = mat.drop(columns=[c for c in CGO_COLS if c in mat.columns])
    c6 = mat["code"].astype(str).str.extract(r"(\d{6})")[0]
    codes = sorted(c6.dropna().unique())
    print(f"矩阵 {len(mat):,} 行 {len(codes)} 只 (源 {src.name})")

    rows, skipped = [], 0
    for i, c in enumerate(codes):
        p = KLINE_DIR / f"{c}.parquet"
        if not p.exists():
            skipped += 1
            continue
        try:
            d = cgo_for_code(p)
        except Exception as e:
            print(f"  WARN {c}: {e}")
            d = None
        if d is None:
            skipped += 1
            continue
        d.insert(1, "_c6", c)
        rows.append(d)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(codes)} ({time.time() - t0:.0f}s)")
    if not rows:
        print("ERROR: 没有任何股票算出 CGO")
        return 2
    F = pd.concat(rows, ignore_index=True)
    # 市场温度: 全池 cgo_rec 日均 (池 = 矩阵里的股票)
    mkt = F.groupby("date")["cgo_rec"].mean().rename("mkt_cgo").reset_index()

    mat["_c6"] = c6
    n0 = len(mat)
    out = mat.merge(F, on=["date", "_c6"], how="left").merge(mkt, on="date", how="left")
    out = out.drop(columns=["_c6"])
    if len(out) != n0:
        print(f"ERROR: 合并后行数变了 {n0:,} -> {len(out):,} (K线有重复日?)")
        return 2

    # 末日体检: 门控模型晚上要用当天的值, 末日覆盖塌了宁可整步失败
    last = out[out["date"] == out["date"].max()]
    cov = last["cgo_rec"].notna().mean() if len(last) else 0.0
    if cov < 0.85:
        print(f"ERROR: 最新日 cgo_rec 覆盖率仅 {cov:.1%} (<85%), 拒绝落盘")
        return 2

    tmp = out_path.with_suffix(".parquet.cgotmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    nn = {c: f"{out[c].notna().mean():.1%}" for c in CGO_COLS}
    print(f"CGO 增广完成 -> {out_path.name} ({out.shape[0]:,} x {out.shape[1]}) "
          f"跳过 {skipped} 只, 耗时 {time.time() - t0:.0f}s")
    print(f"非空率: {nn} | 最新日 cgo_rec 覆盖 {cov:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
