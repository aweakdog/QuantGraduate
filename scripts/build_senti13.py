# 行为情绪增广层构建 — TODO#1 第一步
# 在 v2 矩阵上加 13 列（定义与 08-20 诊断脚本 /tmp/cgo_diag.py /tmp/senti_diag.py 完全一致）:
#   CGO 5 列: cgo_rec(递推参考成本,=诊断版cgo_120) / cgo_60 / cgo_250(截断窗加权RP)
#             / ovh_120(头顶套牢量) / cgo_delta_5d(接近速度)
#   情绪 6 列: max_ret20 / ret_skew60 / near_52wh / limitup_cnt20 / zhaban_cnt20 / to_spike_z
#   市场温度 2 列: mkt_cgo(全池cgo_rec日均) / mkt_limitup5(全池涨停率5日均)
# 输出: data/processed/training_data_pit_2015_tick1_v2s.parquet
#       data/processed/features_V24PUT_senti13.json (80 锁定特征 + 13 新列)
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path.home() / "quant-strategy"
MAT = ROOT / "data/processed/training_data_pit_2015_tick1_v2.parquet"
SRC_JSON = ROOT / "data/processed/wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json"
OUT_MAT = ROOT / "data/processed/training_data_pit_2015_tick1_v2s.parquet"
OUT_JSON = ROOT / "data/processed/features_V24PUT_senti13.json"
KDIR = ROOT / "data/raw/kline"

SENTI13 = ["cgo_rec", "cgo_60", "cgo_250", "ovh_120", "cgo_delta_5d",
           "max_ret20", "ret_skew60", "near_52wh", "limitup_cnt20",
           "zhaban_cnt20", "to_spike_z", "mkt_cgo", "mkt_limitup5"]


def win_rp(P, phi, W):
    """截断窗 turnover 加权参考成本: w_i = phi_i * prod_{j>i}(1-phi_j), 归一化"""
    n = len(P)
    out = np.full(n, np.nan)
    if n < W:
        return out
    q = np.lib.stride_tricks.sliding_window_view(1.0 - phi, W).copy()  # (n-W+1, W)
    # c[k] = prod_{m>k} q[m]: 右侧累乘 (翻转→cumprod→再翻转, 去掉自身)
    cr = np.cumprod(q[:, ::-1], axis=1)[:, ::-1]
    c = np.ones_like(q)
    c[:, :-1] = cr[:, 1:]
    w = np.lib.stride_tricks.sliding_window_view(phi, W) * c
    pw = np.lib.stride_tricks.sliding_window_view(P, W)
    tot = w.sum(axis=1)
    out[W - 1:] = (w * pw).sum(axis=1) / np.where(tot == 0, np.nan, tot)
    return out


mat = pd.read_parquet(MAT)
mat["date"] = pd.to_datetime(mat["date"])
mat["_code6"] = mat["code"].astype(str).str[:6].str.zfill(6)
codes = sorted(mat["_code6"].unique())
print(f"矩阵 {len(mat):,} 行 {len(codes)} 只, 现有列 {mat.shape[1]}")

rows, lu_rows = [], []
for ci, c in enumerate(codes):
    f = KDIR / f"{c}.parquet"
    if not f.exists():
        continue
    k = pd.read_parquet(f, columns=["date", "high", "low", "close", "volume", "turnover"])
    if len(k) < 300:
        continue
    k = k.sort_values("date").reset_index(drop=True)
    P = k["close"].values
    phi = k["turnover"].clip(0, 1).fillna(0).values
    # 递推 RP (与 cgo_diag 逐字一致)
    rp = np.empty(len(k)); rp[0] = P[0]
    for i in range(1, len(k)):
        rp[i] = phi[i] * P[i] + (1 - phi[i]) * rp[i - 1]
    cgo_rec = (P - rp) / np.where(P == 0, np.nan, P)
    # 截断窗变体
    cgo_60 = (P - win_rp(P, phi, 60)) / np.where(P == 0, np.nan, P)
    cgo_250 = (P - win_rp(P, phi, 250)) / np.where(P == 0, np.nan, P)
    # 头顶套牢量 (与 cgo_diag 一致)
    typ = (k["high"] + k["low"] + k["close"]).values / 3.0
    vol = k["volume"].fillna(0).values
    n, W = len(k), 120
    ovh = np.full(n, np.nan)
    if n > W:
        sw_t = np.lib.stride_tricks.sliding_window_view(typ, W)
        sw_v = np.lib.stride_tricks.sliding_window_view(vol, W)
        above = (sw_t > P[W - 1:, None]) * sw_v[: n - W + 1]
        tot = sw_v[: n - W + 1].sum(axis=1)
        ovh[W - 1:] = above.sum(axis=1) / np.where(tot == 0, np.nan, tot)
    # 情绪 6 列 (与 senti_diag 逐字一致)
    ret = k["close"].pct_change()
    lim = 0.198 if c[:2] in ("30", "68") else 0.098
    pc = k["close"].shift(1)
    is_lu = (ret >= lim).astype(float)
    touched = (k["high"] >= pc * (1 + lim)).astype(float)
    zb = ((touched == 1) & (is_lu == 0)).astype(float)
    to_mu = k["turnover"].rolling(60).mean()
    to_sd = k["turnover"].rolling(60).std()
    d = pd.DataFrame({
        "date": k["date"], "_code6": c,
        "cgo_rec": cgo_rec, "cgo_60": cgo_60, "cgo_250": cgo_250, "ovh_120": ovh,
        "max_ret20": ret.rolling(20).max(),
        "ret_skew60": ret.rolling(60).skew(),
        "near_52wh": k["close"] / k["close"].rolling(250).max(),
        "limitup_cnt20": is_lu.rolling(20).sum(),
        "zhaban_cnt20": zb.rolling(20).sum(),
        "to_spike_z": (k["turnover"] - to_mu) / to_sd.replace(0, np.nan),
    })
    d["cgo_delta_5d"] = d["cgo_rec"] - d["cgo_rec"].shift(5)
    lu_rows.append(pd.DataFrame({"date": k["date"], "is_lu": is_lu}))
    rows.append(d.iloc[260:])  # burn-in 与诊断一致
    if (ci + 1) % 200 == 0:
        print(f"  {ci+1}/{len(codes)}")

F = pd.concat(rows, ignore_index=True)
F["date"] = pd.to_datetime(F["date"])
# 市场温度 2 列 (全池口径)
mkt = F.groupby("date")["cgo_rec"].mean().rename("mkt_cgo").to_frame()
lu_all = pd.concat(lu_rows, ignore_index=True)
lu_all["date"] = pd.to_datetime(lu_all["date"])
lu_rate = lu_all.groupby("date")["is_lu"].mean()
mkt["mkt_limitup5"] = lu_rate.rolling(5).mean()
mkt = mkt.reset_index()

out = mat.merge(F, on=["date", "_code6"], how="left").merge(mkt, on="date", how="left")
out = out.drop(columns=["_code6"])
assert len(out) == len(mat), f"行数变了: {len(mat)} -> {len(out)}"
nn = {c: f"{out[c].notna().mean():.1%}" for c in SENTI13}
print("新列非空率:", nn)
out.to_parquet(OUT_MAT, index=False)
print(f"矩阵已存 {OUT_MAT} ({out.shape[0]:,} x {out.shape[1]})")

sel = json.load(open(SRC_JSON, encoding="utf-8"))["selected_features"]
ext = list(sel) + [c for c in SENTI13 if c not in sel]
json.dump({"selected_features": ext,
           "note": "V24PUT 80 锁定特征 + senti13 增广列, 2026-08-20"},
          open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"特征 json 已存 {OUT_JSON}: {len(sel)} + {len(ext)-len(sel)} = {len(ext)} 个")
print("DONE")
