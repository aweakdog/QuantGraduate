"""验证 ret_5d 是滞后收益而非未来收益 (反转护栏的 PIT 前提)

护栏用 ret_5d 的截面分位拉黑候选股, 若 ret_5d 含未来信息, 整个结论都不成立。
对多只样本股, 从原始K线重算 close.pct_change(5) 与 close.shift(-5)/close-1,
分别与训练集里的 ret_5d 比对。
"""
import numpy as np
import pandas as pd

COLS = ["date", "code", "ret_5d", "fwd_5d_ret"]
t = pd.read_parquet("data/processed/training_data_pit_v24.parquet", columns=COLS)
t["date"] = pd.to_datetime(t["date"])

codes = list(pd.unique(t["code"]))[:6]
print(f"{'代码':<8}{'行数':>7}{'vs滞后5日 corr':>16}{'最大误差':>12}{'vs未来5日 corr':>16}")
for code in codes:
    c6 = str(code)[:6]
    try:
        k = pd.read_parquet(f"data/raw/kline/{c6}.parquet").rename(
            columns={"时间": "date", "收盘价": "close"})
    except Exception:
        continue
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date")
    k["trail5"] = k["close"].pct_change(5)
    k["fwd5"] = k["close"].shift(-5) / k["close"] - 1
    m = (t[t["code"] == code].merge(k[["date", "trail5", "fwd5"]], on="date", how="inner")
         .dropna(subset=["ret_5d", "trail5"]))
    if len(m) < 50:
        continue
    c_tr = m["ret_5d"].corr(m["trail5"])
    err = (m["ret_5d"] - m["trail5"]).abs().max()
    c_fw = m["ret_5d"].corr(m["fwd5"])
    print(f"{c6:<8}{len(m):>7}{c_tr:>16.6f}{err:>12.2e}{c_fw:>16.4f}")

s = t.dropna(subset=["ret_5d", "fwd_5d_ret"])
print(f"\n全样本 corr(ret_5d, fwd_5d_ret) = {s['ret_5d'].corr(s['fwd_5d_ret']):.4f}")
print("(接近 0 或轻微负 => 过去涨幅与未来收益基本无关/轻微反转, 符合预期)")
print("\n判定: vs滞后5日 corr≈1.000000 且误差≈0 => ret_5d 是纯滞后数据, 护栏 PIT 安全")
