"""build_b2_augmented 三族的算法钉死 —— 增广列最容易静默算错的三处:
T1C 的 PIT(只能用往年)、T2D 的 lag、T3 的剔自身。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_b2_augmented as B  # noqa: E402


def _write_kline(dirp, code, dates, close, amount=1e8, open_=None):
    dirp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "open": open_ if open_ is not None else close,
                  "close": close, "amount": amount}).to_parquet(
        dirp / f"{code}.parquet", index=False)


# ── T1C ──────────────────────────────────────────────────────
def test_t1c_uses_only_prior_years_same_month(tmp_path, monkeypatch):
    """2024-03 的值 = 2021/2022/2023 三个 3 月的月收益均值, 不含 2024-03 自己"""
    monkeypatch.setattr(B, "KLINE_DIR", tmp_path / "kline")
    days = pd.bdate_range("2020-12-01", "2024-04-30")
    # 造价格: 每年 3 月涨 +10%, 其他月份持平 -> 同月均值 0.10, 其他月 0
    close = np.ones(len(days))
    lvl = 1.0
    for i, d in enumerate(days):
        if d.month == 3 and (i == 0 or days[i - 1].month != 3):
            lvl *= 1.10
        close[i] = lvl
    _write_kline(tmp_path / "kline", "000001", days, close)

    f = B.t1c_frame(["000001"]).set_index(["y", "m"])
    v = f.loc[(2024, 3)]
    assert v["t1c_seas"] == pytest.approx(0.10, abs=1e-9)
    assert v["t1c_seas_hit"] == pytest.approx(1.0)
    assert v["t1c_seas_diff"] == pytest.approx(0.10, abs=1e-9)
    # 2024-04: 往年 4 月都持平 -> 0, 且 diff 为负(其他月含 3 月的涨)
    v4 = f.loc[(2024, 4)]
    assert v4["t1c_seas"] == pytest.approx(0.0, abs=1e-9)
    assert v4["t1c_seas_diff"] < 0
    # 不足 3 个往年观测的月份不能出值: 2022-03 只有 2021 一个往年 3 月
    assert (2022, 3) not in f.index
    assert (2023, 3) not in f.index


# ── T2D ──────────────────────────────────────────────────────
def test_t2d_lag1_and_change_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "MARGIN_DIR", tmp_path / "margin")
    (tmp_path / "margin").mkdir()
    days = pd.bdate_range("2024-01-01", periods=40)
    rzye = np.linspace(1e9, 1.39e9, 40)          # 每天 +1e7
    pd.DataFrame({"trade_date": days.strftime("%Y%m%d"), "ts_code": "000001.SZ",
                  "rzye": rzye, "rzmre": 5e7, "rzche": 4e7}).to_parquet(
        tmp_path / "margin" / "2024.parquet", index=False)
    kl = pd.DataFrame({"_c6": "000001", "date": days, "amount": 1e9})

    f0 = B.t2d_frame(["000001"], kl, lag=0).set_index("date")
    f1 = B.t2d_frame(["000001"], kl, lag=1).set_index("date")
    d, prev = days[30], days[29]
    # 5 日变化率 = rzye[t]/rzye[t-5]-1
    assert f0.loc[d, "t2d_rz_chg5"] == pytest.approx(rzye[30] / rzye[25] - 1)
    # lag1: 矩阵日 D 拿到的是 D-1 的值 (交易所 D+1 早才公布)
    assert f1.loc[d, "t2d_rz_chg5"] == pytest.approx(f0.loc[prev, "t2d_rz_chg5"])
    # 净买入 5 日 = 5*(5e7-4e7)/rzye
    assert f0.loc[d, "t2d_rz_net5"] == pytest.approx(5 * 1e7 / rzye[30])
    # 占比恒定 -> std=0 -> z 必须是 NaN 而不是 inf
    assert np.isnan(f0.loc[d, "t2d_rz_buy_z"])
    assert B.T2D_LAG == 1, "生产口径默认必须滞后一天"


# ── T3 ───────────────────────────────────────────────────────
def test_t3_excludes_self_and_respects_pit_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "SW_MEMBER", tmp_path / "sw_member.parquet")
    days = pd.bdate_range("2024-01-01", periods=30)
    # 7 只同行业: 6 只每日 +1%, 1 只(目标)每日 +3%; 目标行业均值应排除自身 = 6 只的 r5
    parts = []
    for i in range(7):
        rate = 0.03 if i == 0 else 0.01
        close = (1 + rate) ** np.arange(30)
        parts.append(pd.DataFrame({"_c6": f"00000{i}", "date": days, "close": close}))
    # 第 8 只 2024-01-20 才调入该行业 (之前属别的行业)
    parts.append(pd.DataFrame({"_c6": "000009", "date": days, "close": 1.02 ** np.arange(30)}))
    kl = pd.concat(parts, ignore_index=True)
    sw = pd.DataFrame({
        "ts_code": [f"00000{i}.SZ" for i in range(7)] + ["000009.SZ", "000009.SZ"],
        "l1_name": ["银行"] * 7 + ["电子", "银行"],
        "in_date": ["20100101"] * 7 + ["20100101", "20240120"],
        "out_date": [None] * 7 + ["20240119", None],
    })
    sw.to_parquet(tmp_path / "sw_member.parquet", index=False)

    f = B.t3_frame(kl).set_index(["date", "_c6"])
    d_early, d_late = days[10], days[25]
    r5_peer = 5 * np.log(1.01)
    # 调入前: 目标的行业均值 = 6 只 1% 股的 r5 (剔除自身的 3%)
    assert f.loc[(d_early, "000000"), "t3_ind_ret5"] == pytest.approx(r5_peer, rel=1e-6)
    assert f.loc[(d_early, "000000"), "t3_rel5"] == pytest.approx(5 * np.log(1.03) - r5_peer, rel=1e-6)
    # 调入后: 000009 (2%) 进入银行 -> 目标的行业均值上移
    assert f.loc[(d_late, "000000"), "t3_ind_ret5"] > r5_peer
    # 调入前 000009 所在行业("电子")只有它一只 -> 峠不到 5 只同伴, 不出值
    assert np.isnan(f.loc[(d_early, "000009"), "t3_ind_ret5"])
