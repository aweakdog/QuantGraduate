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
    # 调入前 000009 所在行业("电子")只有它一只 -> 凑不到 5 只同伴, 不出值
    assert np.isnan(f.loc[(d_early, "000009"), "t3_ind_ret5"])


# ── T2C 解禁 ─────────────────────────────────────────────────
def test_t2c_event_known_only_after_announcement(tmp_path, monkeypatch):
    """公告 01-10 (盘后) 解禁 01-20: 01-10 当天不能知道, 01-11 起知道; 解禁后进 past20"""
    monkeypatch.setattr(B, "SHARE_FLOAT_DIR", tmp_path / "sf")
    (tmp_path / "sf").mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ"], "ann_date": ["20240110", None],
                  "float_date": ["20240120", "20240301"], "float_ratio": [5.0, 2.0]}).to_parquet(
        tmp_path / "sf" / "2024.parquet", index=False)
    keys = pd.DataFrame({"_c6": "000001", "date": pd.bdate_range("2024-01-02", "2024-03-15")})
    f = B.t2c_frame(keys).set_index("date")
    T = pd.Timestamp
    assert f.loc[T("2024-01-10"), "t2c_days_next"] == B.T2C_HORIZON      # 公告日当天还不知道
    assert f.loc[T("2024-01-10"), "t2c_ratio_next"] == 0.0
    assert f.loc[T("2024-01-11"), "t2c_days_next"] == 9                  # 01-11 -> 01-20
    assert f.loc[T("2024-01-11"), "t2c_ratio_next"] == 5.0
    assert f.loc[T("2024-01-11"), "t2c_ratio_fwd60"] == 5.0
    # 解禁当日 (01-20 是周六, 首个交易日 01-22): 已不在"未来", 进 past20
    assert f.loc[T("2024-01-22"), "t2c_ratio_past20"] == 5.0
    assert f.loc[T("2024-01-22"), "t2c_days_next"] == B.T2C_HORIZON
    # 28 个日历日后滑出 past20
    assert f.loc[T("2024-02-19"), "t2c_ratio_past20"] == 0.0
    # 无公告日的事件(03-01) 解禁前不可见, 解禁后可见
    assert f.loc[T("2024-02-28"), "t2c_ratio_fwd60"] == 0.0
    assert f.loc[T("2024-03-01"), "t2c_ratio_past20"] == 2.0


# ── T2B 龙虎榜 ───────────────────────────────────────────────
def test_t2b_rolling_and_lag(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "TOP_INST_DIR", tmp_path / "ti")
    (tmp_path / "ti").mkdir()
    days = pd.bdate_range("2024-01-01", periods=40)
    listed = days[25]
    pd.DataFrame({"trade_date": [listed.strftime("%Y%m%d")] * 2, "ts_code": "000001.SZ",
                  "exalter": ["机构专用", "某营业部"], "net_buy": [3e6, -9e6]}).to_parquet(
        tmp_path / "ti" / "2024.parquet", index=False)
    kl = pd.DataFrame({"_c6": "000001", "date": days, "amount": 1e8})
    f0 = B.t2b_frame(kl, lag=0).set_index("date")
    f1 = B.t2b_frame(kl, lag=1).set_index("date")
    assert f0.loc[listed, "t2b_cnt20"] == 1 and f0.loc[days[24], "t2b_cnt20"] == 0
    assert f0.loc[listed, "t2b_days_since"] == 0 and f0.loc[days[28], "t2b_days_since"] == 3
    assert f0.loc[days[24], "t2b_days_since"] == B.T2B_SINCE_CAP        # 从未上榜 = 上限
    # 只算机构席位: 3e6 / (20 天 * 1e8)
    assert f0.loc[listed, "t2b_inst_net20"] == pytest.approx(3e6 / 2e9)
    # lag1: 上榜信息次日才进特征
    assert f1.loc[listed, "t2b_cnt20"] == 0 and f1.loc[days[26], "t2b_cnt20"] == 1
    assert B.T2B_LAG == 1


# ── T3H 股东户数 ─────────────────────────────────────────────
def test_t3h_latest_known_report_and_mixed_ann_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "HOLDER_DIR", tmp_path / "hn")
    (tmp_path / "hn").mkdir()
    pd.DataFrame({"ts_code": "000001.SZ",
                  "ann_date": ["20240110", "2024-04-20 15:09:08", "20240720"],
                  "end_date": ["20231231", "20240331", "20240630"],
                  "holder_num": [100000, 110000, 99000]}).to_parquet(
        tmp_path / "hn" / "2024.parquet", index=False)
    keys = pd.DataFrame({"_c6": "000001", "date": pd.bdate_range("2024-01-02", "2024-08-30")})
    f = B.t3h_frame(keys).set_index("date")
    T = pd.Timestamp
    assert np.isnan(f.loc[T("2024-01-11"), "t3h_chg"])                 # 首份报告没有环比
    assert np.isnan(f.loc[T("2024-04-19"), "t3h_chg"])                 # 04-20(周六)公告前不可见
    # 带时分秒的 ann_date 也要解析对: 04-21 起已知, 首个交易日 04-22 拿到 +10%
    assert f.loc[T("2024-04-22"), "t3h_chg"] == pytest.approx(np.log(1.1))
    assert f.loc[T("2024-04-22"), "t3h_days"] == 1
    assert f.loc[T("2024-07-19"), "t3h_chg"] == pytest.approx(np.log(1.1))   # 07-20 公告前仍是上一份
    assert f.loc[T("2024-07-22"), "t3h_chg"] == pytest.approx(np.log(0.9))
    assert f.loc[T("2024-07-22"), "t3h_days"] == 1
