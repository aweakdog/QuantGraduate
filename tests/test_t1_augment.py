"""T1A/T1B 增广列的算法口径契约。

为什么钉死在测试里
──────────────────
这 8 列的回测证据(T1A20 +18.6pp p≈0.004 / T1B20 +26.45pp p≈0.012, 20 面板)
是用特定构造跑出来的。任何"顺手优化"——放宽 min_periods、改大单分位、忘了剔
深交所撤单——都会让线上模型脱离已验证口径, 而且**不会报错**, 只会悄悄变差。

入库时已做过一次全量口径回归(2026-08-31): 新脚本 vs 研究版原代码在同一份
kline/面板上 T1A 逐值全等、T1B 1,124,010 行零差异; 与研究矩阵存档值的
0.57% 差异经隔离确认只出现在 3/519 只近期复权重算的股票上(qfq 前复权改锚会
重算全历史的舍入位, 比率口径漂移无害)。那次是一次性的重活, 这里用手算小样本
把公式本身锁住, 让每次 pytest 都能挡住回归。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_t1_augmented as B
import t1a_order_features as T


def _trades(orders, flag, idc):
    """把 {订单号: [每笔成交量...]} 摊平成逐笔成交行"""
    rows = []
    for oid, fills in orders.items():
        for q in fills:
            rows.append({"成交代码": "0", "BS标志": flag, "成交数量": q,
                         "叫买序号": oid if idc == "叫买序号" else 0,
                         "叫卖序号": oid if idc == "叫卖序号" else 0})
    return rows


def _both_sides(orders):
    """买卖两侧用同一组订单, 便于只断言一侧"""
    return pd.DataFrame(_trades(orders, "B", "叫买序号")
                        + _trades(orders, "S", "叫卖序号"))


def _ramp_orders():
    """30 个订单, 第 i 个成交 i 股 (严格递增 -> 0.9 分位无歧义)

    故意不用“一个大单 + 一堆等量小单”: 那种数据里分位数正好落在小单的
    重复值上, “大单”会把全部小单一起包进来, 测不出真正的分位行为。
    订单 5 与 10 拆成 3 笔 (算多笔), 订单 7 拆成 2 笔 (不算), 其余单笔。
    """
    orders = {i: [i] for i in range(1, 31)}
    orders[5] = [2, 2, 1]
    orders[7] = [4, 3]
    orders[10] = [4, 3, 3]
    return orders


def test_t1a_big_and_long_ratios_hand_computed():
    r = T.ratios_for_code(_both_sides(_ramp_orders()), is_sz=False)
    assert r is not None

    total = sum(range(1, 31))                     # 465
    # quantile(0.9) 于 [1..30] = 27.1 -> 大单 = 订单 28/29/30 (恰好前 10%)
    assert r["t1a_big_buy"] == pytest.approx((28 + 29 + 30) / total)
    # 多笔(>=3 笔) = 订单 5 与 10; 订单 7 只 2 笔 不算
    assert r["t1a_long_buy"] == pytest.approx((5 + 10) / total)
    assert r["t1a_big_sell"] == pytest.approx(r["t1a_big_buy"])


def test_t1a_thin_book_returns_none():
    """每侧订单数 < 30 整码丢弃 —— 分位数在小样本上没有意义"""
    assert T.ratios_for_code(_both_sides({i: [10] for i in range(1, 30)}),
                             is_sz=False) is None
    assert T.MIN_ORDERS == 30, "改这个阈值等于换族, 必须重跑证据链"


def test_t1a_drops_shenzhen_cancels():
    """SZ 撤单在【成交】文件里(成交代码=='C'), 不剔就把撤掉的量算成成交"""
    df = _both_sides(_ramp_orders())
    fake = df.iloc[[0]].copy()
    fake["成交代码"] = "C"
    fake["叫买序号"] = fake["叫卖序号"] = 31
    fake["成交数量"] = 10 ** 6            # 一笔巨额撤单
    polluted = pd.concat([df, fake], ignore_index=True)

    clean = T.ratios_for_code(df, is_sz=False)
    sz = T.ratios_for_code(polluted, is_sz=True)
    assert sz["t1a_big_buy"] == pytest.approx(clean["t1a_big_buy"])
    # 沪市不剔(撤单在委托文件里), 巨额行会污染 -> 必须明显不同,
    # 否则说明过滤被无条件执行了(那就会把沪市真成交当成撤单剔掉)
    sh = T.ratios_for_code(polluted, is_sz=False)
    assert sh["t1a_big_buy"] > 0.99 > clean["t1a_big_buy"]


def test_t1a_drops_nonpositive_order_ids():
    """序号 <=0 是集合竞价/异常行, 必须丢弃"""
    df = _both_sides(_ramp_orders())
    junk = df.iloc[[0]].copy()
    junk["叫买序号"] = 0
    junk["叫卖序号"] = 0
    junk["成交数量"] = 10 ** 6
    r = T.ratios_for_code(pd.concat([df, junk], ignore_index=True), is_sz=False)
    assert r["t1a_big_buy"] == pytest.approx(
        T.ratios_for_code(df, is_sz=False)["t1a_big_buy"])


def test_t1a_ma5_window_params_locked():
    assert (B.T1A_MA, B.T1A_MA_MIN) == (5, 3), "ma5/min3 是回测口径"
    assert B.T1A_COLS == [c + "_ma5" for c in B.T1A_RAW]


def _write_kline(d: Path, code: str, n: int):
    """造一段可手算的 K 线: close 恒 10, open 恒 11 -> intra<0, ovn>0"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "open": 11.0, "close": 10.0}).to_parquet(
        d / f"{code}.parquet", index=False)


def test_t1b_formula_and_full_window_min_periods(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "KLINE_DIR", tmp_path)
    _write_kline(tmp_path, "000001", 70)
    f = B.t1b_frame(["000001"]).sort_values("date").reset_index(drop=True)

    # 每日 intra = log(10/11), ovn = log(11/10) (第 1 天 ovn 无前收 -> NaN)
    intra1, ovn1 = np.log(10 / 11), np.log(11 / 10)
    assert f["t1b_intra20"].iloc[19] == pytest.approx(20 * intra1)
    assert f["t1b_intra60"].iloc[59] == pytest.approx(60 * intra1)
    # ovn 首日 NaN 使窗口右移一天: 第 20 行(idx 19)还凑不满 20 个有效值
    assert np.isnan(f["t1b_ovn20"].iloc[19])
    assert f["t1b_ovn20"].iloc[20] == pytest.approx(20 * ovn1)

    # 满窗 min_periods: 不足窗长一律 NaN, 不许用 min_periods 放宽
    assert f["t1b_intra20"].iloc[:19].isna().all()
    assert f["t1b_intra60"].iloc[:59].isna().all()


def test_t1b_skips_short_and_missing_codes(tmp_path, monkeypatch):
    """K线太短只该给 NaN, 不该报错; 缺文件的股票直接跳过"""
    monkeypatch.setattr(B, "KLINE_DIR", tmp_path)
    _write_kline(tmp_path, "000001", 5)
    f = B.t1b_frame(["000001", "999999"])
    assert set(f["_c6"]) == {"000001"}
    assert f[B.T1B_COLS].isna().all().all()


def test_t1b_all_missing_codes_raises(tmp_path, monkeypatch):
    """一只都算不出必须炸 —— 静默产出空表会让夜链把矩阵写成全 NaN"""
    monkeypatch.setattr(B, "KLINE_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        B.t1b_frame(["999999"])


def test_column_names_are_frozen():
    """列名进了特征表和已训练的证据链, 改名等于换族"""
    assert B.T1A_COLS == ["t1a_big_buy_ma5", "t1a_long_buy_ma5",
                          "t1a_big_sell_ma5", "t1a_long_sell_ma5"]
    assert B.T1B_COLS == ["t1b_ovn20", "t1b_ovn60", "t1b_intra20", "t1b_intra60"]
    assert set(T.T1A_COLS) == set(B.T1A_RAW), "抽取器与增广器的原始列名必须对齐"


def test_min_cover_thresholds_locked():
    """末日覆盖闸: 与研究构建的 assert 同阈值, 塌了宁可整步失败"""
    assert B.MIN_COVER == {"t1a": 0.90, "t1b": 0.85}


def test_augment_cols_exemption_covers_every_built_column():
    """夜链的缺列豁免必须涵盖所有增广列。

    漏一个的后果很隐蔽: feature_engine 每晚重建矩阵会丢掉增广列, 而
    validate_new_train 拿"新矩阵缺了老矩阵的列"当拦截依据 —— 于是从第二晚起
    每晚硬拦, 全线没信号。
    """
    import daily_rebuild as D
    built = set(B.T1A_COLS) | set(B.T1B_COLS)
    assert built <= D.AUGMENT_COLS, f"缺豁免: {sorted(built - D.AUGMENT_COLS)}"


def _fake_panel(tmp: Path, days, code="000001", val0=0.5):
    """造 t1a_daily 面板: 第 i 天的 big_buy = val0 + i/100, 便于肉眼验滞后"""
    tmp.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(days):
        row = {"date": d, "code": code}
        for j, c in enumerate(B.T1A_RAW):
            row[c] = val0 + i / 100 + j
        pd.DataFrame([row]).to_parquet(tmp / f"{d}.parquet", index=False)


def test_t1a_lag1_shifts_by_one_panel_day(tmp_path, monkeypatch):
    """矩阵行 D 必须拿到面板 D-1 的值 —— 逐笔包 D+1 早晨才到"""
    monkeypatch.setattr(B, "T1A_PANEL", tmp_path / "t1a_daily")
    days = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    _fake_panel(tmp_path / "t1a_daily", days)

    f0 = B.t1a_frame(lag=0).set_index("date")
    f1 = B.t1a_frame(lag=1).set_index("date")
    d = pd.Timestamp("2024-01-08")
    prev = pd.Timestamp("2024-01-05")
    col = B.T1A_COLS[0]
    assert f1.loc[d, col] == pytest.approx(f0.loc[prev, col])
    # 首个面板日在 lag1 下必须没有值(没有前一天可领)
    assert np.isnan(f1.loc[pd.Timestamp("2024-01-02"), col])


def test_t1a_edge_padding_fills_signal_day(tmp_path, monkeypatch):
    """面板末日总比矩阵末日早一天 —— 不补行, 信号日那行会静默全 NaN

    这是 2026-08-31 端到端跑时被覆盖闸挡住的那个真问题。
    """
    monkeypatch.setattr(B, "T1A_PANEL", tmp_path / "t1a_daily")
    days = ["20240102", "20240103", "20240104", "20240105"]
    _fake_panel(tmp_path / "t1a_daily", days)
    # 基矩阵多出一天 (01-08), 面板还没有 —— 正是每晚 17:30 的真实状态
    base = pd.DataFrame({
        "_c6": "000001",
        "date": pd.to_datetime(["20240102", "20240103", "20240104",
                                "20240105", "20240108"]),
    })
    col = B.T1A_COLS[0]
    sig = pd.Timestamp("2024-01-08")

    no_pad = B.t1a_frame(lag=1).set_index("date")
    assert sig not in no_pad.index, "不传 base_keys 就没有信号日那行"

    padded = B.t1a_frame(base_keys=base, lag=1).set_index("date")
    panel_last = B.t1a_frame(lag=0).set_index("date").loc[
        pd.Timestamp("2024-01-05"), col]
    assert padded.loc[sig, col] == pytest.approx(panel_last), \
        "信号日应领到面板末日(01-05)的值"


def test_t1a_lag_default_is_production_safe():
    """默认值必须是 1: 有人忘传参数时, 宁可保守也不能静默用上前视口径"""
    assert B.T1A_LAG == 1
    assert B.EDGE_MAX == 3, "与 build_tick_augmented 同值, 改一处要同步"


def test_nightly_hardfail_arms_itself_on_t1_feature_names():
    """§4.3 靠列名前缀判断"线上是否在用", 所以前缀不能改

    切线时只改 features-from, 没人会想起来回夜链改开关 —— 硬失败必须靠
    特征表里的列名自动上膜。
    """
    for c in list(B.T1A_COLS) + list(B.T1B_COLS):
        assert c.startswith(("t1a_", "t1b_")), c
