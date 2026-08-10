"""回归测试: 源列整列无值时, 特征引擎不得把"数据没有"伪造成"值为 0"

事故背景 (2026-08-05)
────────────────────
pull_fundflow_shard.do_merge() 名为合并实为整表覆盖, 把 consolidated 资金流表里
新浪源拿不到的三列 (dde_net / mtss_balance / fund_flow) 整段历史抹成 NaN。
紧接着 calc_fund_features 的 `s = df[src].fillna(0)` 把整列 NaN 填成 0, 于是
线上 80 个入选特征里有 11 个在整个训练集上恒为常量 0, 而全程没有任何报错:

  - 恒常量特征对 LightGBM 毫无信息量, 模型名义 80 特征、实际只有 69 个
  - 更糟的是它还是截面偏差: 有真值的股票和填 0 的股票被放在同一个排序里
  - 而模型是在这些特征有真值的年代筛选/训练的, 学到的分裂点全部落向同一边

两层静默叠加, 所以本文件同时锁住两侧行为:
  1. 整列无值 -> 不生成该组特征 (留缺失, LightGBM 原生处理 NaN)
  2. 散点缺失 -> 仍然生成 (填 0 是可接受的插补, 且不会让特征失去方差)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
except ModuleNotFoundError:      # 本仓库的运行环境里没装 pytest, 不能因此就无法跑回归
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.feature_engine import calc_fund_features, calc_margin_features  # noqa: E402


class _Skip(Exception):
    pass


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


N = 60
LEGACY = ["dde_net", "mtss_balance", "fund_flow"]


def _base(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "date": pd.bdate_range("2024-01-01", periods=N),
        "main_force_net": rng.normal(0, 1e6, N),
        "main_force_pct": rng.normal(0, 1, N),
    }


def test_all_nan_source_produces_no_feature():
    """整列 NaN 的源列不得产出特征列 —— 这正是事故里被伪造成常量 0 的那批。"""
    df = pd.DataFrame({**_base(), **{c: [np.nan] * N for c in LEGACY}})
    feats = calc_fund_features(df)
    for dst in ("dde_net", "mtss", "fund_flow"):
        assert f"{dst}_1d" not in feats.columns, f"{dst}_1d 不该被凭空造出来"
        assert f"{dst}_z" not in feats.columns, f"{dst}_z 不该被凭空造出来"


def test_live_columns_unaffected_when_legacy_dead():
    """独有列全空时, 仍然活着的 main_force_* 特征必须照常产出。"""
    df = pd.DataFrame({**_base(), **{c: [np.nan] * N for c in LEGACY}})
    feats = calc_fund_features(df)
    for c in ("mf_net_1d", "mf_net_z", "mf_pct_1d", "mf_pct_z", "mf_signal"):
        assert c in feats.columns, f"缺少 {c}"
    assert feats["mf_net_1d"].nunique() > 1


def test_real_values_still_produce_varying_features():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({**_base(1), "dde_net": rng.normal(0, 1e5, N),
                       "mtss_balance": rng.normal(0, 1e7, N),
                       "fund_flow": rng.normal(0, 1e5, N)})
    feats = calc_fund_features(df)
    for dst in ("dde_net", "mtss", "fund_flow"):
        assert f"{dst}_1d" in feats.columns
        assert feats[f"{dst}_1d"].nunique() > 1, f"{dst}_1d 不该是常量"


def test_sparse_gaps_are_still_imputed():
    """散点缺失仍然填 0 并产出特征 —— 修复不该把正常插补也一起否掉。"""
    rng = np.random.default_rng(2)
    dde = rng.normal(0, 1e5, N)
    dde[[3, 7, 20]] = np.nan
    df = pd.DataFrame({**_base(2), "dde_net": dde,
                       **{c: [np.nan] * N for c in ("mtss_balance", "fund_flow")}})
    feats = calc_fund_features(df)
    assert "dde_net_1d" in feats.columns
    assert feats["dde_net_1d"].nunique() > 1
    assert feats["dde_net_1d"].iloc[3] == 0.0


def test_level_column_is_forward_filled_not_zero_filled():
    """存量列(融资融券余额)尾部缺失必须前值填充, 不能填 0。

    填 0 等于声称"融资余额一夜归零": 21 日 z-score 窗口里混进一串零之后,
    全市场每只股票都会得到一个巨大的假负值 —— 那比恒常量 0 更坏, 因为它是
    有方向的谎言, 模型会当成真信号去用。
    """
    rng = np.random.default_rng(4)
    bal = rng.normal(1e8, 1e6, N)
    bal[-8:] = np.nan          # 源滞后 8 个交易日, 正是断更时的形态
    df = pd.DataFrame({**_base(4), "mtss_balance": bal,
                       **{c: [np.nan] * N for c in ("dde_net", "fund_flow")}})
    feats = calc_fund_features(df)
    assert "mtss_1d" in feats.columns
    tail = feats["mtss_1d"].iloc[-8:]
    assert (tail > 1e7).all(), f"尾部被填成了 0 或过小: {tail.tolist()[:3]}"
    last_real = bal[~np.isnan(bal)][-1]
    assert np.allclose(tail, last_real), "尾部应等于最后一个真实余额(前值填充)"
    # z-score 不该因为插补而炸出极端值
    assert feats["mtss_z"].abs().max() < 10, "z-score 出现异常极值, 插补方式有问题"


def test_level_ffill_has_a_limit():
    """前值填充必须有上限 —— 断更太久就该留 NaN, 而不是无限沿用旧余额。"""
    rng = np.random.default_rng(5)
    bal = rng.normal(1e8, 1e6, N)
    bal[20:] = np.nan          # 断更 40 个交易日, 远超上限
    df = pd.DataFrame({**_base(5), "mtss_balance": bal,
                       **{c: [np.nan] * N for c in ("dde_net", "fund_flow")}})
    feats = calc_fund_features(df)
    assert feats["mtss_1d"].isna().iloc[-1], "断更远超上限后仍应是 NaN"


def test_flow_column_interior_gap_still_zero_filled():
    """流量列的【内部】缺值填 0 仍然是合理的, 修复不该把正常插补也否掉。

    原本这条测试用的是尾部缺失(dde[-5:]), 断言填 0。2026-08-09 该契约被细化:
    尾部缺失和内部缺失的经济含义不同, 见下一条测试。这里改用内部缺口, 保留
    "别把正常插补一起否掉"这个原有的保护意图。
    """
    rng = np.random.default_rng(6)
    dde = rng.normal(0, 1e5, N)
    dde[[10, 11, 25]] = np.nan          # 内部散点缺口
    df = pd.DataFrame({**_base(6), "dde_net": dde,
                       **{c: [np.nan] * N for c in ("mtss_balance", "fund_flow")}})
    feats = calc_fund_features(df)
    assert (feats["dde_net_1d"].iloc[[10, 11, 25]] == 0).all(), \
        "内部散点缺失应当填 0"
    assert feats["dde_net_1d"].notna().all(), "内部插补后不该还有缺失"


def test_flow_column_trailing_gap_stays_nan():
    """流量列的【尾部】缺值必须留 NaN —— 那是"数据还没到", 不是"当天为 0"。

    为什么这个区分很要紧: 资金流表停在 08-04 而 K 线走到 08-07 时, 无差别 fillna(0)
    会把最近 3 天的净流入全部伪造成 0, 而其中就有出信号的那天。此时截面上每只股票
    都"净流入为 0", z-score 退化成同一个假值, 模型却在拿有真值年代学到的分裂点切它。
    留 NaN 才能让 LightGBM 走缺失分支, 也才能让 daily_rebuild 的护栏看见退化。
    """
    rng = np.random.default_rng(6)
    dde = rng.normal(0, 1e5, N)
    dde[-3:] = np.nan                   # 源滞后 3 个交易日, 正是当前的形态
    df = pd.DataFrame({**_base(6), "dde_net": dde,
                       **{c: [np.nan] * N for c in ("mtss_balance", "fund_flow")}})
    feats = calc_fund_features(df)
    assert feats["dde_net_1d"].iloc[-3:].isna().all(), \
        "尾部缺失被伪造成 0 了 —— 这会落在信号日上"
    assert feats["dde_net_z"].iloc[-3:].isna().all(), \
        "z-score 也必须跟着缺失, 否则 0 会被当成'恰好等于21日均值'的真信号"
    assert feats["dde_net_1d"].iloc[:-3].notna().all(), "尾部之前不该受影响"


def test_flow_column_leading_gap_stays_nan():
    """流量列的【前导】缺值也必须留 NaN —— 那是"数据源还没开始", 不是"当天为 0"。

    真实成因: fund_flow 改由 tushare moneyflow 供给后, 该源只有 2019 年起的数据,
    而 consolidated 资金流表从 2015 年就有行。若把 2015~2018 填 0, 窗口A(2019 起
    训练)的 21 日滚动会回看进这段伪造零值, 早期 z-score 全部失真 —— 而这恰好是
    窗口A 的训练起点, 影响的是最需要干净的那一段。
    """
    rng = np.random.default_rng(7)
    ff = rng.normal(0, 1e5, N)
    ff[:12] = np.nan                    # 前 12 天数据源尚未开始
    df = pd.DataFrame({**_base(7), "fund_flow": ff,
                       **{c: [np.nan] * N for c in ("dde_net", "mtss_balance")}})
    feats = calc_fund_features(df)
    assert feats["fund_flow_1d"].iloc[:12].isna().all(), \
        "前导缺失被伪造成 0 了 —— 会污染窗口A 训练起点的滚动统计"
    assert feats["fund_flow_1d"].iloc[12:].notna().all(), "有值区间不该受影响"


def test_ts_lg_features_share_the_same_protections():
    """大单分级特征(第二批)必须走同一条管道, 并受同样的"不伪造"保护。

    它们是通过 col_map 接进去的, 所以理应自动获得 _1d/_z、整列无值不生成、
    首尾之外不填 0 这几项行为。这条测试防止将来有人给它们加特例逻辑时悄悄破功。
    """
    from pipeline.feature_engine import TS_LG_COLS

    rng = np.random.default_rng(8)
    lg = rng.normal(0, 1e6, N)
    lg[-4:] = np.nan
    df = pd.DataFrame({**_base(8),
                       "ts_lg_net": lg,
                       "ts_lg_buy_pct": rng.uniform(10, 60, N),
                       **{c: [np.nan] * N for c in ("dde_net", "mtss_balance",
                                                    "fund_flow")}})
    feats = calc_fund_features(df)
    assert "ts_lg_net_1d" in feats.columns, "大单特征没走通 col_map 管道"
    assert "ts_lg_net_z" in feats.columns
    assert feats["ts_lg_net_1d"].iloc[-4:].isna().all(), \
        "大单特征的尾部缺失同样不得填 0"
    # 整列无值的那两个不该产出
    for c in TS_LG_COLS:
        if c not in df.columns:
            assert f"{c}_1d" not in feats.columns, f"{c} 整列无值却产出了特征"


def test_mtss_prefers_tushare_and_falls_back_to_legacy():
    """mtss_balance 的拼接契约: tushare 优先, 旧值只补 tushare 未覆盖的区间。

    为什么允许在这一列上拼接(而 fund_flow 坚持不拼): 旧源 mtss_balance 与
    tushare margin_detail.rzrqye 在 136 万重叠样本上比值的 1% 与 99% 分位都恰好是
    1.000000 —— 是同一个数, 拼接不产生口径接缝。而 fund_flow 与 net_mf_amount 的
    比值是 10000.0028(近似而非精确), 底层定义有差别, 所以那列宁可留 NaN。

    这条测试同时锁住三件事:
      1. 重叠期取 tushare 的值(而不是旧值)
      2. tushare 未覆盖的早年回落旧值(不是变成 NaN)
      3. outer 合并既不丢行、也不因重复键放大行数
    """
    import pipeline.feature_engine as fe

    dates = pd.bdate_range("2024-01-01", periods=6)
    legacy = pd.DataFrame({
        "date": list(dates), "code": ["000001"] * 6,
        "main_force_net": [1.0] * 6, "main_force_pct": [1.0] * 6,
        "dde_net": [1.0] * 6,
        # 旧源只到第 4 天, 且早年(前2天)是 tushare 没有的区间
        "mtss_balance": [11.0, 12.0, 13.0, 14.0, np.nan, np.nan],
        "fund_flow": [1.0] * 6,
    })

    tmp = Path(__file__).resolve().parent / "_tmp_ff_mtss.parquet"
    legacy.to_parquet(tmp, index=False)

    # tushare 从第 3 天起有值, 且比旧源多覆盖到第 6 天
    ts_mg = pd.DataFrame({
        "code": ["000001"] * 4, "date": list(dates[2:]),
        "mtss_balance": [93.0, 94.0, 95.0, 96.0],
    })

    old_path, old_cache = fe.FUNDFLOW_PATH, fe._fundflow_cache
    old_mf, old_mg = fe._load_tushare_moneyflow, fe._load_tushare_margin
    try:
        fe.FUNDFLOW_PATH = str(tmp)
        fe._fundflow_cache = None
        fe._load_tushare_moneyflow = lambda codes: None   # 本测试只关心 mtss
        fe._load_tushare_margin = lambda codes: ts_mg
        out = fe._load_fundflow()

        got = out.set_index("date")["mtss_balance"]
        assert len(out) == 6, "outer 合并把行数改变了: %d" % len(out)
        # 重叠期(第3~4天)必须是 tushare 的 93/94, 不是旧源的 13/14
        assert got[dates[2]] == 93.0 and got[dates[3]] == 94.0, \
            "重叠期没有优先用 tushare"
        # tushare 未覆盖的早年回落旧值
        assert got[dates[0]] == 11.0 and got[dates[1]] == 12.0, \
            "早年没有回落到旧值"
        # tushare 独有的尾部日期要补进来 —— 这正是"救回断更"的目的
        assert got[dates[4]] == 95.0 and got[dates[5]] == 96.0, \
            "tushare 独有的尾部日期没补上, 断更没救回来"
    finally:
        fe.FUNDFLOW_PATH, fe._fundflow_cache = old_path, old_cache
        fe._load_tushare_moneyflow, fe._load_tushare_margin = old_mf, old_mg
        tmp.unlink(missing_ok=True)


def test_margin_all_nan_source_produces_no_feature():
    """融资融券侧同样不得伪造 —— 它和资金流是同一个 fillna(0) 模式。"""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=N),
        "rzye": rng.normal(1e8, 1e6, N),
        "rzmre": rng.normal(1e7, 1e5, N),
        "rzche": rng.normal(1e7, 1e5, N),
        "rqye": [np.nan] * N,
    })
    feats = calc_margin_features(df)
    if feats is None:
        _skip("可用源列不足 3 个, calc_margin_features 按设计返回 None")
    assert "short_bal_1d" not in feats.columns, "rqye 整列为空, 不该产出 short_bal_1d"
    assert "marg_bal_1d" in feats.columns
    assert feats["marg_bal_1d"].nunique() > 1


def test_no_selected_feature_is_constant_in_training_matrix():
    """真实训练矩阵体检: 入选特征里不允许出现恒常量。

    这条是事故的最终判据 —— 上面几条锁的是函数行为, 这条锁的是落盘结果。
    只在能读到训练集时运行 (数据在服务器上, 本地跳过)。
    """
    import json

    root = Path(__file__).resolve().parents[1]
    train = root / "data" / "processed" / "training_data_pit_v24.parquet"
    feat_json = next((root / "data" / "processed").glob("wf_daily_REGRESS_CHK_*.json"), None)
    if not train.exists() or feat_json is None:
        _skip("本地无训练集/特征集, 该体检只在有数据的机器上跑")

    import pyarrow.parquet as pq

    sel = json.loads(feat_json.read_text(encoding="utf-8"))["selected_features"]
    have = set(pq.ParquetFile(train).schema_arrow.names)
    cols = [c for c in sel if c in have]
    df = pd.read_parquet(train, columns=cols)
    const = [c for c in cols if df[c].nunique(dropna=True) <= 1]
    assert not const, (
        f"入选特征中有 {len(const)} 个恒为常量, 数据源可能已静默断更: {const}")


def _main():
    """无 pytest 环境下直接 python tests/xxx.py 就能跑。"""
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for n in names:
        try:
            globals()[n]()
            print(f"  PASS  {n}")
        except _Skip as e:
            print(f"  SKIP  {n}: {e}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL  {n}: {str(e)[:160]}")
    print(f"\n{len(names) - len(failed)}/{len(names)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
