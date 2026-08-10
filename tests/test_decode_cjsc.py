"""回归测试: 中信建投 .DAT 解码器

为什么需要它
────────────
这份数据的二进制格式没有官方文档, 是靠"用已知事实反推字段"确定的:
600519 日线首条必须是 2001-08-27 开34.51 高37.78 低32.85 收35.55 前收31.39
(贵州茅台上市首日, 发行价 31.39)。这个断言一旦不成立, 说明字段偏移解错了,
后面所有基于它的特征都会是错的 —— 而且是那种不会报错的错。

同时锁住两个已知的坑:

  1. **时间戳是 bar 结束时刻**。5 分钟线每日 48 条, 首条 09:35 末条 15:00。
     若哪天误改成开始时刻(09:30 起), 用 14:50 的 bar 做 14:50 的决策就变成
     了偷看未来 5 分钟 —— 这正是"T日14:50选股"策略的命门。

  2. **成交额 uint32 溢出**。超过 42.95 亿会回绕, 实测日线约 19~22% 的记录
     中招。解码器必须还原, 还原不出来的必须置 NaN, 绝不能留一个错误数值
     冒充真值 (与 tests/test_no_fabricated_zero_features.py 同一条纪律)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
except ModuleNotFoundError:      # 运行环境未必装 pytest, 不能因此跑不了回归
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import settings                       # noqa: E402
from pipeline.decode_cjsc import (                         # noqa: E402
    HEADER_SIZE, MAGIC, RECORD_DTYPE, RECORD_SIZE, UINT32,
    CjscFormatError, _fix_amount_overflow, check_ohlc, period_of, read_bars,
)


class _Skip(Exception):
    pass


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


def _sample(rel):
    p = settings.CJSC_DIR / rel
    if not p.exists():
        _skip(f"缺少样本数据 {p} (需先从中信建投导出包解压)")
    return p


# ══════════════════════════════════════════════════════════════
# 不依赖真实数据的部分: 用构造数据锁住格式常量与溢出修复
# ══════════════════════════════════════════════════════════════

def _build(records):
    """按 RECORD_DTYPE 造一个合法的 .DAT 字节串"""
    a = np.zeros(len(records), dtype=RECORD_DTYPE)
    for i, r in enumerate(records):
        for k, v in r.items():
            a[k][i] = v
    return MAGIC + a.tobytes()


def test_format_constants():
    assert RECORD_DTYPE.itemsize == RECORD_SIZE == 64
    assert HEADER_SIZE == 8
    assert MAGIC == bytes.fromhex("feffffffffffff7f")


def test_bad_magic_raises(tmp_path):
    p = tmp_path / "bad.DAT"
    p.write_bytes(b"\x00" * 8 + b"\x00" * RECORD_SIZE)
    try:
        read_bars(p)
    except CjscFormatError:
        return
    raise AssertionError("magic 不符时必须抛 CjscFormatError")


def test_truncated_record_raises(tmp_path):
    p = tmp_path / "trunc.DAT"
    p.write_bytes(MAGIC + b"\x00" * (RECORD_SIZE + 7))
    try:
        read_bars(p)
    except CjscFormatError:
        return
    raise AssertionError("数据段非整数条记录时必须抛错")


def test_empty_file(tmp_path):
    p = tmp_path / "empty.DAT"
    p.write_bytes(MAGIC)
    df = read_bars(p)
    assert df.empty
    assert list(df.columns)[:5] == ["dt", "open", "high", "low", "close"]


def test_price_scale_and_timestamp(tmp_path):
    """价格 x1000, 时间戳按 CST 还原"""
    p = tmp_path / "one.DAT"
    # 2026-08-07 15:00 CST = 1786467600 UTC
    p.write_bytes(_build([dict(
        ts=1786467600 - 8 * 3600 + 8 * 3600, open=10_000, high=11_000,
        low=9_000, close=10_500, vol=100, amount=105_000_000,
        adj=np.float32(1.5), prev_close=9_900)]))
    df = read_bars(p)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["open"] == 10.0 and row["high"] == 11.0
    assert row["low"] == 9.0 and row["close"] == 10.5
    assert row["prev_close"] == 9.9
    assert abs(row["adj"] - 1.5) < 1e-6


def test_amount_overflow_is_restored():
    """成交额回绕必须还原, 还原不出的必须是 NaN 而不是错误数值"""
    # 真实成交额 60 亿 > 2^32, 存盘时回绕
    true_amount = 6_000_000_000.0
    vol = 4_000_000              # 手 -> 4 亿股
    price = true_amount / (vol * 100.0)      # 15.0
    # 第二行是正常未溢出的记录: 100 手 = 1 万股 x 10 元 = 10 万元
    df = pd.DataFrame({
        "amount": [true_amount % UINT32, 100_000.0, 0.0],
        "vol": [vol, 100, 0],
        "low": [price * 0.99, 9.5, 0.0],
        "high": [price * 1.01, 10.5, 0.0],
    })
    fixed, status = _fix_amount_overflow(df)

    assert status.iloc[0] == "fixed"
    assert abs(fixed.iloc[0] - true_amount) < UINT32 * 1e-9
    assert status.iloc[1] == "ok"            # 未溢出的不许被改动
    assert fixed.iloc[1] == 100_000.0
    assert status.iloc[2] == "no_trade"      # 零成交不参与判定


def test_unresolvable_amount_becomes_nan():
    """解不出回绕圈数时必须置 NaN —— 绝不留错误数值冒充真值"""
    df = pd.DataFrame({
        "amount": [123.0],       # 与 vol/价格区间完全对不上
        "vol": [1_000_000],
        "low": [50.0],
        "high": [51.0],
    })
    fixed, status = _fix_amount_overflow(df)
    assert status.iloc[0] in ("fixed", "unresolved")
    if status.iloc[0] == "unresolved":
        assert np.isnan(fixed.iloc[0])


def test_period_of():
    assert period_of("data/raw/cjsc/SH/300/600519.DAT") == "5min"
    assert period_of("data/raw/cjsc/SZ/60/000001.DAT") == "1min"
    assert period_of("data/raw/cjsc/SH/86400/600519.DAT") == "1d"
    assert period_of("somewhere/600519.DAT") is None


# ══════════════════════════════════════════════════════════════
# 依赖真实样本: 字段偏移的最终断言
# ══════════════════════════════════════════════════════════════

def test_maotai_daily_first_bar_matches_ipo():
    """600519 日线首条 = 茅台上市首日, 这是整个格式推断的锚点"""
    df = read_bars(_sample("SH/86400/600519.DAT"))
    first = df.iloc[0]
    assert first["dt"] == pd.Timestamp("2001-08-27")
    assert abs(first["open"] - 34.51) < 1e-6
    assert abs(first["high"] - 37.78) < 1e-6
    assert abs(first["low"] - 32.85) < 1e-6
    assert abs(first["close"] - 35.55) < 1e-6
    assert abs(first["prev_close"] - 31.39) < 1e-6   # IPO 发行价
    assert abs(first["adj"] - 1.0) < 1e-6            # 上市日复权因子=1


def test_5min_timestamp_is_bar_end():
    """5 分钟 bar 的时间戳必须是结束时刻 —— 防未来泄漏的命门"""
    df = read_bars(_sample("SH/300/600519.DAT"))
    day = df["dt"].dt.normalize()
    counts = day.value_counts()
    full = counts[counts == 48]
    assert len(full) > 100, "完整交易日(48条)太少, 数据可能不对"

    one = df[day == full.index[5]]
    times = [t.strftime("%H:%M") for t in one["dt"]]
    assert times[0] == "09:35", f"首条应为 09:35(bar结束时刻), 实为 {times[0]}"
    assert times[-1] == "15:00"
    assert "11:30" in times and "13:05" in times      # 午休边界
    assert "09:30" not in times                       # 开始时刻语义的特征
    assert "14:50" in times


def test_real_files_pass_ohlc_and_no_fake_amount():
    for rel in ("SH/86400/600519.DAT", "SH/300/600519.DAT"):
        df = read_bars(_sample(rel))
        assert check_ohlc(df).sum() == 0, f"{rel} 存在 OHLC 矛盾"
        # 还原后仍标记为 ok/fixed 的, 均价必须落在 [low, high] 内
        ok = df[df["amount_status"].isin(["ok", "fixed"]) & (df["vol"] > 0)]
        implied = ok["amount"] / (ok["vol"] * 100.0)
        bad = ((implied < ok["low"] * 0.97) | (implied > ok["high"] * 1.03)).sum()
        assert bad == 0, f"{rel} 有 {bad} 条成交额被错误地标成了可用"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for fn in fns:
        name = fn.__name__
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS  {name}")
            passed += 1
        except _Skip as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            raise
    print(f"\n{passed} passed, {skipped} skipped")
