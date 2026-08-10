"""中信建投客户端本地行情文件(.DAT)解码器

数据来源: 中信建投客户端导出目录 中信建投data-260808.rar (8.1GB, RAR5, 非 solid)
  SH/60/     SZ/60/       1 分钟线   (目录名 = 周期秒数)
  SH/300/    SZ/300/      5 分钟线
  SH/86400/  SZ/86400/    日线
  Finance/{SH,SZ,BJ}/86400/{code}_700{1..8}.DAT   财务(另一种格式, 本模块不处理)
  Sector/    板块成分(纯文本 CSV, 本模块不处理)
  Industry/IndustryData.txt
  DividData/*.ldb

二进制格式
──────────
文件头 8 字节: FE FF FF FF FF FF FF 7F  (magic)
之后定长 64 字节/条, 小端:

    +0x00 uint32   unix 时间戳(秒, UTC), 转 CST 后即 bar 时刻
    +0x04 uint32   开盘 x1000
    +0x08 uint32   最高 x1000
    +0x0c uint32   最低 x1000
    +0x10 uint32   收盘 x1000
    +0x14 uint32   (恒 0)
    +0x18 uint32   成交量(手)
    +0x1c uint32   (常量, 非真实笔数 —— 日线恒 491, 5分钟恒 399, 1分钟恒 504)
    +0x20 uint32   成交额(元)   ← 会溢出, 见下
    +0x24 uint32   (恒 0)
    +0x28 uint32   标志位
    +0x2c float32  (恒 1.0)
    +0x30 float32  复权因子(后复权乘数, 上市日=1.0)
    +0x34 uint32   前收 x1000
    +0x38 uint32   (恒 0)
    +0x3c uint32   哨兵

格式验证 (600519 日线首条):
    2001-08-27  开 34.51  高 37.78  低 32.85  收 35.55  前收 31.39
    = 贵州茅台上市首日, 发行价 31.39 —— 完全吻合

两个必须知道的坑
────────────────
1. **时间戳是 bar 的结束时刻**, 不是开始时刻。
   每个交易日 48 条 5 分钟 bar, 首条 09:35 (覆盖 09:30~09:35), 末条 15:00。
   所以 14:50 那条在 14:50 时点【已经完成】, 用它做 14:50 的决策不构成未来泄漏。
   若误当成开始时刻, 就会用到未来 5 分钟的数据。

2. **成交额是 uint32, 超过 42.95 亿(2^32)会回绕**。
   实测 600519 日线 18.97% 的记录溢出, 601318 日线 22.38%。
   5 分钟几乎不溢出(600519/601318 为 0.00%, 300750 为 0.02%)。
   本模块默认按 "还原后的均价必须落在 [low, high] 区间内" 反解回绕圈数 k,
   解不出来的置 NaN —— 绝不留一个错误的数值冒充真值。
   日线成交额建议直接用 tushare(我们已有全市场 8.77M 行), 不要用这份。

用法
────
    from pipeline.decode_cjsc import read_bars
    df = read_bars("data/raw/cjsc/SH/300/600519.DAT")

    python pipeline/decode_cjsc.py 路径.DAT [更多路径...]
"""
from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MAGIC = bytes.fromhex("feffffffffffff7f")
HEADER_SIZE = 8
RECORD_SIZE = 64
UINT32 = 1 << 32
CST = timezone(timedelta(hours=8))

# 价格与成交额的存储单位
PRICE_SCALE = 1000.0
SHARES_PER_LOT = 100.0

RECORD_DTYPE = np.dtype([
    ("ts", "<u4"),
    ("open", "<u4"),
    ("high", "<u4"),
    ("low", "<u4"),
    ("close", "<u4"),
    ("_zero1", "<u4"),
    ("vol", "<u4"),
    ("_const", "<u4"),
    ("amount", "<u4"),
    ("_zero2", "<u4"),
    ("flag", "<u4"),
    ("_one", "<f4"),
    ("adj", "<f4"),
    ("prev_close", "<u4"),
    ("_zero3", "<u4"),
    ("_sentinel", "<u4"),
])
assert RECORD_DTYPE.itemsize == RECORD_SIZE

# 目录名 -> 周期
PERIOD_BY_DIR = {"60": "1min", "300": "5min", "86400": "1d"}
BARS_PER_DAY = {"1min": 240, "5min": 48, "1d": 1}


class CjscFormatError(ValueError):
    """文件头不是预期的 magic, 或长度不是整数条记录"""


def _fix_amount_overflow(df: pd.DataFrame, tol: float = 0.03):
    """还原 uint32 回绕的成交额

    真值 = amount + k * 2^32。约束是还原后的均价必须落在当根 bar 的
    [low, high] 内(留 tol 的容差, 因为成交额含手续费口径差异且价格有舍入)。

    对每条记录解出满足约束的最小非负 k:
        k_lo = ceil ((low  * (1-tol) * vol * 100 - amount) / 2^32)
        k_hi = floor((high * (1+tol) * vol * 100 - amount) / 2^32)
    k_lo <= k_hi 时取 max(0, k_lo); 无解则置 NaN。

    返回 (修正后的成交额 Series[float], 标记 Series[str])
    标记取值: ok / fixed / unresolved / no_trade
    """
    amount = df["amount"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)

    shares = vol * SHARES_PER_LOT
    status = np.full(len(df), "ok", dtype=object)
    fixed = amount.copy()

    traded = shares > 0
    status[~traded] = "no_trade"

    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(traded, amount / np.where(traded, shares, 1.0), np.nan)
    in_range = traded & (implied >= low * (1 - tol)) & (implied <= high * (1 + tol))

    need = traded & ~in_range
    if need.any():
        lo_bound = low[need] * (1 - tol) * shares[need]
        hi_bound = high[need] * (1 + tol) * shares[need]
        k_lo = np.ceil((lo_bound - amount[need]) / UINT32)
        k_hi = np.floor((hi_bound - amount[need]) / UINT32)
        k = np.maximum(k_lo, 0.0)
        solvable = (k_lo <= k_hi) & (k <= k_hi)

        idx = np.flatnonzero(need)
        ok_idx = idx[solvable]
        bad_idx = idx[~solvable]
        fixed[ok_idx] = amount[ok_idx] + k[solvable] * UINT32
        status[ok_idx] = "fixed"
        fixed[bad_idx] = np.nan
        status[bad_idx] = "unresolved"

    return pd.Series(fixed, index=df.index), pd.Series(status, index=df.index)


def read_bars(path, fix_amount: bool = True, strict: bool = True) -> pd.DataFrame:
    """读取一个 .DAT 行情文件

    Args:
        path: .DAT 文件路径
        fix_amount: 是否还原溢出的成交额。False 则原样返回 uint32 值
        strict: magic 不符时抛错; False 则仅在长度可整除时尽力解析

    Returns:
        DataFrame[dt, open, high, low, close, vol, amount, adj, prev_close,
                  amount_status]  —— dt 为 CST naive 时间
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        return _empty_frame()

    magic = raw[:HEADER_SIZE]
    if magic != MAGIC:
        if strict:
            raise CjscFormatError(f"{path}: 文件头 {magic.hex()} != {MAGIC.hex()}")

    body = raw[HEADER_SIZE:]
    n, rem = divmod(len(body), RECORD_SIZE)
    if rem and strict:
        raise CjscFormatError(
            f"{path}: 数据段 {len(body)} 字节不是 {RECORD_SIZE} 的整数倍(余 {rem})")
    if n == 0:
        return _empty_frame()

    a = np.frombuffer(body[: n * RECORD_SIZE], dtype=RECORD_DTYPE)
    df = pd.DataFrame({
        "dt": pd.to_datetime(a["ts"], unit="s", utc=True)
              .tz_convert(CST).tz_localize(None),
        "open": a["open"] / PRICE_SCALE,
        "high": a["high"] / PRICE_SCALE,
        "low": a["low"] / PRICE_SCALE,
        "close": a["close"] / PRICE_SCALE,
        "vol": a["vol"].astype("int64"),
        "amount": a["amount"].astype("float64"),
        "adj": a["adj"].astype("float64"),
        "prev_close": a["prev_close"] / PRICE_SCALE,
    })

    if fix_amount:
        df["amount"], df["amount_status"] = _fix_amount_overflow(df)
    else:
        df["amount_status"] = "raw"
    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "dt": pd.Series(dtype="datetime64[ns]"),
        "open": pd.Series(dtype="float64"),
        "high": pd.Series(dtype="float64"),
        "low": pd.Series(dtype="float64"),
        "close": pd.Series(dtype="float64"),
        "vol": pd.Series(dtype="int64"),
        "amount": pd.Series(dtype="float64"),
        "adj": pd.Series(dtype="float64"),
        "prev_close": pd.Series(dtype="float64"),
        "amount_status": pd.Series(dtype="object"),
    })


def check_ohlc(df: pd.DataFrame) -> pd.Series:
    """OHLC 内部一致性: high >= max(o,c,l), low <= min(o,c,h)。返回违规布尔序列"""
    if df.empty:
        return pd.Series(dtype=bool)
    return (
        (df["high"] < df["low"])
        | (df["close"] > df["high"]) | (df["close"] < df["low"])
        | (df["open"] > df["high"]) | (df["open"] < df["low"])
    )


def period_of(path) -> str | None:
    """从路径推断周期: .../SH/300/600519.DAT -> '5min'"""
    parts = Path(path).parts
    for p in reversed(parts[:-1]):
        if p in PERIOD_BY_DIR:
            return PERIOD_BY_DIR[p]
    return None


def _main(argv):
    if not argv:
        print(__doc__)
        return 1
    for path in argv:
        df = read_bars(path)
        period = period_of(path) or "?"
        print("=" * 78)
        print(f"{path}   周期={period}   记录数={len(df)}")
        if df.empty:
            continue
        days = df["dt"].dt.normalize().nunique()
        print(f"区间: {df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]}  ({days} 个交易日)")
        print(f"复权因子: 首 {df['adj'].iloc[0]:.4f}  末 {df['adj'].iloc[-1]:.4f}")
        vc = df["amount_status"].value_counts()
        print("成交额: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
        bad = check_ohlc(df).sum()
        print(f"OHLC 矛盾: {bad} 条")
        print(df.head(3).to_string(index=False))
        print("...")
        print(df.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
