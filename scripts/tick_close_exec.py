# -*- coding: utf-8 -*-
"""从 Wind L2 逐笔快照标定 exec_mode=t1close 的真实执行成本

为什么需要这个
──────────────
回测长期用 --slippage 0.002 (20bp/边), 这个数字是拍的, 而它的量级(往返 0.4%,
一年 48 次调仓 = -19%/年)比选股能力的影响还大。滑点假设错了, 所有策略排序都不可信。

现实的执行是 14:50-15:00 尾盘, 这个窗口里有两段性质完全不同的机制:
  14:50-14:57 连续竞价     —— 市价单要穿价, 且成交价 != 收盘价, 有时点偏差
  14:57-15:00 收盘集合竞价 —— 沪深都以【单一清算价】撮合, 而那个价就是收盘价,
                              所以在这一段成交, "按收盘价成交"这个假设精确成立
换句话说 t1close 的滑点是 0 还是 20bp, 取决于订单落在哪一段。本脚本把这件事量清楚。

产出 data/processed/tick_exec/<date>.parquet, 每行一个股票日:
  auc_share    集合竞价成交额占全日比例  -> 容量够不够吃下我们的单
  auc_dev      集合竞价 VWAP / 收盘价 - 1 -> 应恒为 0, 验证撮合机制
  cont_dev     14:50-14:57 VWAP / 收盘价 - 1 -> 不参与集合竞价的代价
  win_dev      14:50-15:00 整窗 VWAP / 收盘价 - 1
  half_spread  窗口内 (卖1-买1)/2/中价 -> 穿价成本
  ask_dev/bid_dev  按卖1买入 / 按买1卖出 相对收盘价的偏离
  one_side_ratio   窗口内单边报价(涨跌停, 根本成不了交)的快照占比

实测结论 (2026-08-14, 1851 个股票日):
  auc_dev 中位 0.000000 (95%分位 3e-7) —— 集合竞价成交价恰为收盘价
  auc_share 中位 1.06% (5%分位 0.55%) —— 我们 1~3万/笔占其零点几个百分点
  half_spread 中位 2.7bp;  cont_dev 中位 -5.2bp (5%~95% = -28bp~+16bp)
  -> 真实单边成本 0bp(集合竞价) ~ 2.7bp(连续竞价穿价), 不是 20bp
  与 35 笔实盘确认单的实测(中位 0bp)独立吻合

用法
────
    python scripts/tick_close_exec.py 20220901 20260807        # 全窗口
    NW=16 python scripts/tick_close_exec.py 20260801 20260807  # 指定并发

前置: pip install py7zr; 逐笔数据在 TICK 指向的目录 (单日 7z 2~6GB,
解压后 56GB/日, 所以必须按需 targets 抽取, 绝不能整包解)。
"""
import os
import sys
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import py7zr
except ImportError:
    sys.exit("需要 py7zr: pip install py7zr")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TICK = Path(os.environ.get(
    "TICK_DIR", "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
OUT = ROOT / "data/processed/tick_exec"
PDIV = 10000.0                      # 逐笔里价格单位是 1/10000 元
T_1450, T_1457, T_1500 = 145000000, 145700000, 150000000   # 时间格式 HHMMSSmmm

USE = ["时间", "成交价", "当日累计成交量", "当日成交额", "申卖价1", "申买价1",
       "申卖量1", "申买量1", "前收盘", "开盘价", "最高价", "最低价"]


def resolve(day: str):
    """2017-2021 用 <年>/<01..12>/, 2022+ 用 <年>/<年月>/ —— 两种命名都要认"""
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def one_stock(path: Path):
    try:
        df = pd.read_csv(path, encoding="gbk", usecols=lambda c: c in USE,
                         on_bad_lines="skip")
    except Exception:
        return None
    if df.empty or "时间" not in df.columns:
        return None
    df = df.sort_values("时间")
    cum_v = df["当日累计成交量"].astype(float).values
    cum_a = df["当日成交额"].astype(float).values
    t = df["时间"].astype("int64").values
    px = df["成交价"].astype(float).values / PDIV

    nz = np.nonzero(px > 0)[0]
    if len(nz) == 0:
        return None
    close = px[nz[-1]]                      # 收盘价 = 最后一笔有成交的快照价
    tot_v, tot_a = cum_v[nz[-1]], cum_a[nz[-1]]
    if tot_v <= 0:
        return None

    def cum_at(ts):
        i = np.searchsorted(t, ts, side="right") - 1
        return (0.0, 0.0) if i < 0 else (float(cum_v[i]), float(cum_a[i]))

    v1450, a1450 = cum_at(T_1450)
    v1457, a1457 = cum_at(T_1457)

    def vwap(dv, da):
        return (da / dv) if dv > 0 else np.nan

    cont_vwap = vwap(v1457 - v1450, a1457 - a1450)
    auc_vwap = vwap(tot_v - v1457, tot_a - a1457)
    win_vwap = vwap(tot_v - v1450, tot_a - a1450)

    m = (t >= T_1450) & (t < T_1500)
    ask = df["申卖价1"].astype(float).values[m] / PDIV
    bid = df["申买价1"].astype(float).values[m] / PDIV
    ok = (ask > 0) & (bid > 0)
    if ok.sum() > 0:
        mid = (ask[ok] + bid[ok]) / 2
        half_sp = float(np.median((ask[ok] - bid[ok]) / 2 / mid))
        ask_dev = float(np.median(ask[ok] / close - 1))
        bid_dev = float(np.median(1 - bid[ok] / close))
    else:
        half_sp = ask_dev = bid_dev = np.nan

    def dev(v):
        return (v / close - 1) if v == v else np.nan

    return dict(
        close=close, prev_close=float(df["前收盘"].iloc[-1]) / PDIV,
        day_amt=tot_a, day_vol=tot_v,
        auc_amt=tot_a - a1457,
        auc_share=(tot_a - a1457) / tot_a if tot_a > 0 else np.nan,
        win_amt=tot_a - a1450,
        auc_dev=dev(auc_vwap), cont_dev=dev(cont_vwap), win_dev=dev(win_vwap),
        half_spread=half_sp, ask_dev=ask_dev, bid_dev=bid_dev,
        one_side_ratio=float(1 - ok.mean()) if m.sum() else np.nan,
    )


def run_day(day: str, codes):
    outf = OUT / f"{day}.parquet"
    if outf.exists():
        return f"{day} skip"
    p = resolve(day)
    if p is None:
        return f"{day} 无包"
    with py7zr.SevenZipFile(p, mode="r") as z:
        names = z.getnames()
    inner = names[0].split("/")[0]
    have = {n.split("/")[1] for n in names if n.count("/") == 2}
    want = [c for c in codes if c in have]
    if not want:
        return f"{day} 无匹配"
    tmp = tempfile.mkdtemp(prefix=f"ex{day}_", dir="/tmp")
    try:
        with py7zr.SevenZipFile(p, mode="r") as z:
            z.extract(path=tmp, targets=[f"{inner}/{c}/行情.csv" for c in want])
        rows = []
        for c in want:
            f = Path(tmp) / inner / c / "行情.csv"
            if not f.exists():
                continue
            r = one_stock(f)
            if r:
                r["code"], r["date"] = c[:6], day
                rows.append(r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not rows:
        return f"{day} 空"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(outf, index=False)
    return f"{day} ok {len(rows)}"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.split("用法")[1].split("前置")[0])
    start, end = sys.argv[1], sys.argv[2]

    u = pd.read_parquet(ROOT / "data/universe/universe_pit_2019.parquet")
    c6 = sorted(u["code"].astype(str).str.zfill(6).unique())
    codes = [f"{c}.{'SH' if c[0] == '6' else 'SZ'}" for c in c6]

    days = sorted(p.stem for p in TICK.glob("*/*/*.7z") if start <= p.stem <= end)
    if not days:
        sys.exit(f"{TICK} 下没有 {start}~{end} 的数据")
    print(f"股票 {len(codes)} 只  交易日 {len(days)} 天  {days[0]}~{days[-1]}", flush=True)

    import time
    from functools import partial
    from multiprocessing import Pool

    nw = int(os.environ.get("NW", "16"))
    t0 = time.time()
    with Pool(nw) as pool:
        for i, msg in enumerate(pool.imap_unordered(
                partial(run_day, codes=codes), days, chunksize=1), 1):
            if i % 25 == 0 or "无" in msg or "空" in msg:
                el = time.time() - t0
                print(f"[{i}/{len(days)}] {msg}  {el/60:.1f}min "
                      f"剩余~{el/i*(len(days)-i)/60:.0f}min", flush=True)
    print(f"完成, 输出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
