# -*- coding: utf-8 -*-
"""日内择时·阶段 1: 从逐笔仓库建 PIT 池的 1 分钟面板 (跑在 eez040, 逐笔仓库所在机)

要回答的问题是"卖出日盘中什么时候卖" —— 决策只能用决策时刻之前的信息, 所以要一份
带时间戳的日内面板。日线 OHLC 只能算出"满分"(|开盘-收盘| 均值 172bp), 算不出可预测性。

来源 (同一份 7z, 与 t1a/tick_micro 抽取器同仓):
  逐笔成交.csv  -> 每分钟 OHLC / 成交量额 / 笔数 / 主动买卖额(BS 标志)
  行情.csv      -> 每分钟 买一卖一 价差(bp)与挂单不平衡(均值)、分钟末买一卖一价、前收盘
两类字段分开存, 因为 QMT 实盘 L1 只有后者(+OHLCV); BS 标志是 L2 才有, 研究里要单独记账
它贡献了多少 ρ, 上线时好取舍。

分钟约定 (bar 用起始分钟 HHMM 标记, 覆盖 [HHMM:00, HHMM+1:00)):
  0925          开盘集合竞价 (9:15-9:30 的一切都归这里)
  0930..1129    上午 120 根;  11:30:00 整的成交归 1129
  1300..1456    下午 117 根
  1457          收盘集合竞价 14:57-15:00 (沪深同制), 它的 close 就是收盘价
每码每日最多 239 行。SZ 撤单(成交代码=='C')先剔; 价或量 <=0 的行丢。
qc_vol_ratio = 逐笔成交量合计 / 快照最后一笔累计成交量, 应 ≈1, 偏离说明两文件不同批次。

产出 data/processed/min1/<YYYYMMDD>.parquet, 列:
  date code minute open high low close vol amt n_trd buy_amt sell_amt
  spread_bp qimb bid1 ask1 prev_close qc_vol_ratio

用法
────
    python scripts/build_minute_bars.py 20260904 20260904        # 单日
    NW=8 python scripts/build_minute_bars.py 20220901 20260907   # 补历史 (共享机器, NW 别吃满)
幂等: 已有日文件直接跳过。
"""
import os
import shutil
import sys
import tempfile
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import py7zr
except ImportError:
    py7zr = None

ROOT = Path(__file__).resolve().parents[1]
TICK = Path(os.environ.get(
    "TICK_DIR", "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
OUT = ROOT / Path(os.environ.get("MIN1_OUT", "data/processed/min1"))

F_TRD, F_QUO = "逐笔成交.csv", "行情.csv"
U_TRD = ("时间", "成交代码", "BS标志", "成交价格", "成交数量")
U_QUO = ("时间", "申卖价1", "申买价1", "申卖量1", "申买量1", "前收盘", "当日累计成交量")
PDIV = 10000.0
CHUNK = int(os.environ.get("CHUNK", "50"))

F32 = ("open", "high", "low", "close", "spread_bp", "qimb", "bid1", "ask1", "prev_close", "qc_vol_ratio")


def resolve(day: str):
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def minute_of(t: np.ndarray) -> np.ndarray:
    """HHMMSSmmm -> bar 标签 HHMM, 含开/收盘集合竞价与 11:30 的归并"""
    hhmm = t // 100_000
    hhmm = np.where(hhmm < 930, 925, hhmm)
    hhmm = np.where((hhmm >= 1130) & (hhmm < 1300), 1129, hhmm)
    hhmm = np.where(hhmm >= 1457, 1457, hhmm)
    return hhmm.astype("int16")


def _read(path: Path, use):
    d = pd.read_csv(path, encoding="gbk", engine="c", on_bad_lines="skip", low_memory=False,
                    usecols=lambda c: c.strip() in use)
    d.columns = [c.strip() for c in d.columns]
    return d


def bars_for_code(f_trd: Path, f_quo, is_sz: bool):
    d = _read(f_trd, U_TRD)
    if "成交价格" not in d.columns or "时间" not in d.columns:
        return None
    if is_sz and "成交代码" in d.columns:
        d = d[d["成交代码"].astype(str).str.strip() != "C"]
    t = pd.to_numeric(d["时间"], errors="coerce")
    px = pd.to_numeric(d["成交价格"], errors="coerce") / PDIV
    qty = pd.to_numeric(d["成交数量"], errors="coerce")
    ok = t.notna() & (px > 0) & (qty > 0)
    if not ok.any():
        return None
    bs = d["BS标志"].astype(str).str.strip() if "BS标志" in d.columns else pd.Series("", index=d.index)
    d = pd.DataFrame({"t": t[ok].astype("int64").values, "px": px[ok].values,
                      "qty": qty[ok].values, "bs": bs[ok].values})
    d = d.sort_values("t", kind="stable")
    d["m"] = minute_of(d["t"].values)
    d["amt"] = d["px"] * d["qty"]
    g = d.groupby("m", sort=True)
    bars = pd.DataFrame({
        "open": g["px"].first(), "high": g["px"].max(), "low": g["px"].min(), "close": g["px"].last(),
        "vol": g["qty"].sum(), "amt": g["amt"].sum(), "n_trd": g.size().astype("int32"),
        "buy_amt": d.loc[d["bs"] == "B"].groupby("m")["amt"].sum(),
        "sell_amt": d.loc[d["bs"] == "S"].groupby("m")["amt"].sum(),
    })
    bars[["buy_amt", "sell_amt"]] = bars[["buy_amt", "sell_amt"]].fillna(0.0)

    prev_close, qc = np.nan, np.nan
    if f_quo is not None and f_quo.exists():
        try:
            q = _read(f_quo, U_QUO)
            tq = pd.to_numeric(q["时间"], errors="coerce")
            ask = pd.to_numeric(q["申卖价1"], errors="coerce") / PDIV
            bid = pd.to_numeric(q["申买价1"], errors="coerce") / PDIV
            asz = pd.to_numeric(q["申卖量1"], errors="coerce")
            bsz = pd.to_numeric(q["申买量1"], errors="coerce")
            v = tq.notna() & (ask > 0) & (bid > 0) & (ask >= bid)
            if v.any():
                qq = pd.DataFrame({
                    "m": minute_of(tq[v].astype("int64").values),
                    "spread_bp": ((ask - bid) / ((ask + bid) / 2) * 1e4)[v].values,
                    "qimb": ((bsz - asz) / (bsz + asz).replace(0, np.nan))[v].values,
                    "bid": bid[v].values, "ask": ask[v].values})
                gq = qq.groupby("m", sort=True).agg(spread_bp=("spread_bp", "mean"), qimb=("qimb", "mean"),
                                                    bid1=("bid", "last"), ask1=("ask", "last"))
                bars = bars.join(gq, how="left")
            if "前收盘" in q.columns:
                pc = pd.to_numeric(q["前收盘"], errors="coerce")
                pc = pc[pc > 0]
                if len(pc):
                    prev_close = float(pc.iloc[-1]) / PDIV
            if "当日累计成交量" in q.columns:
                cv = pd.to_numeric(q["当日累计成交量"], errors="coerce")
                cv = cv[cv > 0]
                if len(cv):
                    qc = float(bars["vol"].sum() / cv.iloc[-1])
        except Exception:
            pass
    for c in ("spread_bp", "qimb", "bid1", "ask1"):
        if c not in bars.columns:
            bars[c] = np.nan
    bars["prev_close"] = prev_close
    bars["qc_vol_ratio"] = qc
    bars.index.name = "minute"
    bars = bars.reset_index()
    for c in F32:
        bars[c] = bars[c].astype("float32")
    return bars


def run_day(day: str, codes):
    if py7zr is None:
        raise RuntimeError("抽取需要 py7zr: pip install py7zr")
    f_out = OUT / f"{day}.parquet"
    if f_out.exists():
        return f"{day} skip"
    zf = resolve(day)
    if zf is None:
        return f"{day} 无7z包"
    t0 = time.time()
    frames = []
    try:
        with py7zr.SevenZipFile(zf) as z:
            names = set(z.getnames())
        targets = {}
        for c in codes:
            for pre in (f"{day}/{c}", c):
                if f"{pre}/{F_TRD}" in names:
                    ps = [f"{pre}/{F_TRD}"]
                    if f"{pre}/{F_QUO}" in names:
                        ps.append(f"{pre}/{F_QUO}")
                    targets[c] = ps
                    break
        flat = [p for ps in targets.values() for p in ps]
        tmp = tempfile.mkdtemp(prefix=f"min1{day}_", dir="/tmp")
        try:
            for i in range(0, len(flat), CHUNK):
                with py7zr.SevenZipFile(zf) as z:
                    z.extract(path=tmp, targets=flat[i:i + CHUNK])
            for c, ps in targets.items():
                try:
                    b = bars_for_code(Path(tmp) / ps[0], Path(tmp) / ps[1] if len(ps) > 1 else None,
                                      c.endswith(".SZ"))
                except Exception:
                    continue
                if b is None or b.empty:
                    continue
                b.insert(0, "code", c[:6])
                frames.append(b)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        return f"{day} 出错 {type(e).__name__}: {e}"
    if not frames:
        return f"{day} 空结果(未落盘)"
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "date", day)
    qc = df.groupby("code")["qc_vol_ratio"].first()
    tmp_f = f_out.with_suffix(".parquet.min1tmp")
    df.to_parquet(tmp_f, index=False)
    tmp_f.replace(f_out)
    return (f"{day} ok:{df['code'].nunique()}码 {len(df)}行 qc中位 {qc.median():.3f} "
            f"(偏离>5%: {(abs(qc - 1) > 0.05).sum()}码) {time.time() - t0:.0f}s")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.split("用法")[1])
    start, end = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)

    up = ROOT / os.environ.get("T1A_UNIVERSE", "data/universe/universe_pit.parquet")
    if not up.exists():
        sys.exit(f"找不到 universe: {up}")
    c6 = sorted(set(pd.read_parquet(up)["code"].astype(str)
                    .str.extract(r"(\d{6})")[0].dropna()))
    codes = [f"{c}.{'SH' if c[0] == '6' else 'SZ'}" for c in c6]

    days = sorted(p.stem for p in TICK.glob("*/*/*.7z") if start <= p.stem <= end)
    if not days:
        sys.exit(f"{TICK} 下没有 {start}~{end}")
    nw = int(os.environ.get("NW", "1"))
    print(f"股票 {len(codes)} 只 ({up.name})  交易日 {len(days)} 天 "
          f"{days[0]}~{days[-1]}  并发 {nw}  批 {CHUNK}  -> {OUT}", flush=True)

    t0 = time.time()
    with Pool(nw) as pool:
        for i, msg in enumerate(pool.imap_unordered(
                partial(run_day, codes=codes), days, chunksize=1), 1):
            if i % 10 == 0 or "ok" not in msg:
                el = time.time() - t0
                print(f"[{i}/{len(days)}] {msg}  {el / 60:.1f}min "
                      f"剩余~{el / i * (len(days) - i) / 60:.0f}min", flush=True)
    n = len(list(OUT.glob("*.parquet")))
    last = max((p.stem for p in OUT.glob("*.parquet")), default="-")
    print(f"完成: 面板 {n} 天, 最新 {last} -> {OUT}", flush=True)


if __name__ == "__main__":
    try:
        from proctitle import lowkey
        lowkey("mltask/feat")
    except Exception:
        pass
    main()
