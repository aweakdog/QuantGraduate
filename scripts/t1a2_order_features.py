# -*- coding: utf-8 -*-
"""T1A 二期(研究臂): 订单结构族扩列的日度抽取 (跑在 eez040, 逐笔仓库所在机)

生产在用的 T1A 4 列(t1a_order_features.py, 口径已锁)只覆盖了券商框架里的
"主动订单 x {大单, 多笔}"两个格子。本抽取器补两个没碰过的维度, 每码每日 4 个比率:

    t1a2_dur_buy    漫长买单占比: 按叫买序号聚合**全部**成交(不分 BS 标志)得到每个买订单的
                    一生, 首末成交间隔 >= 60s 的订单量占当日总量 —— 挂单慢成交/先吃后挂的
                    耐心资金足迹。注意不能只看主动侧: 主动单的自身成交全在同一撮合周期,
                    时长恒为 0(20260904 实测), 它剩余部分挂在盘口被别人打成的那些成交带反向标志
    t1a2_dur_sell   同上, 按叫卖序号
    t1a2_pbig_buy   被动大单买占比: 被主动卖打到的挂单买订单(BS=S 按叫买序号聚合)里,
                    量前 10% 分位的订单, 其量占该侧总量 —— 挂大单承接(冰山/机构吸筹)
    t1a2_pbig_sell  同上, 被动卖侧(BS=B 按叫卖序号聚合)

与 T1A 共用的口径(逐字沿用, 保证两族可比):
  * 大单阈值 = 该侧当日订单量 0.9 分位, 每码每日各自算
  * SZ 撤单在成交文件里(成交代码=='C'), 先剔; 订单序号 <= 0 丢弃
  * 任一侧订单数 < 30 的码日整条丢弃(分位数在小样本上没意义, 宁缺勿假)
  * 股票池 = universe_pit.parquet(519 只), 与 t1a_daily 面板同池
新增口径:
  * 时长阈值 60s 固定(不用分位: 多数订单单笔成交, 时长 0, 分位阈值在冷门股上退化为 0)
  * 时间列 HHMMSSmmm 不是线性的, 先换算成当日秒数再求差; 跨午休的订单时长含 90 分钟
    午休, 反正 >= 60s 一律算漫长, 不影响判定

这是研究臂, 不进夜链; 过探索门后再决定是否合进 t1a_order_features.py(那时同样口径锁定)。
产出 data/processed/t1a2_daily/<YYYYMMDD>.parquet, 由 scripts/build_t1a2_augmented.py
做 ma5 + lag1 并进研究矩阵。

用法
────
    python scripts/t1a2_order_features.py 20260901 20260901        # 单日
    NW=8 python scripts/t1a2_order_features.py 20220901 20261231   # 补历史(共享机器, NW 别吃满)
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

import pandas as pd

try:
    import py7zr
except ImportError:
    py7zr = None

ROOT = Path(__file__).resolve().parents[1]
TICK = Path(os.environ.get(
    "TICK_DIR", "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
OUT = ROOT / Path(os.environ.get("T1A2_OUT", "data/processed/t1a2_daily"))

F_TRD = "逐笔成交.csv"
U_TRD = ("时间", "成交代码", "BS标志", "成交数量", "叫卖序号", "叫买序号")
QTY = "成交数量"

CHUNK = int(os.environ.get("CHUNK", "50"))
MIN_ORDERS = 30
BIG_Q = 0.9
DUR_MIN_SEC = 60.0

T1A2_COLS = ("t1a2_dur_buy", "t1a2_dur_sell", "t1a2_pbig_buy", "t1a2_pbig_sell")


def resolve(day: str):
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def to_sec(t: pd.Series) -> pd.Series:
    """HHMMSSmmm -> 当日秒数 (逐笔时间戳不是线性的, 不能直接相减)"""
    t = pd.to_numeric(t, errors="coerce").fillna(0).astype("int64")
    return ((t // 10_000_000) * 3600 + ((t // 100_000) % 100) * 60
            + (t // 1000) % 100 + (t % 1000) / 1000.0)


def ratios_for_code(d: pd.DataFrame, is_sz: bool):
    """单只股票单日的 4 个比率; 任一侧订单不足就返回 None (整码丢弃)"""
    if is_sz and "成交代码" in d.columns:
        d = d[d["成交代码"] != "C"]
    d = d.assign(_sec=to_sec(d["时间"]))
    out = {}
    # 订单一生: 不分 BS 标志, 按买/卖订单号聚合全部成交
    for pre, idc in (("buy", "叫买序号"), ("sell", "叫卖序号")):
        g = d.groupby(idc).agg(vol=(QTY, "sum"), t0=("_sec", "min"), t1=("_sec", "max"))
        g = g[g.index > 0]
        tot = g["vol"].sum()
        if tot <= 0 or len(g) < MIN_ORDERS:
            return None
        out[f"t1a2_dur_{pre}"] = g.loc[(g["t1"] - g["t0"]) >= DUR_MIN_SEC, "vol"].sum() / tot
    # 被动侧: 主动卖(S)成交的买方订单 = 挂单买; 主动买(B)成交的卖方订单 = 挂单卖
    for pre, flag, idc in (("buy", "S", "叫买序号"), ("sell", "B", "叫卖序号")):
        sd = d[d["BS标志"] == flag]
        g = sd.groupby(idc)[QTY].sum()
        g = g[g.index > 0]
        tot = g.sum()
        if tot <= 0 or len(g) < MIN_ORDERS:
            return None
        thr = g.quantile(BIG_Q)
        out[f"t1a2_pbig_{pre}"] = g[g >= thr].sum() / tot
    return out


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
    rows = []
    try:
        with py7zr.SevenZipFile(zf) as z:
            names = set(z.getnames())
        targets = []
        for c in codes:
            for p in (f"{day}/{c}/{F_TRD}", f"{c}/{F_TRD}"):
                if p in names:
                    targets.append(p)
                    break
        tmp = tempfile.mkdtemp(prefix=f"t1a2{day}_", dir="/tmp")
        try:
            for i in range(0, len(targets), CHUNK):
                with py7zr.SevenZipFile(zf) as z:
                    z.extract(path=tmp, targets=targets[i:i + CHUNK])
            for rel in targets:
                code = rel.split("/")[-2]
                try:
                    d = pd.read_csv(Path(tmp) / rel, encoding="gbk", engine="c",
                                    usecols=lambda c: c.strip() in U_TRD)
                    d.columns = [c.strip() for c in d.columns]
                    r = ratios_for_code(d, code.endswith(".SZ"))
                    if r is not None:
                        rows.append({"date": day, "code": code[:6], **r})
                except Exception:
                    continue
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        return f"{day} 出错 {type(e).__name__}: {e}"
    if not rows:
        return f"{day} 空结果(未落盘)"
    tmp_f = f_out.with_suffix(".parquet.t1a2tmp")
    pd.DataFrame(rows).to_parquet(tmp_f, index=False)
    tmp_f.replace(f_out)
    return f"{day} ok:{len(rows)} {time.time() - t0:.0f}s"


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
          f"{days[0]}~{days[-1]}  并发 {nw}  批 {CHUNK}", flush=True)

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
