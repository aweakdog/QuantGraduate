# -*- coding: utf-8 -*-
"""T1A 订单结构日度抽取 (跑在 eez040, 逐笔仓库所在机)

逐笔成交 -> 按委托ID聚合成"订单" -> 每码每日 4 个比率:
    t1a_big_buy    大单买入占比: 买方订单里成交量前 10% 分位的订单, 其量占买方总量
    t1a_long_buy   多笔买入占比: 买方订单里成交笔数 >=3 的订单, 其量占买方总量
    t1a_big_sell   同上, 卖方
    t1a_long_sell  同上, 卖方
产出 data/processed/t1a_daily/<YYYYMMDD>.parquet, 由 041 的
scripts/build_t1_augmented.py 做 ma5 平滑后并进训练矩阵。

为什么是订单而不是成交
──────────────────────
券商研究(国信/华泰)的"改进大单占比"口径是按**委托**统计的: 一笔大委托可能被
拆成几十笔小成交, 只看成交会把它误判成散户。所以必须先用 叫买/叫卖序号 把
成交聚合回订单, 再按订单量分档。这是本族与既有 tk_* 逐笔列(OFI/TED/委托族,
不含订单大小与时长分类)的区别所在。

口径锁定 (2026-08-31 入库)
──────────────────────────
本文件的算法与研究构建 (/tmp/t1a_extract.py, 产出 965 天面板, T1A20 20 面板
+18.6pp / p≈0.004 的证据就是它跑出来的) **逐字一致**, 任何"顺手优化"都会让
线上模型脱离已验证的构造。特别地:
  * 大单阈值 = 该侧当日订单量的 0.9 分位 (每码每日各自算, 不用全市场统一阈值)
  * 多笔阈值 = 订单内成交笔数 >= 3
  * SZ 撤单在**成交**文件里 (成交代码=='C'), 必须先剔除; SH 的撤单在委托文件,
    本脚本不读委托文件所以天然无此问题 (沪深编码不同, 搞错=静默垃圾)
  * 订单序号 <= 0 的行丢弃 (集合竞价/异常行)
  * **每侧订单数 < 30 的码日整条丢弃**: 分位数在小样本上没有意义。这条是
    "宁可缺值也不产垃圾"——LightGBM 原生吃 NaN, 但吃不了假比率
  * 股票池 = universe_pit.parquet (519 只), 与 965 天历史面板同池。
    ⚠ 别改成 tick_micro 那样的两池并集: 面板会变成"老日子 519 只 / 新日子 630 只"
    的非同质覆盖, 正是 watchlist_216 那类"覆盖集怎么选出来的"陷阱。扩池要
    整段重抽, 用 T1A_UNIVERSE 覆盖。

用法
────
    python scripts/t1a_order_features.py 20260901 20260901     # 单日
    NW=8 python scripts/t1a_order_features.py 20220901 20260828  # 补历史
幂等: 已有日文件直接跳过, 所以可以反复跑同一区间。
单日约 52~89s (519 只, 单进程), 补历史时 NW 别吃满 —— 共享机器。
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

try:      # 只有真抽取才需要; 缺包时仍可导入本模块跑算法测试
    import py7zr
except ImportError:
    py7zr = None

ROOT = Path(__file__).resolve().parents[1]
TICK = Path(os.environ.get(
    "TICK_DIR", "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
OUT = ROOT / Path(os.environ.get("T1A_OUT", "data/processed/t1a_daily"))

F_TRD = "逐笔成交.csv"
U_TRD = ("成交代码", "BS标志", "成交数量", "叫卖序号", "叫买序号")
QTY = "成交数量"     # 与 tick_micro_features.U_TRD 同一列名, 不做别名兜底

CHUNK = int(os.environ.get("CHUNK", "50"))
MIN_ORDERS = 30      # 每侧订单数下限, 低于此码日整条丢弃
BIG_Q = 0.9          # 大单 = 该侧当日订单量的 0.9 分位以上
LONG_MIN_FILLS = 3   # 多笔 = 订单内成交笔数 >= 3

T1A_COLS = ("t1a_big_buy", "t1a_long_buy", "t1a_big_sell", "t1a_long_sell")


def resolve(day: str):
    """2017-2021 用 <年>/<01..12>/, 2022+ 用 <年>/<年月>/ —— 两种命名都要认"""
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def ratios_for_code(d: pd.DataFrame, is_sz: bool):
    """单只股票单日的 4 个比率; 任一侧订单不足就返回 None (整码丢弃)"""
    if is_sz and "成交代码" in d.columns:
        d = d[d["成交代码"] != "C"]        # SZ 撤单在成交文件里
    out = {}
    for pre, flag, idc in (("buy", "B", "叫买序号"), ("sell", "S", "叫卖序号")):
        sd = d[d["BS标志"] == flag]
        g = sd.groupby(idc)[QTY].agg(["sum", "size"])
        g = g[g.index > 0]
        tot = g["sum"].sum()
        if tot <= 0 or len(g) < MIN_ORDERS:
            return None
        thr = g["sum"].quantile(BIG_Q)
        out[f"t1a_big_{pre}"] = g.loc[g["sum"] >= thr, "sum"].sum() / tot
        out[f"t1a_long_{pre}"] = g.loc[g["size"] >= LONG_MIN_FILLS, "sum"].sum() / tot
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
        tmp = tempfile.mkdtemp(prefix=f"t1a{day}_", dir="/tmp")
        try:
            # 分批解压: 一次 targets 全给 py7zr 会把整包摊进内存
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
                    continue      # 单只坏文件不拖累整天
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        return f"{day} 出错 {type(e).__name__}: {e}"
    if not rows:
        return f"{day} 空结果(未落盘)"
    # 原子落盘: 半截文件会被后续 run 的 exists() 当成"已完成"
    tmp_f = f_out.with_suffix(".parquet.t1atmp")
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
    try:      # htop 低调化 (见 scripts/proctitle.py)
        from proctitle import lowkey
        lowkey("mltask/feat")
    except Exception:
        pass
    main()
