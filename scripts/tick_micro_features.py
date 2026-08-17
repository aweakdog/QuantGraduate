# -*- coding: utf-8 -*-
"""从 Wind L2 逐笔委托/成交/快照抽取日频微观结构特征

为什么做这个
────────────
现有特征体系有两个硬伤, 逐笔数据同时治两个:

1. 覆盖偏差 (docs 方法论 §8.4)。84 个低覆盖列的"有值"集合 100% 落在一份 2026-07 手挑的
   216 只名单里, 模型学的是"这只票在名单里"而不是特征值本身。
   逐笔包每天含【全市场 7817 只】, 结构上不可能产生这种掩码 —— 这是最干净的特征来源。
2. 资金流源已死。同花顺 dde_net / fund_flow 2026-06-30 断更且无前向替代。
   逐笔可以自己算净委托流, 口径自己定, 而且能算到 2017 年。

沪深编码完全不同, 搞错就静默产出垃圾 (实测确认, scripts/tick_probe2.py 可复现):
                新增委托                    撤单                          成交
    SH   委托文件 委托类型=='A'        委托文件 委托类型=='D'         成交文件全部(BS标志 B/S)
    SZ   委托文件全部行               【成交】文件 成交代码=='C'      成交文件 成交代码=='0'
                                      价格恒0, 方向看 叫买/叫卖序号
                                      哪个非零 (已验证 100% 一致)
两所 委托编号 恒为 0, 只能用 交易所委托号; 价格单位都是 1/10000 元。

历史可用性 (实测): SH 逐笔委托 2017-2021 全缺, 2022 起才有; 其余全程有。
=> 委托类特征的全市场起点是 2022-01, 而训练矩阵起点 2022-09, 刚好完整覆盖。

产出 data/processed/tick_micro/<date>.parquet, 每行一个股票日, 约 30 列。
所有量都归一化成比例, 保证横截面可比。

用法
────
    python scripts/tick_micro_features.py 20220901 20260807
    NW=16 CHUNK=50 python scripts/tick_micro_features.py 20260801 20260807
"""
import os
import shutil
import sys
import tempfile
import time
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
OUT = ROOT / Path(os.environ.get("TM_OUT", "data/processed/tick_micro"))
PDIV = 10000.0

F_ORD, F_TRD, F_QUO = "逐笔委托.csv", "逐笔成交.csv", "行情.csv"
U_ORD = ["时间", "委托类型", "委托代码", "委托价格", "委托数量"]
U_TRD = ["时间", "成交代码", "BS标志", "成交价格", "成交数量", "叫卖序号", "叫买序号"]
U_QUO = ["时间", "成交价", "当日累计成交量", "当日成交额", "申卖价1", "申买价1",
         "申卖量1", "申买量1", "叫卖总量", "叫买总量", "前收盘"]

# 分档阈值 (元), 对齐同花顺/tushare 的单笔金额口径
BKT = [(0, 4e4, "s"), (4e4, 2e5, "m"), (2e5, 1e6, "l"), (1e6, np.inf, "xl")]

T_OPEN, T_OPEN30, T_CLOSE30, T_END = 93000000, 100000000, 143000000, 150000000


def resolve(day: str):
    """2017-2021 用 <年>/<01..12>/, 2022+ 用 <年>/<年月>/ —— 两种命名都要认"""
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def _sec(t):
    """HHMMSSmmm -> 当日秒数。逐笔时间戳不是线性的, 做时间分箱前必须换算"""
    t = np.asarray(t, dtype="int64")
    return (t // 10000000) * 3600 + (t // 100000 % 100) * 60 + (t // 1000 % 100)


def _rd(fp: Path, use):
    """编码不统一: 多数天是 GBK, 但实测 20241127/20250827 是 UTF-8 带 BOM。
    编码猜错时列名解不出来, usecols 匹配为空, 会静默丢掉一整天, 所以必须逐个试。"""
    if not fp.exists():
        return None
    try:
        with open(fp, "rb") as fh:
            bom = fh.read(3)
    except Exception:
        return None
    encs = ("utf-8-sig",) if bom == b"\xef\xbb\xbf" else ("gbk", "utf-8")
    for e in encs:
        try:
            df = pd.read_csv(fp, encoding=e, usecols=lambda c: c in use,
                             on_bad_lines="skip")
        except Exception:
            continue
        if len(df.columns):
            return df
    return None


def _bucket_net(amt, is_buy):
    """按单笔金额分档, 返回各档 (买额-卖额)"""
    out = {}
    for lo, hi, nm in BKT:
        m = (amt >= lo) & (amt < hi)
        out[nm] = float(amt[m & is_buy].sum() - amt[m & ~is_buy].sum())
        out[nm + "_amt"] = float(amt[m].sum())
    return out


def one_stock(base: Path, code: str, day: str):
    is_sh = code.endswith(".SH")
    trd = _rd(base / F_TRD, U_TRD)
    if trd is None or trd.empty or "成交价格" not in trd.columns:
        return None

    t_t = pd.to_numeric(trd["时间"], errors="coerce").fillna(0).astype("int64")
    px = pd.to_numeric(trd["成交价格"], errors="coerce").fillna(0) / PDIV
    qty = pd.to_numeric(trd["成交数量"], errors="coerce").fillna(0)
    bs = trd["BS标志"].astype(str)
    tcode = trd["成交代码"].astype(str) if "成交代码" in trd else pd.Series("0", index=trd.index)

    # 成交与撤单在 SZ 是混在同一个文件里的
    if is_sh:
        m_trd = bs.isin(["B", "S"]) & (px > 0) & (qty > 0)
        m_cxl = pd.Series(False, index=trd.index)
    else:
        m_trd = (tcode == "0") & (px > 0) & (qty > 0)
        m_cxl = tcode == "C"

    if m_trd.sum() < 10:
        return None
    amt = px * qty
    day_amt = float(amt[m_trd].sum())
    day_vol = float(qty[m_trd].sum())
    if day_amt <= 0 or day_vol <= 0:
        return None
    vwap = day_amt / day_vol
    close = float(px[m_trd].iloc[-1])
    prev_close = np.nan
    r = dict(code=code[:6], date=day, day_amt=day_amt,
             n_trd=int(m_trd.sum()), avg_trd_amt=day_amt / int(m_trd.sum()),
             vwap=vwap, close=close, close_vs_vwap=close / vwap - 1)

    # ---------- 主动成交分档 ----------
    a, q, tt = amt[m_trd], qty[m_trd], t_t[m_trd]
    buy = bs[m_trd].eq("B") if is_sh else bs[m_trd].eq("B")
    nb = _bucket_net(a, buy)
    for _, _, nm in BKT:
        r[f"trd_net_{nm}"] = nb[nm] / day_amt
    r["trd_xl_share"] = nb["xl_amt"] / day_amt
    r["act_buy_ratio"] = float(a[buy].sum()) / day_amt

    # ---------- 日内时段结构 ----------
    sgn = np.where(buy, 1.0, -1.0) * a.values
    m_o = (tt.values >= T_OPEN) & (tt.values < T_OPEN30)
    m_c = (tt.values >= T_CLOSE30) & (tt.values <= T_END)
    r["act_net_open30"] = float(sgn[m_o].sum()) / day_amt
    r["act_net_close30"] = float(sgn[m_c].sum()) / day_amt
    r["act_net_tail_vs_open"] = r["act_net_close30"] - r["act_net_open30"]
    r["amt_close30_share"] = float(a.values[m_c].sum()) / day_amt

    # ---------- 微观流动性: 5 分钟分箱 ----------
    # HHMMSSmmm 不是线性时间(095959999 + 1ms = 100000000, 差 40001), 直接整除会错位,
    # 必须先换算成当日秒数
    key = _sec(tt.values) // 300
    dfb = pd.DataFrame({"k": key, "px": px[m_trd].values,
                        "amt": a.values, "sgn": sgn})
    g = dfb.groupby("k").agg(px=("px", "last"), amt=("amt", "sum"),
                             sgn=("sgn", "sum"))
    if len(g) >= 10:
        ret = np.log(g["px"]).diff().dropna()
        r["rv_5m"] = float((ret ** 2).sum())
        x = g["sgn"].iloc[1:].values / day_amt
        y = ret.values
        if np.std(x) > 0:
            r["kyle_lambda"] = float(np.polyfit(x, y, 1)[0])
        else:
            r["kyle_lambda"] = np.nan
    else:
        r["rv_5m"] = r["kyle_lambda"] = np.nan

    # ---------- 撤单 ----------
    ordf = _rd(base / F_ORD, U_ORD)
    if ordf is not None and not ordf.empty and "委托数量" in ordf.columns:
        o_q = pd.to_numeric(ordf["委托数量"], errors="coerce").fillna(0)
        o_p = pd.to_numeric(ordf["委托价格"], errors="coerce").fillna(0) / PDIV
        o_d = ordf["委托代码"].astype(str)
        o_t = ordf["委托类型"].astype(str)
        if is_sh:
            m_add = (o_t == "A") & o_d.isin(["B", "S"])
            m_del = (o_t == "D") & o_d.isin(["B", "S"])
            del_b = o_q[m_del & o_d.eq("B")].sum()
            del_s = o_q[m_del & o_d.eq("S")].sum()
        else:
            m_add = o_d.isin(["B", "S"])
            m_del = pd.Series(False, index=ordf.index)
            # SZ 撤单在成交文件, 方向看哪个序号非零 (已验证 100% 与委托方向一致)
            cb = pd.to_numeric(trd["叫买序号"], errors="coerce").fillna(0)[m_cxl]
            cs = pd.to_numeric(trd["叫卖序号"], errors="coerce").fillna(0)[m_cxl]
            cq = qty[m_cxl]
            del_b = float(cq[cb > 0].sum())
            del_s = float(cq[cs > 0].sum())

        add_b = float(o_q[m_add & o_d.eq("B")].sum())
        add_s = float(o_q[m_add & o_d.eq("S")].sum())
        r["cxl_rate_b"] = del_b / add_b if add_b > 0 else np.nan
        r["cxl_rate_s"] = del_s / add_s if add_s > 0 else np.nan
        r["cxl_imb"] = r["cxl_rate_b"] - r["cxl_rate_s"]

        # ---------- 净委托流分档 (用限价单, 市价单价格为 0 时按 vwap 折算) ----------
        oa = o_q[m_add] * np.where(o_p[m_add] > 0, o_p[m_add], vwap)
        ob = o_d[m_add].eq("B")
        nbo = _bucket_net(oa, ob)
        tot_ord = float(oa.sum())
        for _, _, nm in BKT:
            r[f"ord_net_{nm}"] = nbo[nm] / day_amt
        r["ord_xl_share"] = nbo["xl_amt"] / tot_ord if tot_ord > 0 else np.nan
        r["ord_amt_ratio"] = tot_ord / day_amt
        r["ord_n"] = int(m_add.sum())
    else:
        for k in ["cxl_rate_b", "cxl_rate_s", "cxl_imb", "ord_xl_share",
                  "ord_amt_ratio", "ord_n"]:
            r[k] = np.nan
        for _, _, nm in BKT:
            r[f"ord_net_{nm}"] = np.nan

    # ---------- 盘口 ----------
    quo = _rd(base / F_QUO, U_QUO)
    if quo is not None and not quo.empty and "申卖价1" in quo.columns:
        qt = pd.to_numeric(quo["时间"], errors="coerce").fillna(0).astype("int64")
        m = (qt >= T_OPEN) & (qt <= T_END)
        ask = pd.to_numeric(quo["申卖价1"], errors="coerce")[m] / PDIV
        bid = pd.to_numeric(quo["申买价1"], errors="coerce")[m] / PDIV
        ok = (ask > 0) & (bid > 0)
        if ok.sum() > 0:
            mid = (ask[ok] + bid[ok]) / 2
            r["spread_bp"] = float(np.median((ask[ok] - bid[ok]) / mid)) * 1e4
        else:
            r["spread_bp"] = np.nan
        bq = pd.to_numeric(quo["申买量1"], errors="coerce")[m]
        aq = pd.to_numeric(quo["申卖量1"], errors="coerce")[m]
        tot = bq + aq
        r["depth_imb_1"] = float(((bq - aq) / tot)[tot > 0].mean())
        if "叫买总量" in quo.columns:
            tb = pd.to_numeric(quo["叫买总量"], errors="coerce")[m]
            ta = pd.to_numeric(quo["叫卖总量"], errors="coerce")[m]
            s = tb + ta
            r["depth_imb_all"] = float(((tb - ta) / s)[s > 0].mean())
        else:
            r["depth_imb_all"] = np.nan
        if "前收盘" in quo.columns:
            pc = pd.to_numeric(quo["前收盘"], errors="coerce")
            pc = pc[pc > 0]
            if len(pc):
                prev_close = float(pc.iloc[-1]) / PDIV
    else:
        r["spread_bp"] = r["depth_imb_1"] = r["depth_imb_all"] = np.nan

    r["prev_close"] = prev_close
    r["amihud"] = (abs(close / prev_close - 1) / (day_amt / 1e8)
                   if prev_close and prev_close == prev_close else np.nan)
    return r


def run_day(day: str, codes, chunk: int):
    outf = OUT / f"{day}.parquet"
    if outf.exists():
        return f"{day} skip"
    pack = resolve(day)
    if pack is None:
        return f"{day} 无包"
    with py7zr.SevenZipFile(pack, mode="r") as z:
        names = set(z.getnames())
    # 包内有两种结构: <日期>/<代码>/<文件> 与 <代码>/<文件>(无日期前缀)。
    # 写死日期前缀会整段漏掉后一种(实测 91 天, 集中在 2026 年), 所以扫一遍条目自动推断。
    dirs = {}
    tail = "/" + F_TRD
    for n in names:
        if n.endswith(tail):
            dirs[n.split("/")[-2]] = n[:-len(tail)]
    have = [c for c in codes if c in dirs]
    if not have:
        return f"{day} 无匹配"

    rows = []
    # 分批解压即用即删: 600 只全解开约 14GB, 多天并发会把磁盘和内存打满
    for i in range(0, len(have), chunk):
        part = have[i:i + chunk]
        tgt = [f"{dirs[c]}/{f}" for c in part for f in (F_ORD, F_TRD, F_QUO)
               if f"{dirs[c]}/{f}" in names]
        tmp = tempfile.mkdtemp(prefix=f"tm{day}_", dir="/tmp")
        try:
            with py7zr.SevenZipFile(pack, mode="r") as z:
                z.extract(path=tmp, targets=tgt)
            for c in part:
                base = Path(tmp) / dirs[c]
                if not base.exists():
                    continue
                try:
                    rec = one_stock(base, c, day)
                except Exception:
                    rec = None
                if rec:
                    rows.append(rec)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        return f"{day} 空"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(outf, index=False)
    return f"{day} ok {len(rows)}"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.split("用法")[1])
    start, end = sys.argv[1], sys.argv[2]

    # 取所有 universe 的并集: 线上池 519 只与实验池 630 只并不互相包含,
    # 一次抽完省得为第二个矩阵重跑一遍。TM_UNIVERSE 可整体覆盖 (逗号分隔, 扩池用)
    if os.environ.get("TM_UNIVERSE"):
        ups = [ROOT / p for p in os.environ["TM_UNIVERSE"].split(",")]
    else:
        ups = [ROOT / "data/universe/universe_pit.parquet",
               ROOT / "data/universe/universe_pit_2019.parquet"]
    ups = [p for p in ups if p.exists()]
    if not ups:
        sys.exit("找不到 universe 文件")
    c6 = set()
    for p in ups:
        c6 |= set(pd.read_parquet(p)["code"].astype(str)
                  .str.extract(r"(\d{6})")[0].dropna())
    c6 = sorted(c6)
    codes = [f"{c}.{'SH' if c[0] == '6' else 'SZ'}" for c in c6]
    upath = " + ".join(p.name for p in ups)

    days = sorted(p.stem for p in TICK.glob("*/*/*.7z") if start <= p.stem <= end)
    if not days:
        sys.exit(f"{TICK} 下没有 {start}~{end}")
    nw = int(os.environ.get("NW", "16"))
    chunk = int(os.environ.get("CHUNK", "50"))
    print(f"股票 {len(codes)} 只 ({upath})  交易日 {len(days)} 天 "
          f"{days[0]}~{days[-1]}  并发 {nw}  批 {chunk}", flush=True)

    from functools import partial
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(nw) as pool:
        for i, msg in enumerate(pool.imap_unordered(
                partial(run_day, codes=codes, chunk=chunk), days, chunksize=1), 1):
            if i % 10 == 0 or "无" in msg or "空" in msg:
                el = time.time() - t0
                print(f"[{i}/{len(days)}] {msg}  {el / 60:.1f}min "
                      f"剩余~{el / i * (len(days) - i) / 60:.0f}min", flush=True)
    print(f"完成, 输出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
