"""逐笔包格式适配: 各版本的包统一成 v1 的列名/取值口径, 下游抽取器
(tick_micro_features / t1a_order_features / tick_qc_early) 的算法一行不改。

版本 (2026-11-04 供应商换格式后实测)
────
v1  <=20261030  <日>/<代码>.<所>/{逐笔委托,逐笔成交,行情}.csv   GBK, 价格 1/10000 元
v2  20261102    目录同 v1, 文件是 .dat, 头 b"WDPK" + 加密/私有压缩 —— 读不了, 当缺包
                (供应商 11-04 已用 v3 格式重发同名包)
v3  20261103    <日>/{逐笔委托,逐笔成交,行情快照}<日>/<6位>.parquet   价格 1/100 元
v4  >=20261104  <日>/<6位>/{逐笔委托原始,逐笔委托,逐笔成交,行情快照}.csv   UTF-8, 中文列头 =
                供应商说明里的中文列名, 取值与 v3 逐列相同 (11-05 实测: 方向 0 = 主动买 100%、
                委托类型编码相同、价格单位分、成交合计 = 快照成交总量)。⚠ 成交价格/申卖价1 等列名
                与 v1 相同但单位差 100 倍, 只能按目录 (6 位代码、无交易所后缀) 认 v4, 不能按列名

v3 -> v1 (20261103 实测。供应商的字段说明有三处与数据不符, 以数据为准):
  成交 Side 0=主动买 B, 1=主动卖 S —— 说明写反了。两条独立证据: 主动方 = 序号大的一方
       (两版都 100% 成立); Side=0 的成交 91% 落在前一快照卖一及以上, Side=1 的 89% 在
       买一及以下。SZ 的 -1/-11 = 撤买/撤卖 (价 0, 量是撤单量而非说明写的 0)
       -> 成交代码 'C'、BS 空 (v1 SZ 撤单也在成交文件)。集合竞价成交也按 Side 标:
       v1 多数日子 SH 竞价成交按同一序号规则标了 B/S (10-22/10-27 抽查 100% 一致),
       只有 10-12、10-28~30 等少数日子供应商留空 (那几天 SH 成交额比日线少 1~2%)
  委托 必须用 "逐笔委托原始" (= 交易所原始流, 与 v1 同口径)。加工版 "逐笔委托" 的 SH 多了
       34535 条 (600000) 由成交反推的主动方订单, SZ 多了撤单重复行 -> 新增委托量虚高、
       撤单率虚低 (试抽实测 cxl_rate 掉 20%+)。
       SH OrderType 0/10 = 新增买/卖 (A), -1/-11 = 撤买/撤卖 (D) —— 委托价相对前成交价
       买低卖高, 方向分明; 说明里 "10=委托成交、-11=系统撤销" 不符。
       SZ 个位 1/2/3 = 市价/限价/本方最优 (v1 的 '1'/'0'/'U'), +10 = 卖 —— 2/12 各 10 万+
       且价格互为镜像, 说明里 "1=限价、2=IOC、12=对手方最优" 不符 (SZ 抽取不用委托类型)
  快照 没有 前收盘; 委托 LastPrice 在开盘前回填的是当日开盘价 (614 只 100% = 开盘价,
       只有 12% = 昨收), 不能当前收 -> prev_close/amihud 留空, 由 041 端用前复权日线补
       (build_tick_augmented.fill_amihud_v3)
  成交量合计 == 快照末笔 TotalVolume == 日线成交量 (000001 20261103 逐股全等)
"""
from pathlib import Path

import numpy as np
import pandas as pd

V1 = {"ord": "逐笔委托.csv", "trd": "逐笔成交.csv", "quo": "行情.csv"}
V3 = {"ord": "逐笔委托原始", "trd": "逐笔成交", "quo": "行情快照"}
V4 = {k: v + ".csv" for k, v in V3.items()}
V4_COLS = {   # v4 中文列 -> v3 列名, 之后与 v3 走同一套换算
    "trd": {"时间": "DealTime", "成交编号": "DealID", "叫买序号": "BuyID", "叫卖序号": "SellID",
            "成交价格": "Price", "成交数量": "Volume", "方向": "Side"},
    "ord": {"订单时间": "OrderTime", "交易所委托号": "OrderID", "委托类型": "OrderType",
            "委托价格": "Price", "委托数量": "Volume"},
    "quo": {"时间": "TickTime", "成交价": "Price", "成交总量": "TotalVolume", "成交总金额": "TotalTurnover",
            "申卖价1": "AskPrice1", "申买价1": "BidPrice1", "申卖量1": "AskVolume1", "申买量1": "BidVolume1",
            "未成交委卖总量": "TotalAskVolume", "未成交委买总量": "TotalBidVolume"}}
PX = 100                     # v3/v4 价格(分) -> v1 口径(1/10000 元)
SZ_TYPE = {1: "1", 2: "0", 3: "U"}


def code_of(c6: str) -> str:
    return f"{c6}.{'SH' if c6[0] == '6' else 'SZ'}"


def layout(names) -> str:
    names = list(names)
    if any(n.endswith("/" + V4["ord"]) for n in names):      # 先于 v1: 两者都有 /逐笔成交.csv
        return "v4"
    if any(n.endswith("/" + V1["trd"]) for n in names):
        return "v1"
    if any(n.endswith(".parquet") and f"/{V3['trd']}" in n for n in names):
        return "v3"
    if any(n.endswith("/逐笔成交.dat") for n in names):
        return "v2"
    return "?"


def members(names, day: str, kinds=("ord", "trd", "quo")):
    """返回 (版本, {代码.所: {kind: 包内路径}})。v1 有/无日期前缀两种目录都认。"""
    names = list(names)
    lay, out = layout(names), {}
    for n in names:
        parts = n.split("/")
        if len(parts) < 2:
            continue
        for k in kinds:
            if lay == "v1" and parts[-1] == V1[k]:
                out.setdefault(parts[-2], {})[k] = n
            elif lay == "v3" and parts[-2] == V3[k] + day and parts[-1].endswith(".parquet"):
                out.setdefault(code_of(parts[-1][:-len(".parquet")]), {})[k] = n
            elif lay == "v4" and parts[-1] == V4[k] and parts[-2].isdigit():
                out.setdefault(code_of(parts[-2]), {})[k] = n
    return lay, out


def is_v4(fp: Path) -> bool:
    return fp.suffix == ".csv" and fp.parent.name.isdigit()


def read_csv(fp: Path, use):
    """编码不统一: 多数天是 GBK, 但实测 20241127/20250827 是 UTF-8 带 BOM。
    编码猜错时列名解不出来, usecols 匹配为空, 会静默丢掉一整天, 所以必须逐个试。
    2018-07~2019-05 供应商批次的列头逗号后带空格 (" 成交价格"), 精确匹配会静默
    丢整天 (2026-08-19 回填 2017 时 136 天全军覆没查出来的), 所以匹配前先 strip。"""
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
            df = pd.read_csv(fp, encoding=e,
                             usecols=lambda c: c.strip() in use,
                             on_bad_lines="skip")
        except Exception:
            continue
        if len(df.columns):
            return df.rename(columns=lambda c: c.strip())
    return None


def _trd(d: pd.DataFrame, is_sh: bool) -> pd.DataFrame:
    side = d["Side"].to_numpy()
    tcode = np.full(len(d), " ") if is_sh else np.where(side < 0, "C", "0")
    return pd.DataFrame({"时间": d["DealTime"].astype("int64").to_numpy(),
                         "成交编号": d["DealID"].to_numpy(), "成交代码": tcode,
                         "BS标志": np.select([side == 0, side == 1], ["B", "S"], ""),
                         "成交价格": d["Price"].to_numpy() * PX,
                         "成交数量": d["Volume"].to_numpy(),
                         "叫卖序号": d["SellID"].to_numpy(), "叫买序号": d["BuyID"].to_numpy()})


def _ord(d: pd.DataFrame, is_sh: bool) -> pd.DataFrame:
    if not is_sh:      # 原始流本就没有; 防误用加工版时撤单重复行混进新增委托
        d = d[d["OrderType"] >= 0]
    ot = d["OrderType"].astype("int64").to_numpy()
    typ = (np.where(ot < 0, "D", "A") if is_sh
           else np.array([SZ_TYPE.get(x % 10, "?") for x in ot], dtype=object))
    return pd.DataFrame({"时间": d["OrderTime"].astype("int64").to_numpy(), "委托编号": 0,
                         "交易所委托号": d["OrderID"].to_numpy(), "委托类型": typ,
                         "委托代码": np.where(np.abs(ot) >= 10, "S", "B"),
                         "委托价格": d["Price"].to_numpy() * PX, "委托数量": d["Volume"].to_numpy()})


def _quo(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"时间": d["TickTime"].astype("int64").to_numpy(),
                         "成交价": d["Price"].to_numpy() * PX,
                         "当日累计成交量": d["TotalVolume"].to_numpy(),
                         "当日成交额": d["TotalTurnover"].to_numpy(),
                         "申卖价1": d["AskPrice1"].to_numpy() * PX, "申买价1": d["BidPrice1"].to_numpy() * PX,
                         "申卖量1": d["AskVolume1"].to_numpy(), "申买量1": d["BidVolume1"].to_numpy(),
                         "叫卖总量": d["TotalAskVolume"].to_numpy(), "叫买总量": d["TotalBidVolume"].to_numpy()})


def read(fp, kind: str, use, is_sh: bool):
    """kind ∈ ord/trd/quo; 返回只含 use 中列的 v1 口径 DataFrame, 读不出返回 None"""
    if fp is None or not Path(fp).exists():
        return None
    fp = Path(fp)
    if is_v4(fp):
        d = pd.read_csv(fp, encoding="utf-8", usecols=list(V4_COLS[kind])).rename(columns=V4_COLS[kind])
    elif fp.suffix != ".parquet":
        return read_csv(fp, use)
    else:
        d = pd.read_parquet(fp)
    d = _trd(d, is_sh) if kind == "trd" else _ord(d, is_sh) if kind == "ord" else _quo(d)
    return d[[c for c in d.columns if c in use]]
