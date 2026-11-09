"""逐笔包 v3 (2026-11-03 起, parquet) 与 v1 (CSV) 的口径等价性。

同一组成交/委托/快照事件分别写成 v1 CSV 和 v3 parquet, 抽出来的特征必须逐值相同
(只有 prev_close/amihud 例外: v3 快照没有 前收盘, 由 041 端用日线补)。
这就把 tick_pack 里每一条映射 (Side 0=主动买、委托用原始流、SZ 撤单、集合竞价成交
照样标方向、价格单位) 都钉住了: 任何一条写反, 特征会悄悄变而不报错。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_tick_augmented as bta  # noqa: E402
import t1a_order_features as t1a  # noqa: E402
import tick_micro_features as tmf  # noqa: E402
import tick_pack  # noqa: E402

DAY = "20261103"


def _hms(sec):
    return int((sec // 3600) * 10000000 + (sec // 60 % 60) * 100000 + (sec % 60) * 1000)


def _cont_times(rng, n):
    s = np.concatenate([rng.integers(34200, 41400, n // 2), rng.integers(46800, 53820, n - n // 2)])
    return [_hms(x) for x in np.sort(s)]


def _events(is_sh, seed=0):
    rng = np.random.default_rng(seed)
    pc = 1000                                     # 昨收 10.00 元, 单位: 分
    trd, ordr, quo = [], [], []
    aggr = 1000
    times = [92500000] * 5 + _cont_times(rng, 300) + [150000000] * 5
    for i, t in enumerate(times):
        aggr += 1
        passive = int(rng.integers(1, 60))
        buy_aggr = bool(rng.integers(0, 2))
        bid, sid = (aggr, passive) if buy_aggr else (passive, aggr)
        trd.append(dict(t=t, id=i + 1, px=pc + int(rng.integers(-20, 21)),
                        q=100 * int(rng.integers(1, 50)), bid=bid, sid=sid, cxl=0))
    for t in _cont_times(rng, 20):                # 撤单: SZ 在成交文件, SH 在委托文件
        b = bool(rng.integers(0, 2))
        oid = int(rng.integers(1, 60))
        trd.append(dict(t=t, id=0, px=0, q=100 * int(rng.integers(1, 20)),
                        bid=oid if b else 0, sid=0 if b else oid, cxl=1 if b else 2))
    trd.sort(key=lambda r: r["t"])
    pre = sorted(_hms(x) for x in rng.integers(33300, 33899, 15))
    for k, t in enumerate(pre + _cont_times(rng, 200)):
        mkt = (not is_sh) and k % 25 == 7
        ordr.append(dict(t=t, id=k + 1, buy=bool(rng.integers(0, 2)), mkt=mkt,
                         px=0 if mkt else pc + int(rng.integers(-30, 31)),
                         q=100 * int(rng.integers(1, 80)), last=pc if t < 92500000 else pc + 3))
    for s in range(34200, 54000, 180):
        k = int(rng.integers(-10, 11))
        quo.append(dict(t=_hms(s), px=pc + k, ask=pc + k + 1, bid=pc + k,
                        aq=100 * int(rng.integers(1, 500)), bq=100 * int(rng.integers(1, 500)),
                        ta=1000 * int(rng.integers(1, 900)), tb=1000 * int(rng.integers(1, 900))))
    return pc, trd, ordr, quo


def _write_v1(base: Path, code, is_sh, ev):
    pc, trd, ordr, quo = ev
    base.mkdir(parents=True)
    rows = []
    for r in trd:
        if r["cxl"] and is_sh:
            continue
        bs = "" if r["cxl"] else ("B" if r["bid"] > r["sid"] else "S")   # 竞价成交也按序号标
        rows.append({"万得代码": code, "交易所代码": code[:6], "自然日": DAY, "时间": r["t"],
                     "成交编号": r["id"], "成交代码": " " if is_sh else ("C" if r["cxl"] else "0"),
                     "委托代码": "0", "BS标志": bs, "成交价格": r["px"] * 100, "成交数量": r["q"],
                     "叫卖序号": r["sid"], "叫买序号": r["bid"]})
    pd.DataFrame(rows).to_csv(base / "逐笔成交.csv", index=False, encoding="gbk")
    rows = [{"时间": o["t"], "委托编号": 0, "交易所委托号": o["id"],
             "委托类型": "A" if is_sh else ("1" if o["mkt"] else "0"),
             "委托代码": "B" if o["buy"] else "S", "委托价格": o["px"] * 100, "委托数量": o["q"]}
            for o in ordr]
    if is_sh:
        rows += [{"时间": r["t"], "委托编号": 0, "交易所委托号": r["bid"] or r["sid"], "委托类型": "D",
                  "委托代码": "B" if r["cxl"] == 1 else "S", "委托价格": 0, "委托数量": r["q"]}
                 for r in trd if r["cxl"]]
        rows.sort(key=lambda x: x["时间"])
    pd.DataFrame(rows).to_csv(base / "逐笔委托.csv", index=False, encoding="gbk")
    pd.DataFrame([{"时间": q["t"], "成交价": q["px"] * 100, "当日累计成交量": 0, "当日成交额": 0,
                   "申卖价1": q["ask"] * 100, "申买价1": q["bid"] * 100, "申卖量1": q["aq"],
                   "申买量1": q["bq"], "叫卖总量": q["ta"], "叫买总量": q["tb"],
                   "前收盘": pc * 100} for q in quo]).to_csv(base / "行情.csv", index=False, encoding="gbk")
    return {"trd": base / "逐笔成交.csv", "ord": base / "逐笔委托.csv", "quo": base / "行情.csv"}


def _write_v3(root: Path, code, is_sh, ev):
    pc, trd, ordr, quo = ev
    c6 = code[:6]
    paths = {k: root / DAY / f"{tick_pack.V3[k]}{DAY}" / f"{c6}.parquet" for k in tick_pack.V3}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    t_rows, o_rows = [], []
    for r in trd:
        if r["cxl"]:
            side = -1 if r["cxl"] == 1 else -11
            o_rows.append({"OrderTime": r["t"], "OrderID": r["bid"] or r["sid"], "OrderType": side,
                           "Price": pc, "Volume": float(r["q"]), "LastPrice": pc + 3})
            if is_sh:
                continue
        else:
            side = 0 if r["bid"] > r["sid"] else 1
        t_rows.append({"SecuCode": int(c6), "TradingDay": int(DAY), "DealTime": r["t"],
                       "DealID": r["id"], "BuyID": r["bid"], "SellID": r["sid"], "Price": r["px"],
                       "Volume": float(r["q"]), "Side": side, "Channel": 1, "BizIndex": -9})
    pd.DataFrame(t_rows).to_parquet(paths["trd"], index=False)
    for o in ordr:
        typ = (0 if is_sh else (1 if o["mkt"] else 2)) + (0 if o["buy"] else 10)
        o_rows.append({"OrderTime": o["t"], "OrderID": o["id"], "OrderType": typ,
                       "Price": o["px"], "Volume": float(o["q"]), "LastPrice": o["last"]})
    od = pd.DataFrame(o_rows).sort_values("OrderTime", kind="stable")
    od.insert(0, "SecuCode", int(c6))
    od.to_parquet(paths["ord"], index=False)
    pd.DataFrame([{"SecuCode": float(c6), "TickTime": float(q["t"]), "Price": float(q["px"]),
                   "AskPrice1": float(q["ask"]), "BidPrice1": float(q["bid"]),
                   "AskVolume1": float(q["aq"]), "BidVolume1": float(q["bq"]),
                   "TotalAskVolume": float(q["ta"]), "TotalBidVolume": float(q["tb"]),
                   "TotalVolume": 0.0, "TotalTurnover": 0.0} for q in quo]).to_parquet(paths["quo"], index=False)
    return paths


def _write_v4(root: Path, code, is_sh, ev):
    """v4 = v3 的内容换成每股一个目录的中文列头 CSV (列名照供应商说明)"""
    v3 = _write_v3(root / "src", code, is_sh, ev)
    base = root / DAY / code[:6]
    base.mkdir(parents=True)
    out = {}
    for k, cols in tick_pack.V4_COLS.items():
        d = pd.read_parquet(v3[k]).rename(columns={v: c for c, v in cols.items()})
        d.insert(0, "交易所代码", code[:6])
        out[k] = base / tick_pack.V4[k]
        d[["交易所代码", *cols]].to_csv(out[k], index=False, encoding="utf-8")
    return out


def _same(a, b, skip=()):
    assert a.keys() == b.keys()
    for k in a:
        if k in skip:
            continue
        if isinstance(a[k], str):
            assert a[k] == b[k], k
        else:
            assert np.isclose(a[k], b[k], equal_nan=True, rtol=1e-12, atol=1e-15), (k, a[k], b[k])


@pytest.mark.parametrize("code", ["000001.SZ", "600000.SH"])
def test_v3_features_equal_v1(tmp_path, code):
    is_sh = code.endswith(".SH")
    ev = _events(is_sh, seed=7)
    f1 = _write_v1(tmp_path / "v1" / code, code, is_sh, ev)
    f3 = _write_v3(tmp_path / "v3", code, is_sh, ev)
    r1, r3 = tmf.one_stock(f1, code, DAY), tmf.one_stock(f3, code, DAY)
    assert r1 is not None and np.isfinite(r1["cxl_rate_b"]) and np.isfinite(r1["amihud"])
    _same(r1, r3, skip=("prev_close", "amihud"))
    assert np.isnan(r3["prev_close"]) and np.isnan(r3["amihud"])


@pytest.mark.parametrize("code", ["000001.SZ", "600000.SH"])
def test_v4_csv_features_equal_v3_and_v1(tmp_path, code):
    is_sh = code.endswith(".SH")
    ev = _events(is_sh, seed=11)
    r1 = tmf.one_stock(_write_v1(tmp_path / "v1" / code, code, is_sh, ev), code, DAY)
    f4 = _write_v4(tmp_path / "v4", code, is_sh, ev)
    r3, r4 = tmf.one_stock(_write_v3(tmp_path / "v3", code, is_sh, ev), code, DAY), tmf.one_stock(f4, code, DAY)
    _same(r3, r4)
    _same(r1, r4, skip=("prev_close", "amihud"))
    d = tick_pack.read(f4["trd"], "trd", t1a.U_TRD, is_sh)
    assert t1a.ratios_for_code(d, not is_sh) == t1a.ratios_for_code(
        tick_pack.read(_write_v3(tmp_path / "v3b", code, is_sh, ev)["trd"], "trd", t1a.U_TRD, is_sh), not is_sh)


@pytest.mark.parametrize("code", ["000001.SZ", "600000.SH"])
def test_v3_t1a_ratios_equal_v1(tmp_path, code):
    is_sh = code.endswith(".SH")
    ev = _events(is_sh, seed=3)
    f1 = _write_v1(tmp_path / "v1" / code, code, is_sh, ev)
    f3 = _write_v3(tmp_path / "v3", code, is_sh, ev)
    d1 = pd.read_csv(f1["trd"], encoding="gbk", usecols=lambda c: c in t1a.U_TRD)
    d3 = tick_pack.read(f3["trd"], "trd", t1a.U_TRD, is_sh)
    r1, r3 = t1a.ratios_for_code(d1, not is_sh), t1a.ratios_for_code(d3, not is_sh)
    assert r1 is not None
    _same(r1, r3)


def test_v3_sz_orders_drop_cancel_duplicates(tmp_path):
    ev = _events(False, seed=1)
    f3 = _write_v3(tmp_path / "v3", "000001.SZ", False, ev)
    o = tick_pack.read(f3["ord"], "ord", tmf.U_ORD, False)
    assert len(o) == len(ev[2])
    assert set(o["委托类型"]) <= {"0", "1"} and set(o["委托代码"]) == {"B", "S"}


def test_v3_quote_has_no_pre_close(tmp_path):
    """委托 LastPrice 开盘前回填的是当日开盘价, 不是昨收 —— 不许拿它冒充 前收盘"""
    f3 = _write_v3(tmp_path / "v3", "600000.SH", True, _events(True, seed=2))
    q = tick_pack.read(f3["quo"], "quo", tmf.U_QUO, True)
    assert "前收盘" not in q.columns and "申卖价1" in q.columns


def test_fill_amihud_v3_from_qfq_kline(tmp_path):
    pd.DataFrame({"date": ["2026-10-30", "2026-11-02", "2026-11-03"],
                  "close": [10.0, 10.5, 10.29]}).to_parquet(tmp_path / "000001.parquet")
    d = pd.to_datetime
    df = pd.DataFrame({"c6": ["000001", "000001", "000001", "000002"],
                       "date": d(["2026-10-30", "2026-11-02", "2026-11-03", "2026-11-03"]),
                       "amihud": [np.nan, 0.7, np.nan, np.nan],
                       "day_amt": [1e8, 1e8, 2e8, 1e8]})
    out = bta.fill_amihud_v3(df.copy(), kline_dir=tmp_path)
    assert np.isnan(out["amihud"].iloc[0])                  # v3 之前的空值不动 (历史不变)
    assert out["amihud"].iloc[1] == 0.7                     # 已有值不动
    assert np.isclose(out["amihud"].iloc[2], 0.02 / 2)      # |10.29/10.5-1| / (2e8/1e8)
    assert np.isnan(out["amihud"].iloc[3])                  # 没日线 -> 仍为空


def test_members_and_layout():
    v1 = [f"{DAY}/000001.SZ/逐笔成交.csv", f"{DAY}/000001.SZ/逐笔委托.csv", "600000.SH/逐笔成交.csv"]
    lay, m = tick_pack.members(v1, DAY)
    assert lay == "v1" and set(m) == {"000001.SZ", "600000.SH"} and set(m["000001.SZ"]) == {"trd", "ord"}
    v3 = [f"{DAY}/逐笔成交{DAY}/000001.parquet", f"{DAY}/逐笔委托{DAY}/000001.parquet",
          f"{DAY}/逐笔委托原始{DAY}/600000.parquet", f"{DAY}/行情快照{DAY}/600000.parquet"]
    lay, m = tick_pack.members(v3, DAY)
    assert lay == "v3" and m == {          # 委托只认原始流, 加工版 逐笔委托 不要
        "000001.SZ": {"trd": f"{DAY}/逐笔成交{DAY}/000001.parquet"},
        "600000.SH": {"ord": f"{DAY}/逐笔委托原始{DAY}/600000.parquet",
                      "quo": f"{DAY}/行情快照{DAY}/600000.parquet"}}
    assert tick_pack.layout(["20261102/000001.SZ/逐笔成交.dat"]) == "v2"
    v4 = [f"{DAY}/000001/逐笔成交.csv", f"{DAY}/000001/逐笔委托.csv", f"{DAY}/000001/逐笔委托原始.csv",
          f"{DAY}/000001/行情快照.csv"]
    lay, m = tick_pack.members(v4, DAY)
    assert lay == "v4" and m == {"000001.SZ": {"trd": v4[0], "ord": v4[2], "quo": v4[3]}}
