"""构建 point-in-time 股票池所需的元数据底表

产出 data/universe/pit_metadata.parquet:
    code        6位代码
    name        简称
    list_date   真实上市日期 (交易所权威)
    delist_date 终止上市日期 (在市股票为 NaT)
    board       主板/创业板/科创板/北交所
    is_st_now   当前是否风险警示 (仅当前快照, 历史ST不可得)

数据源均为交易所/非东财接口, 规避东财 IP 封禁。
"""
import signal
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/universe/pit_metadata.parquet"


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(label, fn, sec=60):
    signal.alarm(sec)
    try:
        d = fn()
        signal.alarm(0)
        n = len(d) if d is not None else 0
        print(f"  {label:34s} {n:>5} 行")
        return d
    except T:
        signal.alarm(0)
        print(f"  {label:34s} 超时")
    except Exception as e:
        signal.alarm(0)
        print(f"  {label:34s} FAIL {type(e).__name__}: {str(e)[:45]}")
    return None


def board_of(code: str) -> str:
    c = str(code)
    if c.startswith("688"):
        return "科创板"
    if c.startswith(("300", "301")):
        return "创业板"
    if c.startswith(("83", "87", "43", "92")):
        return "北交所"
    return "主板"


rows = {}   # code -> dict

print("=== 在市股票 + 上市日期 ===")
for sym in ("主板A股", "科创板"):
    d = go(f"沪市 {sym}", lambda s=sym: ak.stock_info_sh_name_code(symbol=s))
    time.sleep(0.8)
    if d is None:
        continue
    cc = next(c for c in d.columns if "证券代码" in str(c))
    nc = next(c for c in d.columns if "证券简称" in str(c))
    dc = next(c for c in d.columns if "上市日期" in str(c))
    for _, r in d.iterrows():
        code = str(r[cc]).zfill(6)
        rows[code] = {"code": code, "name": str(r[nc]),
                      "list_date": pd.to_datetime(r[dc], errors="coerce"),
                      "delist_date": pd.NaT}

d = go("深市 stock_info_sz_name_code", lambda: ak.stock_info_sz_name_code(symbol="A股列表"))
time.sleep(0.8)
if d is not None:
    cc = next((c for c in d.columns if "证券代码" in str(c) or "A股代码" in str(c)), None)
    nc = next((c for c in d.columns if "证券简称" in str(c) or "A股简称" in str(c)), None)
    dc = next((c for c in d.columns if "上市日期" in str(c)), None)
    if cc and nc:
        for _, r in d.iterrows():
            code = str(r[cc]).zfill(6)
            rows[code] = {"code": code, "name": str(r[nc]),
                          "list_date": pd.to_datetime(r[dc], errors="coerce") if dc else pd.NaT,
                          "delist_date": pd.NaT}

print("\n=== 退市股票 ===")
d = go("上交所 stock_info_sh_delist", lambda: ak.stock_info_sh_delist())
time.sleep(0.8)
if d is not None:
    for _, r in d.iterrows():
        code = str(r["公司代码"]).zfill(6)
        rows[code] = {"code": code, "name": str(r["公司简称"]),
                      "list_date": pd.to_datetime(r["上市日期"], errors="coerce"),
                      "delist_date": pd.to_datetime(r.get("暂停上市日期"), errors="coerce")}

d = go("深交所 stock_info_sz_delist", lambda: ak.stock_info_sz_delist(symbol="终止上市公司"))
time.sleep(0.8)
if d is not None:
    cc = next((c for c in d.columns if "证券代码" in str(c)), None)
    nc = next((c for c in d.columns if "证券简称" in str(c)), None)
    lc = next((c for c in d.columns if "上市日期" in str(c)), None)
    tc = next((c for c in d.columns if "终止上市日期" in str(c)), None)
    if cc:
        for _, r in d.iterrows():
            code = str(r[cc]).zfill(6)
            rows[code] = {"code": code, "name": str(r[nc]) if nc else "",
                          "list_date": pd.to_datetime(r[lc], errors="coerce") if lc else pd.NaT,
                          "delist_date": pd.to_datetime(r[tc], errors="coerce") if tc else pd.NaT}

print("\n=== 当前风险警示(ST) ===")
st_codes = set()
d = go("stock_zh_a_st_em", lambda: ak.stock_zh_a_st_em())
if d is not None:
    st_codes = {str(c).zfill(6) for c in d["代码"]}

m = pd.DataFrame(rows.values())
m["board"] = m["code"].map(board_of)
m["is_st_now"] = m["code"].isin(st_codes)

OUT.parent.mkdir(parents=True, exist_ok=True)
m.sort_values("code").to_parquet(OUT, index=False)

print(f"\n已写入 {OUT}")
print(f"  合计 {len(m)} 只 | 在市 {m['delist_date'].isna().sum()} | "
      f"已退市 {m['delist_date'].notna().sum()} | 当前ST {m['is_st_now'].sum()}")
print(f"  上市日期可得: {m['list_date'].notna().sum()} 只")
print(f"\n  板块分布:\n{m['board'].value_counts().to_string()}")

win = m[(m["delist_date"] >= "2023-09-20") & (m["delist_date"] <= "2026-07-20")]
print(f"\n  回测窗口内(2023-09~2026-07)退市: {len(win)} 只  <- 幸存者偏差直接来源")
if len(win):
    print(win.sort_values("delist_date")[["code", "name", "delist_date"]]
          .tail(10).to_string(index=False))
