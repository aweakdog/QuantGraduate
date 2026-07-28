"""校准 mtss_balance 口径: 交易所官方两融明细 vs 本地现有值

本地 mtss_balance 来自 thsdk/iFinD, 需确认其定义是:
  (a) 融资余额          融资买入后未偿还
  (b) 融资融券余额       融资余额 + 融券余量金额
只有口径确认一致才能续接。
"""
import signal
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(fn, sec=40):
    signal.alarm(sec)
    try:
        r = fn()
        signal.alarm(0)
        return r
    except T:
        signal.alarm(0)
        print("    超时")
    except Exception as e:
        signal.alarm(0)
        print(f"    FAIL {type(e).__name__}: {str(e)[:60]}")
    return None


loc = pd.read_parquet(FF)
loc["date"] = pd.to_datetime(loc["date"])
mt = loc[loc["mtss_balance"].notna()]
print(f"本地 mtss_balance: {len(mt):,} 行, {mt['code'].nunique()} 只, "
      f"{mt['date'].min().date()} ~ {mt['date'].max().date()}")
print(f"  数值样例: {mt['mtss_balance'].describe()[['min','50%','max']].to_dict()}")

# 选一个本地有数据的交易日做口径比对
probe_dates = sorted(mt["date"].unique())[-6:]
print(f"\n候选比对日: {[str(pd.Timestamp(d).date()) for d in probe_dates]}")

for d in reversed(probe_dates):
    ds = pd.Timestamp(d).strftime("%Y%m%d")
    print(f"\n=== 比对日 {pd.Timestamp(d).date()} ===")

    print("  [上交所 stock_margin_detail_sse]")
    sse = go(lambda: ak.stock_margin_detail_sse(date=ds))
    if sse is None or not len(sse):
        continue
    print(f"    {len(sse)} 行, 列: {list(sse.columns)}")
    time.sleep(0.6)

    print("  [深交所 stock_margin_detail_szse]")
    szse = go(lambda: ak.stock_margin_detail_szse(date=ds))
    if szse is not None and len(szse):
        print(f"    {len(szse)} 行, 列: {list(szse.columns)}")
    else:
        print("    不可用")

    # 组装交易所侧
    parts = []
    s = sse.rename(columns={"标的证券代码": "code", "融资余额": "rz",
                            "融券余量金额": "rq_amt"})
    s["code"] = s["code"].astype(str).str.zfill(6)
    if "rq_amt" not in s.columns:
        s["rq_amt"] = 0.0
    parts.append(s[["code", "rz", "rq_amt"]])

    if szse is not None and len(szse):
        cmap = {}
        for c in szse.columns:
            cs = str(c)
            if "证券代码" in cs:
                cmap[c] = "code"
            elif cs == "融资余额":
                cmap[c] = "rz"
            elif "融券余量金额" in cs or cs == "融券余额":
                cmap[c] = "rq_amt"
        z = szse.rename(columns=cmap)
        if "code" in z.columns and "rz" in z.columns:
            z["code"] = z["code"].astype(str).str.zfill(6)
            if "rq_amt" not in z.columns:
                z["rq_amt"] = 0.0
            parts.append(z[["code", "rz", "rq_amt"]])

    ex = pd.concat(parts, ignore_index=True)
    ex["rz"] = pd.to_numeric(ex["rz"], errors="coerce")
    ex["rq_amt"] = pd.to_numeric(ex["rq_amt"], errors="coerce").fillna(0)
    ex["rz_rq"] = ex["rz"] + ex["rq_amt"]
    print(f"    交易所侧合计 {len(ex)} 只")

    cur = mt[mt["date"] == d][["code", "mtss_balance"]].copy()
    cur["code"] = cur["code"].astype(str).str.zfill(6)
    m = cur.merge(ex, on="code", how="inner").dropna(subset=["mtss_balance", "rz"])
    print(f"    与本地重叠 {len(m)} 只")
    if len(m) < 10:
        continue

    for col, lbl in [("rz", "融资余额"), ("rz_rq", "融资融券余额")]:
        dev = (m[col] / m["mtss_balance"] - 1).abs()
        print(f"    口径[{lbl:8s}] 中位偏差 {dev.median()*100:8.4f}%  "
              f"<1%的占比 {100*(dev<0.01).mean():5.1f}%  相关性 {m[col].corr(m['mtss_balance']):.6f}")
    print(f"    样例: 本地={m['mtss_balance'].iloc[0]:,.0f}  "
          f"融资={m['rz'].iloc[0]:,.0f}  融资融券={m['rz_rq'].iloc[0]:,.0f}  ({m['code'].iloc[0]})")
    break
