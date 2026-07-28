"""确认公告接口可用性 (主机 np-anotice-stock 已确认通)"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))


def go(name, fn, sec=45):
    t0 = time.time()
    signal.alarm(sec)
    try:
        df = fn()
        signal.alarm(0)
        print(f"\n[{name}] OK {len(df)} 行 {time.time()-t0:.1f}s")
        print(f"  列: {list(df.columns)}")
        print(df.head(4).to_string(index=False)[:600])
        return df
    except T:
        signal.alarm(0)
        print(f"\n[{name}] 超时 >{sec}s")
    except Exception as e:
        signal.alarm(0)
        print(f"\n[{name}] FAIL {type(e).__name__}: {str(e)[:120]}")
    return None


go("全市场公告 stock_notice_report 2026-07-27",
   lambda: ak.stock_notice_report(symbol="全部", date="2026-07-27"))

print("\n" + "=" * 70)
print("本地 announcements 现状")
import pathlib
fs = sorted(pathlib.Path('data/raw/announcements').glob('*.parquet'))
mx = None
for f in fs[:60]:
    d = pd.read_parquet(f, columns=['date'])
    m = pd.to_datetime(d['date'], errors='coerce').max()
    if pd.notna(m) and (mx is None or m > mx):
        mx = m
print(f"  {len(fs)} 个文件, 抽查前60个最新日期 = {mx}")
d0 = pd.read_parquet(fs[0])
print(f"  样例列: {list(d0.columns)}")
print(f"  type 取值样例: {d0['type'].value_counts().head(8).to_dict()}")
