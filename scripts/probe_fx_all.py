"""列出 akshare 全部汇率相关接口并逐个试探, 找 USDJPY/USDCNH/USDIND 的可用日频源"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))

names = [n for n in dir(ak)
         if any(k in n.lower() for k in ("fx", "currency", "forex", "exchange", "rmb", "usd"))]
print(f"=== akshare 汇率相关接口 ({len(names)} 个) ===")
for n in names:
    print(f"  {n}")

print("\n=== 逐个试探 (无参调用, 20s 超时) ===")
skip = {"currency_boc_sina", "currency_boc_safe"}
for n in names:
    if n in skip:
        continue
    fn = getattr(ak, n)
    if not callable(fn):
        continue
    signal.alarm(20)
    try:
        df = fn()
        signal.alarm(0)
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"  {n:34s} 空/非DF")
            continue
        dc = next((c for c in df.columns
                   if "日期" in str(c) or "date" in str(c).lower() or "时间" in str(c)), None)
        latest = pd.to_datetime(df[dc], errors="coerce").max() if dc else None
        tag = f"最新={latest.date()}" if latest is not None and pd.notna(latest) else "无日期列"
        print(f"  {n:34s} OK {len(df):>6}行 {tag}")
        print(f"      列: {list(df.columns)[:9]}")
    except T:
        signal.alarm(0)
        print(f"  {n:34s} 超时")
    except TypeError as e:
        signal.alarm(0)
        print(f"  {n:34s} 需参数: {str(e)[:60]}")
    except Exception as e:
        signal.alarm(0)
        print(f"  {n:34s} FAIL {type(e).__name__}: {str(e)[:45]}")
    time.sleep(0.4)
