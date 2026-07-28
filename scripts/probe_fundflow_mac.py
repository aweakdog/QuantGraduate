"""资金流现状 + Mac 可用源探测

本地字段 (feature_engine.calc_fund_features 消费):
  mf_net / mf_pct / dde_net / mtss / fund_flow ...
目标: 找到能续接这些字段的 Mac 可用日频源
"""
import signal
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))

print("=" * 64)
print("【1. 本地资金流现状】")
full = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"
if full.exists():
    d = pd.read_parquet(full)
    dc = next((c for c in d.columns if "date" in str(c).lower() or "日期" in str(c)), None)
    s = pd.to_datetime(d[dc], errors="coerce")
    print(f"  合并档 fundflow_history.parquet")
    print(f"    {len(d):,} 行 | 列: {list(d.columns)}")
    print(f"    日期 {s.min().date()} ~ {s.max().date()}  (滞后 {(pd.Timestamp('2026-07-27')-s.max()).days} 天)")
    cc = next((c for c in d.columns if "code" in str(c).lower() or "代码" in str(c)), None)
    if cc:
        print(f"    覆盖 {d[cc].nunique()} 只")
    print(f"\n    最后3行:\n{d.sort_values(dc).tail(3).to_string()[:600]}")

pd_dir = ROOT / "data/raw/fund_flow"
fs = sorted(pd_dir.glob("*.parquet"))
if fs:
    d0 = pd.read_parquet(fs[0])
    print(f"\n  按股票目录 fund_flow/: {len(fs)} 个文件")
    print(f"    样例 {fs[0].stem} 列: {list(d0.columns)}")

print("\n" + "=" * 64)
print("【2. akshare 资金流接口探测 (Mac)】")


def go(name, fn, sec=30):
    t0 = time.time()
    signal.alarm(sec)
    try:
        df = fn()
        signal.alarm(0)
    except T:
        signal.alarm(0)
        print(f"  {name:42s} 超时>{sec}s")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"  {name:42s} FAIL {type(e).__name__}: {str(e)[:38]}")
        return None
    if df is None or not len(df):
        print(f"  {name:42s} 空")
        return None
    dc = next((c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()), None)
    mx = ""
    if dc:
        s = pd.to_datetime(df[dc], errors="coerce")
        if s.notna().any():
            mx = f" 最新={s.max().date()}"
    print(f"  {name:42s} OK {len(df):>5}行 {time.time()-t0:4.1f}s{mx}")
    print(f"      列: {list(df.columns)[:10]}")
    time.sleep(0.8)
    return df


import akshare as ak

print("\n-- 个股资金流历史 (最关键) --")
go("stock_individual_fund_flow 000063 sz",
   lambda: ak.stock_individual_fund_flow(stock="000063", market="sz"))
go("stock_individual_fund_flow 600519 sh",
   lambda: ak.stock_individual_fund_flow(stock="600519", market="sh"))

print("\n-- 同花顺个股资金流 --")
go("stock_fund_flow_individual 即时",
   lambda: ak.stock_fund_flow_individual(symbol="即时"))

print("\n-- 大盘/板块资金流 --")
go("stock_market_fund_flow", lambda: ak.stock_market_fund_flow())
go("stock_sector_fund_flow_rank 今日",
   lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流"))

print("\n-- 融资融券 (mtss 字段来源) --")
go("stock_margin_sse 近期",
   lambda: ak.stock_margin_sse(start_date="20260701", end_date="20260727"))
go("stock_margin_detail_sse 单日",
   lambda: ak.stock_margin_detail_sse(date="20260724"))
go("stock_margin_underlying_info_szse",
   lambda: ak.stock_margin_underlying_info_szse(date="20260724"))

print("\n-- 龙虎榜 / 北向 --")
go("stock_lhb_detail_em",
   lambda: ak.stock_lhb_detail_em(start_date="20260720", end_date="20260724"))
