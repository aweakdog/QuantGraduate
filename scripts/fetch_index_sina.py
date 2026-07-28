"""用新浪/腾讯源抓 A股宽基指数 (东财被封时的备选), 与216池等权做对比"""
import json
import signal
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "2023-09-20", "2026-07-20"
OUT = ROOT / "data/raw/macro/index"


class T(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(T()))

IDX = [("sh000300", "沪深300"), ("sh000905", "中证500"), ("sz399006", "创业板指"),
       ("sh000688", "科创50"), ("sh000001", "上证指数"), ("sz399001", "深证成指"),
       ("sz399905", "中证500深"), ("sh000852", "中证1000")]


def try_src(fn, sym, sec=35):
    signal.alarm(sec)
    try:
        d = fn(symbol=sym)
        signal.alarm(0)
        return d
    except T:
        signal.alarm(0)
        return None
    except Exception:
        signal.alarm(0)
        return None


OUT.mkdir(parents=True, exist_ok=True)
res = {}
for sym, nm in IDX:
    d = try_src(ak.stock_zh_index_daily, sym)
    src = "sina"
    if d is None or not len(d):
        d = try_src(ak.stock_zh_index_daily_tx, sym)
        src = "tencent"
    if d is None or not len(d):
        print(f"  {nm:10s} 两源均失败")
        continue
    dc = "date" if "date" in d.columns else d.columns[0]
    d[dc] = pd.to_datetime(d[dc])
    d = d[(d[dc] >= START) & (d[dc] <= END)].sort_values(dc)
    if len(d) < 50:
        print(f"  {nm:10s} 区间数据不足 ({len(d)})")
        continue
    cc = "close" if "close" in d.columns else [c for c in d.columns if "clos" in str(c).lower()][0]
    c = pd.to_numeric(d[cc], errors="coerce").dropna()
    r = c.iloc[-1] / c.iloc[0] - 1
    res[nm] = r
    d.rename(columns={dc: "date", cc: "close"})[["date", "close"]].to_parquet(
        OUT / f"{sym}.parquet", index=False)
    print(f"  {nm:10s} [{src:7s}] {len(d):>4}行  总收益 {r*100:+8.1f}%")

if res:
    (OUT / "index_returns.json").write_text(
        json.dumps({k: round(v, 6) for k, v in res.items()}, ensure_ascii=False, indent=2))
    yrs = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    print(f"\n=== 年化 ({START} ~ {END}, {yrs:.2f}年) ===")
    for k, v in sorted(res.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} 总 {v*100:+8.1f}%   年化 {((1+v)**(1/yrs)-1)*100:+7.1f}%")
