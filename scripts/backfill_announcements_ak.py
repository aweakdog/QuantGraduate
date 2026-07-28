"""用 akshare 批量补全个股公告 (Mac 可用)

数据源: 东财 stock_notice_report(symbol='全部', date=YYYYMMDD) —— 按【交易日】
        返回全市场当日公告(约 1000+ 条), 一次请求覆盖所有股票, 比逐股抓高效得多。

两阶段:
  1. fetch : 逐交易日抓取, 落盘到 data/raw/announcements/_bulk/{YYYYMMDD}.parquet
             (断点续跑: 已存在的日期跳过)
  2. merge : 把 _bulk 按股票拆分, 与既有 data/raw/announcements/{code6}.parquet
             按 (date,title,url) 去重合并

用法:
  python scripts/backfill_announcements_ak.py --start 2022-06-01
  python scripts/backfill_announcements_ak.py --merge-only
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / "data/raw/announcements"
BULK_DIR = ANN_DIR / "_bulk"
CAL_SRC = ROOT / "data/raw/kline/000001.parquet"

COL_MAP = {"公告日期": "date", "公告标题": "title", "网址": "url", "公告类型": "type"}


def bar(i, n, t0, tag=""):
    pct = i / n if n else 1
    filled = int(pct * 30)
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {i}/{n} ({pct*100:5.1f}%) "
          f"用时 {el/60:.1f}min ETA {eta/60:.1f}min  {tag:<12}", end="", flush=True)


def trading_days(start, end):
    d = pd.read_parquet(CAL_SRC, columns=["date"])["date"]
    d = pd.to_datetime(d)
    return sorted(d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].unique())


def _fetch_day(ak, ds, retry):
    """抓单日全市场公告, 返回 'ok'/'empty'/'fail'"""
    for attempt in range(retry):
        try:
            df = ak.stock_notice_report(symbol="全部", date=ds)
            if df is None or not len(df):
                # 写空文件占位, 避免重复请求无公告的日期
                pd.DataFrame(columns=["代码", "公告日期", "公告标题", "网址", "公告类型"]) \
                    .to_parquet(BULK_DIR / f"{ds}.parquet", index=False)
                return "empty"
            df.to_parquet(BULK_DIR / f"{ds}.parquet", index=False)
            return "ok"
        except Exception:
            if attempt < retry - 1:
                time.sleep(1.5 * (attempt + 1))
    return "fail"


def fetch(a):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import akshare as ak

    BULK_DIR.mkdir(parents=True, exist_ok=True)
    days = trading_days(a.start, a.end)
    todo = [f"{pd.Timestamp(d):%Y%m%d}"
            for d in days if not (BULK_DIR / f"{pd.Timestamp(d):%Y%m%d}.parquet").exists()]
    print(f"交易日 {len(days)} 天 | 待抓 {len(todo)} 天 (已有 {len(days)-len(todo)} 天) | "
          f"并发 {a.workers}")

    stats = {"ok": 0, "empty": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(_fetch_day, ak, ds, a.retry): ds for ds in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            stats[fut.result()] += 1
            bar(i, len(todo), t0, futs[fut])
    print(f"\nfetch 完成: 有数据 {stats['ok']} 天, 空 {stats['empty']} 天, "
          f"失败 {stats['fail']} 天 | 用时 {(time.time()-t0)/60:.1f} min")


def merge(a):
    files = sorted(BULK_DIR.glob("*.parquet"))
    if not files:
        print("没有 _bulk 数据, 先跑 fetch")
        return
    print(f"合并 {len(files)} 个日文件 ...")
    parts = []
    for p in files:
        try:
            d = pd.read_parquet(p)
        except Exception:
            continue
        if len(d):
            parts.append(d)
    bulk = pd.concat(parts, ignore_index=True)
    bulk["代码"] = bulk["代码"].astype(str).str.zfill(6)
    bulk = bulk.rename(columns=COL_MAP)
    bulk["date"] = pd.to_datetime(bulk["date"], errors="coerce")
    bulk = bulk.dropna(subset=["date"])
    print(f"  共 {len(bulk):,} 条公告, {bulk['代码'].nunique()} 只股票, "
          f"{bulk['date'].min():%Y-%m-%d} ~ {bulk['date'].max():%Y-%m-%d}")

    if a.watchlist:
        w = json.loads((ROOT / "data/universe" / a.watchlist).read_text(encoding="utf-8"))
        want = {s["code"][:6] for s in w["watchlist"]}
        bulk = bulk[bulk["代码"].isin(want)]
        print(f"  过滤到股票池: {len(bulk):,} 条, {bulk['代码'].nunique()} 只")

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    new = updated = 0
    t0 = time.time()
    groups = list(bulk.groupby("代码"))
    for i, (code, g) in enumerate(groups, 1):
        bar(i, len(groups), t0, code)
        g = g[["date", "title", "url", "type"]].copy()
        path = ANN_DIR / f"{code}.parquet"
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old["date"] = pd.to_datetime(old["date"], errors="coerce")
                g = pd.concat([old, g], ignore_index=True)
                updated += 1
            except Exception:
                new += 1
        else:
            new += 1
        g = (g.dropna(subset=["date"])
              .drop_duplicates(subset=["date", "title", "url"])
              .sort_values("date").reset_index(drop=True))
        g.to_parquet(path, index=False)
    print(f"\nmerge 完成: 新建 {new} 只, 更新 {updated} 只")

    if a.watchlist:
        w = json.loads((ROOT / "data/universe" / a.watchlist).read_text(encoding="utf-8"))
        codes = [s["code"][:6] for s in w["watchlist"]]
        have = sum((ANN_DIR / f"{c}.parquet").exists() for c in codes)
        print(f"股票池覆盖: {have}/{len(codes)} ({100*have/len(codes):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--watchlist", default="watchlist_pit.json")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6, help="并发抓取线程数")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    a = ap.parse_args()
    if a.end is None:
        a.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if not a.merge_only:
        fetch(a)
    if not a.fetch_only:
        merge(a)


if __name__ == "__main__":
    main()
