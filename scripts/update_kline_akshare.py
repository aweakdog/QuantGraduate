"""用 akshare 更新日K线 (Mac 可用, 替代 Windows-only 的 xtdata/thsdk)

数据源: 新浪 stock_zh_a_daily (qfq)
  - 已验证与本地现有 kline 口径一致 (000063/300750 偏差 0.0000%)
  - 返回列与本地 schema 完全相同: date/open/high/low/close/volume/amount/
    outstanding_share/turnover
  - 东财 stock_zh_a_hist 当前 IP 被限流(RemoteDisconnected), 仅作后备

注意1: 前复权价格在除权除息后会【追溯改变】, 所以采用【全量重拉覆盖】而非追加,
       以保证单只股票历史内部口径自洽。
注意2: 新浪接口底层用 py_mini_racer 解密, 【非线程安全】, 多线程会直接 crash,
       故强制 workers=1。216 只约 8 分钟。

用法:
    python scripts/update_kline_akshare.py --scope universe        # 仅216池
    python scripts/update_kline_akshare.py --scope all             # 全市场5533
    python scripts/update_kline_akshare.py --scope universe --dry-run --limit 5
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE_DIR = ROOT / "data" / "raw" / "kline"
UNIVERSE = ROOT / "data" / "universe" / "watchlist_216.json"
START_DATE = "20210101"

FINAL_COLS = ["date", "open", "high", "low", "close", "volume",
              "amount", "outstanding_share", "turnover"]
EM_COLMAP = {"日期": "date", "开盘": "open", "收盘": "close",
             "最高": "high", "最低": "low", "成交量": "volume",
             "成交额": "amount", "换手率": "turnover"}


def sina_symbol(code):
    c = str(code)[:6]
    if c.startswith("6"):
        return "sh" + c
    if c.startswith(("4", "8", "9")):
        return "bj" + c
    return "sz" + c


def universe_codes():
    w = json.loads(UNIVERSE.read_text())
    items = w.get("watchlist", w) if isinstance(w, dict) else w
    return [str(x["code"])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]


def all_local_codes():
    return sorted(p.stem for p in KLINE_DIR.glob("*.parquet"))


def _finalize(df):
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df[[c for c in FINAL_COLS if c in df.columns]]


def fetch_sina(code, end_date):
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date=START_DATE,
                             end_date=end_date, adjust="qfq")
    if df is None or not len(df):
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in FINAL_COLS[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _finalize(df)


def fetch_em(code, end_date):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=str(code)[:6], period="daily",
                            start_date=START_DATE, end_date=end_date, adjust="qfq")
    if df is None or not len(df):
        return None
    df = df.rename(columns=EM_COLMAP)
    df = df[[c for c in EM_COLMAP.values() if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0     # 手 -> 股
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0  # % -> 小数
    for c in ("open", "high", "low", "close", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outstanding_share"] = (df["volume"] / df["turnover"]).replace(
        [float("inf"), float("-inf")], pd.NA)
    return _finalize(df)


def fetch_one(code, end_date):
    try:
        df = fetch_sina(code, end_date)
        if df is not None and len(df):
            return df
    except Exception:
        pass
    return fetch_em(code, end_date)


def process(code, end_date, dry_run, retries=3):
    out = KLINE_DIR / f"{code}.parquet"
    old_max = None
    if out.exists():
        try:
            o = pd.read_parquet(out, columns=["date"])
            old_max = pd.to_datetime(o["date"]).max()
        except Exception:
            pass
    for attempt in range(retries):
        try:
            df = fetch_one(code, end_date)
            if df is None or not len(df):
                return code, "empty", old_max, None
            new_max = df["date"].max()
            if dry_run:
                return code, "dry", old_max, new_max
            tmp = out.with_suffix(".tmp.parquet")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            added = 0 if old_max is None else int((df["date"] > old_max).sum())
            return code, f"ok+{added}", old_max, new_max
        except Exception as e:
            if attempt == retries - 1:
                return code, f"err:{type(e).__name__}", old_max, None
            time.sleep(1.5 * (attempt + 1))
    return code, "err", old_max, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["universe", "all"], default="universe")
    ap.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    codes = universe_codes() if a.scope == "universe" else all_local_codes()
    if a.limit:
        codes = codes[:a.limit]
    KLINE_DIR.mkdir(parents=True, exist_ok=True)

    if a.workers != 1:
        print(f"[注意] 新浪源底层 py_mini_racer 非线程安全, 已将 workers 从 {a.workers} 强制改为 1")
        a.workers = 1

    print(f"更新日K线 | scope={a.scope} | {len(codes)} 只 | 截止 {a.end_date} | "
          f"{a.workers} 线程 | {'DRY-RUN' if a.dry_run else '写入'}")
    t0 = time.time()
    ok = err = empty = 0
    added_total = 0
    newest = None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, c, a.end_date, a.dry_run): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            code, st, om, nm = f.result()
            if st.startswith("ok"):
                ok += 1
                added_total += int(st.split("+")[1])
            elif st == "dry":
                ok += 1
            elif st == "empty":
                empty += 1
            else:
                err += 1
                if err <= 5:
                    print(f"  [!] {code}: {st}")
            if nm is not None and (newest is None or nm > newest):
                newest = nm
            if i % 50 == 0 or i == len(codes):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(codes) - i) / rate if rate else 0
                print(f"  [{i}/{len(codes)}] ok={ok} err={err} empty={empty} "
                      f"新增{added_total}行 | {el:.0f}s | {rate:.1f}只/s | 剩余~{eta/60:.1f}min",
                      flush=True)

    print(f"\n完成 {time.time()-t0:.0f}s | ok={ok} err={err} empty={empty}")
    print(f"新增 {added_total} 行 | 最新交易日 {newest.date() if newest is not None else '-'}")


if __name__ == "__main__":
    main()
