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
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE_DIR = ROOT / "data" / "raw" / "kline"
UNIVERSE = ROOT / "data" / "universe" / "watchlist_216.json"
START_DATE = "20190101"    # 保留扩容后的训练历史, 日常全量重拉不得截回 2021

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


def universe_codes(watchlist_file=None):
    path = (ROOT / "data" / "universe" / watchlist_file) if watchlist_file else UNIVERSE
    w = json.loads(path.read_text())
    items = w.get("watchlist", w) if isinstance(w, dict) else w
    return [str(x["code"])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]


def all_local_codes():
    return sorted(p.stem for p in KLINE_DIR.glob("*.parquet"))


class FetchTimeoutError(Exception):
    """单次拉取超时"""


@contextmanager
def hard_timeout(seconds):
    """给无超时的 akshare 请求加硬墙

    akshare 的新浪/东财接口底层不传 timeout, 被限流时连接会无限挂住,
    重试逻辑永远等不到异常。用 SIGALRM 强制中断 (仅主线程生效,
    多进程模式下每个 worker 都是自己的主线程, 所以可靠)。
    """
    if seconds <= 0:
        yield
        return
    try:
        signal.signal(signal.SIGALRM,
                      lambda *_: (_ for _ in ()).throw(FetchTimeoutError()))
    except ValueError:
        yield          # 不在主线程 (多线程模式), 退化为无保护
        return
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def _finalize(df):
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df[[c for c in FINAL_COLS if c in df.columns]]


def fetch_sina(code, end_date, start_date=START_DATE):
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date=start_date,
                             end_date=end_date, adjust="qfq")
    if df is None or not len(df):
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in FINAL_COLS[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return _finalize(df)


def fetch_em(code, end_date, start_date=START_DATE):
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=str(code)[:6], period="daily",
                            start_date=start_date, end_date=end_date, adjust="qfq")
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


def _finalize_tx(df, code):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if str(code)[:6].startswith("000"):
        df["volume"] *= 100.0
    df["outstanding_share"] = (df["volume"] / df["turnover"]).replace(
        [float("inf"), float("-inf")], pd.NA)
    return _finalize(df)


def fetch_tx(code, end_date, start_date=START_DATE, timeout=25):
    import akshare as ak
    symbol = sina_symbol(code)
    if symbol.startswith("bj"):
        return None
    df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start_date,
                               end_date=end_date, adjust="qfq", timeout=timeout)
    if df is None or not len(df):
        return None
    return _finalize_tx(df, code)


def fetch_one(code, end_date, timeout=25, start_date=START_DATE,
              sina_tries=3):
    """按 新浪 -> 东财 -> 腾讯 的顺序取数, 返回 (df, 来源名)

    为什么要多试新浪几次再降级 (2026-08-04)
    ──────────────────────────────────────
    三个源的【前复权口径不一致】: 同一只股票, 新浪与腾讯的日收益率最大能差
    3.1%(实测 000001), 相关性 0.998 而非 1.0 —— 也就是说复权后的收益率本身
    就不一样, 不只是价格水平的差异。

    而原实现是"新浪一失败立刻换源", 限流稍微抖一下就降级。实测每次全量拉取
    约 6% 的股票落到备用源, 且【每次是哪些股票都不一样】:
        同一批 400 只, 前后两次拉取: 16 只新变成备用源, 15 只修回正常源
    后果是训练数据不可复现 —— 同样的代码同样的参数, 两次拉取得到不同的价格
    序列, 于是特征筛选、IC、回测结论都会漂移。这比"少数股票口径不同"严重得多。

    所以: 新浪多试几次(带退避)再降级, 把降级压到真正取不到数的少数股票。
    并把实际用的来源返回给调用方记进 manifest, 让这件事可观测、可度量。
    """
    for i in range(max(1, sina_tries)):
        try:
            with hard_timeout(timeout):
                df = fetch_sina(code, end_date, start_date)
            if df is not None and len(df):
                return df, "sina"
        except Exception:
            pass
        if i < sina_tries - 1:
            time.sleep(0.8 * (i + 1))
    try:
        with hard_timeout(timeout):
            df = fetch_em(code, end_date, start_date)
        if df is not None and len(df):
            return df, "em"
    except Exception:
        pass
    with hard_timeout(timeout):
        df = fetch_tx(code, end_date, start_date, timeout)
    return df, ("tx" if df is not None and len(df) else None)


def process(code, end_date, dry_run, retries=3, timeout=25, start_date=START_DATE):
    out = KLINE_DIR / f"{code}.parquet"
    old_max = None
    if out.exists():
        try:
            o = pd.read_parquet(out, columns=["date"])
            old_max = pd.to_datetime(o["date"]).max()
        except Exception:
            pass
    # 返回值多带一个"实际用了哪个源": 三个源的前复权口径不一致, 必须可追溯,
    # 否则数据里混着不同口径而无从察觉 (见 fetch_one 的注释)
    for attempt in range(retries):
        try:
            df, src = fetch_one(code, end_date, timeout, start_date)
            if df is None or not len(df):
                return code, "empty", old_max, None, src
            new_max = df["date"].max()
            if dry_run:
                return code, "dry", old_max, new_max, src
            tmp = out.with_suffix(".tmp.parquet")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            added = 0 if old_max is None else int((df["date"] > old_max).sum())
            return code, f"ok+{added}", old_max, new_max, src
        except Exception as e:
            if attempt == retries - 1:
                return code, f"err:{type(e).__name__}", old_max, None, None
            time.sleep(1.5 * (attempt + 1))
    return code, "err", old_max, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["universe", "all"], default="universe")
    ap.add_argument("--watchlist", default=None,
                    help="data/universe/ 下的股票池 json (仅 --scope universe 生效)")
    ap.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--start", default=START_DATE,
                    help="历史起点 YYYYMMDD。改早会全量重拉覆盖 (前复权本就需要整段自洽)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--procs", type=int, default=0,
                    help="多进程并发数 (>0 时用多进程绕开新浪 py_mini_racer 线程不安全)")
    ap.add_argument("--timeout", type=int, default=25,
                    help="单只单次拉取硬超时秒数 (0=关闭)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--codes", default=None,
                    help="只拉指定的代码(逗号分隔)或代码清单文件路径(每行一个)。"
                         "用于补拉个别股票, 例如已退市股 —— 它们只有腾讯源有数据, "
                         "而 all_local_codes() 是按本地已有文件列的, 不含从未拉到过的股")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.codes:
        _p = Path(a.codes)
        _raw = (_p.read_text(encoding="utf-8").splitlines() if _p.exists()
                else a.codes.split(","))
        codes = sorted({x.split("#")[0].strip().zfill(6)[:6]
                        for x in _raw if x.split("#")[0].strip()})
    else:
        codes = (universe_codes(a.watchlist) if a.scope == "universe"
                 else all_local_codes())
    if a.limit:
        codes = codes[:a.limit]
    KLINE_DIR.mkdir(parents=True, exist_ok=True)

    if a.procs > 0:
        mode = f"{a.procs} 进程"
    else:
        if a.workers != 1:
            print(f"[注意] 新浪源底层 py_mini_racer 非线程安全, 已将 workers 从 {a.workers} 强制改为 1"
                  f" (要并发请用 --procs N)")
            a.workers = 1
        mode = f"{a.workers} 线程"

    print(f"更新日K线 | scope={a.scope} | {len(codes)} 只 | {a.start}~{a.end_date} | "
          f"{mode} | {'DRY-RUN' if a.dry_run else '写入'}")
    t0 = time.time()
    ok = err = empty = 0
    added_total = 0
    newest = None
    # 记录每只股票实际用了哪个源。三个源的前复权口径不一致(见 fetch_one),
    # 混在一起而不可见是个隐患: 训练数据会随每次拉取悄悄变化。
    src_of = {}
    pool_class = ProcessPoolExecutor if a.procs > 0 else ThreadPoolExecutor
    n_par = a.procs if a.procs > 0 else a.workers
    with pool_class(max_workers=n_par) as ex:
        futs = {ex.submit(process, c, a.end_date, a.dry_run, 3, a.timeout,
                          a.start): c
                for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            code, st, om, nm, src = f.result()
            if src:
                src_of[code] = src
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

    # ── 数据来源分布 ──────────────────────────────────────
    # 必须每次打出来: 备用源占比一旦升高, 说明相当一部分股票的复权口径变了,
    # 而这会让训练数据不可复现(同参数两次拉取得到不同价格序列)。
    if src_of and not a.dry_run:
        from collections import Counter
        cnt = Counter(src_of.values())
        tot = sum(cnt.values())
        print("\n数据来源分布 (三个源的前复权口径不一致, 备用源占比越低越好):")
        for k in ("sina", "em", "tx"):
            if cnt.get(k):
                print(f"  {k:5} {cnt[k]:5} 只 ({cnt[k]/tot*100:5.1f}%)"
                      + ("   <- 主源" if k == "sina" else "   <- 备用源, 口径与主源不同"))
        mani = KLINE_DIR.parent / "kline_source_manifest.json"
        prev = {}
        if mani.exists():
            try:
                prev = json.loads(mani.read_text(encoding="utf-8")).get("source_of", {})
            except Exception:
                prev = {}
        flipped = [c for c, v in src_of.items() if c in prev and prev[c] != v]
        if flipped:
            print(f"  本次有 {len(flipped)} 只股票的来源发生变化 -> 它们的历史收益率被改写")
            print(f"    例: {flipped[:8]}")
        # 必须【合并】而不是覆盖: 部分拉取(--codes/--limit)只涉及少数股票,
        # 整个覆盖会把其余几千只的来源记录抹掉, 于是这份清单就失去了意义。
        merged = dict(prev)
        merged.update(src_of)
        mani.write_text(json.dumps(
            {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "start": a.start,
             "scope": (f"codes({len(codes)})" if a.codes else a.scope),
             "counts_this_run": dict(cnt), "flipped_from_last": flipped,
             "n_total_tracked": len(merged),
             "source_of": merged}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  来源清单已写入 {mani}")


if __name__ == "__main__":
    main()
