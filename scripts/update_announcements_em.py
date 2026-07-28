"""增量更新公告 (Mac 可用) — 直连东财公告 API

为什么不用 akshare:
  ak.stock_notice_report(symbol="全部") 要翻数百页, 45s 都跑不完。
  这里直连 np-anotice-stock.eastmoney.com 按股票拉, 约 1 秒/只。
  (注: 被封的是 push2his.eastmoney.com, 公告主机 np-anotice-stock 正常)

输出: data/raw/announcements/{code6}.parquet
schema 与现有一致: [date, title, url, type]
增量: 只补本地最新日期之后的公告, 按 (date,title) 去重

用法:
    python scripts/update_announcements_em.py                  # 216池增量
    python scripts/update_announcements_em.py --limit 5 --dry-run
    python scripts/update_announcements_em.py --max-pages 8    # 补更久的缺口
"""
import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT / "data" / "raw" / "announcements"
UNIVERSE = ROOT / "data" / "universe" / "watchlist_216.json"

BASE = "https://np-anotice-stock.eastmoney.com/api/security/ann"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Referer": "https://data.eastmoney.com/"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def universe_codes():
    w = json.loads(UNIVERSE.read_text())
    items = w.get("watchlist", w) if isinstance(w, dict) else w
    return [str(x["code"])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]


def fetch_page(code6, page, size=50, timeout=20):
    q = {"page_size": size, "page_index": page, "ann_type": "A",
         "client_source": "web", "stock_list": code6, "f_node": 0, "s_node": 0}
    req = urllib.request.Request(BASE + "?" + urllib.parse.urlencode(q), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_page(j):
    rows = []
    for it in ((j or {}).get("data") or {}).get("list") or []:
        cols = it.get("columns") or []
        art = it.get("art_code", "")
        code6 = ""
        for c in it.get("codes") or []:
            code6 = str(c.get("stock_code", ""))[:6] or code6
        rows.append({
            "date": (it.get("notice_date") or "")[:10],
            "title": it.get("title", ""),
            "url": (f"https://data.eastmoney.com/notices/detail/{code6}/{art}.html"
                    if art else ""),
            "type": cols[0].get("column_name") if cols else "",
        })
    return rows


def update_one(code6, since, max_pages, dry_run, retries=3):
    out = ANN_DIR / f"{code6}.parquet"
    old = None
    if out.exists():
        try:
            old = pd.read_parquet(out)
            old["date"] = pd.to_datetime(old["date"], errors="coerce")
        except Exception:
            old = None

    new_rows, stop = [], False
    for page in range(1, max_pages + 1):
        j = None
        for a in range(retries):
            try:
                j = fetch_page(code6, page)
                break
            except Exception:
                if a == retries - 1:
                    return code6, "err", 0
                time.sleep(1.5 * (a + 1))
        rows = parse_page(j)
        if not rows:
            break
        for r in rows:
            if not r["date"]:
                continue
            d = pd.Timestamp(r["date"])
            if since is not None and d <= since:
                stop = True
                continue
            new_rows.append(r)
        if stop:
            break
        time.sleep(0.25)

    if not new_rows:
        return code6, "none", 0
    nd = pd.DataFrame(new_rows)
    nd["date"] = pd.to_datetime(nd["date"])
    if dry_run:
        return code6, "dry", len(nd)

    comb = pd.concat([old, nd], ignore_index=True) if old is not None else nd
    comb = (comb.dropna(subset=["date"])
                .drop_duplicates(subset=["date", "title"], keep="first")
                .sort_values("date").reset_index(drop=True))
    added = len(comb) - (len(old) if old is not None else 0)
    tmp = out.with_suffix(".tmp.parquet")
    comb.to_parquet(tmp, index=False)
    tmp.replace(out)
    return code6, "ok", added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    codes = universe_codes()
    if a.limit:
        codes = codes[:a.limit]

    print(f"更新公告 | {len(codes)} 只 | 每只最多 {a.max_pages} 页 | "
          f"{'DRY-RUN' if a.dry_run else '写入'}")
    t0 = time.time()
    ok = err = none = 0
    total_added = 0
    for i, c in enumerate(codes, 1):
        p = ANN_DIR / f"{c}.parquet"
        since = None
        if p.exists():
            try:
                d = pd.read_parquet(p, columns=["date"])
                since = pd.to_datetime(d["date"], errors="coerce").max()
            except Exception:
                pass
        code6, st, n = update_one(c, since, a.max_pages, a.dry_run)
        if st in ("ok", "dry"):
            ok += 1
            total_added += n
        elif st == "none":
            none += 1
        else:
            err += 1
            if err <= 5:
                print(f"  [!] {code6} 失败")
        if i % 25 == 0 or i == len(codes):
            el = time.time() - t0
            eta = (len(codes) - i) / (i / el) if el else 0
            print(f"  [{i}/{len(codes)}] 新增{total_added}条 ok={ok} 无更新={none} "
                  f"err={err} | {el:.0f}s | 剩余~{eta/60:.1f}min", flush=True)

    print(f"\n完成 {time.time()-t0:.0f}s | 新增 {total_added} 条公告 | "
          f"ok={ok} 无更新={none} err={err}")


if __name__ == "__main__":
    main()
