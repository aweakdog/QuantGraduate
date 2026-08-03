"""资金流: 新浪财经源 — 历史回拉 + 每日增量 (可直接跑在服务器上)

为什么是新浪 (2026-08-02 实测, 见 docs/progress_2026-08-02.md):
  - 东财: 服务器 IP 直接拿空响应, Mac 上也要跟指纹反爬+突发频控缠斗
  - 同花顺数据中心: JS 反爬 (chameleon)
  - 腾讯 ff_ 接口: 已下线
  - 新浪 MoneyFlow API: 服务器直连可用, 历史到 2016+, 沪深全覆盖 ✓

字段映射 (新浪 ssl_qsfx_zjlrqs → consolidated 5列 schema):
  netamount   (元, 主力净流入)      → main_force_net
  ratioamount (小数, 主力净流入率)  → main_force_pct (×100 转成 %, 与旧源一致)
  dde_net / mtss_balance / fund_flow 为旧源(thsdk)独有 → 新增行留 NaN
  (特征层 fillna(0); mtss 两融数据将来可从交易所官方源单独补)

单位对账: 合并前用新旧都有的股票在重叠日期上对 net 与 pct 的中位比值,
超出 [0.2, 5] 视为单位错误拒绝混入 (口径差 ~±10% 属供应商方法论差异, 放行)。

用法:
  python -m pipeline.pull_fundflow_sina                  # 回拉池内缺失股票全历史
  python -m pipeline.pull_fundflow_sina --refill-all     # 全池 519 只重灌 net/pct 两列
  python -m pipeline.pull_fundflow_sina --incremental    # 每日增量: 所有股票补最近缺的日期
  python -m pipeline.pull_fundflow_sina --dry-run
  python -m pipeline.pull_fundflow_sina --since 2019-01-01 --sleep 0.5

--refill-all 的动机 (2026-08-02, 为 2019 扩容):
  旧源(thsdk)只有 2020-01 起且已断更; 若只补缺失股票, 2019 年横截面上
  377 只有资金流、271 只全 NaN, 按日 z-score 会系统性歪斜。
  重灌后 main_force_net/pct 两列全池单一来源(新浪, 2019-01起);
  dde_net/mtss_balance/fund_flow 三列为旧源独有, 按 (code,date) 保留 ——
  单一供应商原则按【列】执行, 每列内部口径自洽。
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings  # noqa: E402

DATA = settings.DATA_DIR
FF_DIR = DATA / "raw" / "fund_flow_full"
CONS_PATH = FF_DIR / "fundflow_history.parquet"
WATCHLIST = DATA / "universe" / "watchlist_pit.json"

API = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
       "MoneyFlow.ssl_qsfx_zjlrqs")
PAGE_SIZE = 500         # 实测 num=500 可用 (2026-08-02): 2019至今只需 ~4 页/股
                        # 页大 → 请求少 7 倍, 频控触发概率同比下降


def to_daima(code6: str) -> str:
    return f"sh{code6}" if code6.startswith(("60", "68")) else f"sz{code6}"


def http_get_text(url: str) -> str:
    """curl 子进程: 与东财一课 (requests 的 TLS 指纹容易被单独对待), 统一走 curl"""
    r = subprocess.run(
        ["curl", "-s", "-m", "60",   # 被限速时传输会变慢, 宽容慢传输而非杀掉 (rc=28 教训)
         "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
         "-H", "Referer: https://finance.sina.com.cn/",
         url],
        capture_output=True, text=True, timeout=70)
    if r.returncode != 0 or not r.stdout.strip():
        raise ConnectionError(f"curl rc={r.returncode}, empty={not r.stdout.strip()}")
    return r.stdout


def fetch_page(code6: str, page: int) -> list[dict]:
    url = f"{API}?page={page}&num={PAGE_SIZE}&sort=opendate&asc=0&daima={to_daima(code6)}"
    txt = http_get_text(url)
    if txt.strip().lower() in ("null", "[]"):
        return []
    return json.loads(txt)


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "date": r["opendate"],
        "main_force_net": float(r["netamount"]) if r.get("netamount") not in (None, "") else None,
        # 新浪是小数 (−0.16 = −16%), 旧源与东财都是百分数 → ×100 对齐
        "main_force_pct": float(r["ratioamount"]) * 100 if r.get("ratioamount") not in (None, "") else None,
        "super_large_net": float(r["r0_net"]) if r.get("r0_net") not in (None, "") else None,
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def pull_history(code6: str, since: pd.Timestamp, sleep: float) -> pd.DataFrame:
    """逐页往回翻直到 since 或翻尽。

    每页先自行重试 3 次 (间隔 20/60s, 同页续拉); 耗尽才返 None 让调用方
    全局静默。新浪的频控是渐进限速(越拉越慢直到超时), 短休后同页重试
    通常就能过, 比整只股票从头重拉便宜得多。
    """
    out = []
    for page in range(1, 13):                       # 12 页 × 500 行 = 24 年, 足够
        rows = None
        for attempt in range(3):
            try:
                rows = fetch_page(code6, page)
                break
            except (ConnectionError, json.JSONDecodeError) as e:
                if attempt < 2:
                    print(f"  慢/超时 {code6} p{page} (第{attempt + 1}次): {e}", flush=True)
                    time.sleep((20, 60)[attempt])
                else:
                    print(f"  撞墙 {code6} p{page}: {e}", flush=True)
                    return None
        if not rows:
            break
        df = rows_to_df(rows)
        out.append(df)
        if df["date"].min() < since:
            break
        time.sleep(sleep * random.uniform(0.5, 1.5))
    if not out:
        return pd.DataFrame()
    full = pd.concat(out, ignore_index=True).drop_duplicates("date")
    return full[full["date"] >= since].sort_values("date").reset_index(drop=True)


def universe_codes() -> list[str]:
    wl = json.loads(WATCHLIST.read_text())
    items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
    return sorted({str(it["code"] if isinstance(it, dict) else it)[:6] for it in items})


def unit_check(cons: pd.DataFrame, overlap_codes: list[str]) -> None:
    """新浪 vs 旧源(thsdk) 中位比值; 只拦数量级错误"""
    net_r, pct_r = [], []
    for c in overlap_codes[:5]:
        df = pull_history(c, pd.Timestamp("2025-01-01"), 0.3)
        if df is None or df.empty:
            continue
        old = cons[cons["code"] == c][["date", "main_force_net", "main_force_pct"]].dropna()
        m = old.merge(df, on="date", suffixes=("_old", "_new")).dropna()
        m = m[m["main_force_net_new"].abs() > 1e4]
        if len(m) >= 30:
            net_r.append((m["main_force_net_old"] / m["main_force_net_new"]).median())
            mp = m[m["main_force_pct_new"].abs() > 0.5]
            if len(mp) >= 30:
                pct_r.append((mp["main_force_pct_old"] / mp["main_force_pct_new"]).median())
    if not net_r:
        raise SystemExit("ERROR: 无重叠样本可对账, 拒绝盲合")
    rn = pd.Series(net_r).median()
    rp = pd.Series(pct_r).median() if pct_r else float("nan")
    print(f"单位对账(新浪 vs 旧源): net {rn:.4f} | pct {rp:.4f} ({len(net_r)} 只)", flush=True)
    for name, r in (("net", rn), ("pct", rp)):
        if r == r and not 0.2 <= r <= 5.0:
            raise SystemExit(f"ERROR: {name} 中位比值 {r:.4f} 超出 [0.2,5], 疑似单位错, 拒绝混入")


def merge_into_consolidated(cons: pd.DataFrame, new_rows: list[pd.DataFrame]) -> None:
    add = pd.concat(new_rows, ignore_index=True)
    add = add[add["date"] >= cons["date"].min()]
    # 防重: 同 (code,date) 已有的行不再加 (增量模式跑多次也安全)
    key = cons.set_index(["code", "date"]).index
    add = add[~add.set_index(["code", "date"]).index.isin(key)]
    if add.empty:
        print("无新增行, consolidated 不变", flush=True)
        return
    bak = CONS_PATH.with_name(
        f"fundflow_history.bak_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet")
    CONS_PATH.rename(bak)
    out = pd.concat([cons, add], ignore_index=True).sort_values(["code", "date"])
    out.to_parquet(CONS_PATH, index=False)
    print(f"完成: {len(cons):,} -> {len(out):,} 行 (+{len(add):,}), "
          f"覆盖 {out['code'].nunique()} 只 (旧表备份 {bak.name})", flush=True)


def refill_consolidated(cons: pd.DataFrame, new_rows: list[pd.DataFrame]) -> None:
    """整列替换 net/pct: 新浪为准, 新浪缺的 (code,date) 保留旧值; 其余三列全保留"""
    sina = pd.concat(new_rows, ignore_index=True)[
        ["date", "code", "main_force_net", "main_force_pct"]]
    old_extra = cons[["date", "code", "main_force_net", "main_force_pct",
                      "dde_net", "mtss_balance", "fund_flow"]].rename(
        columns={"main_force_net": "net_old", "main_force_pct": "pct_old"})
    out = sina.merge(old_extra, on=["code", "date"], how="outer")
    out["main_force_net"] = out["main_force_net"].fillna(out["net_old"])
    out["main_force_pct"] = out["main_force_pct"].fillna(out["pct_old"])
    out = out.drop(columns=["net_old", "pct_old"]).sort_values(["code", "date"])
    n_sina = sina["code"].nunique()
    bak = CONS_PATH.with_name(
        f"fundflow_history.bak_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet")
    CONS_PATH.rename(bak)
    out.to_parquet(CONS_PATH, index=False)
    print(f"重灌完成: {len(cons):,} -> {len(out):,} 行, net/pct 来自新浪 {n_sina} 只 "
          f"(旧表备份 {bak.name})", flush=True)


def cons_schema_row(code6: str, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "date": df["date"], "code": code6,
        "main_force_net": df["main_force_net"],
        "main_force_pct": df["main_force_pct"],
        "dde_net": pd.NA, "mtss_balance": pd.NA, "fund_flow": pd.NA,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--incremental", action="store_true",
                    help="每日增量: 全部池内股票只补比 consolidated 更新的日期 (每只 1 页)")
    ap.add_argument("--refill-all", action="store_true",
                    help="全池重灌: 所有池内股票拉新浪全历史, 整列替换 net/pct")
    ap.add_argument("--since", default="2019-01-01", help="历史回拉起点")
    ap.add_argument("--sleep", type=float, default=0.5, help="页间隔秒 (±50%% 抖动)")
    ap.add_argument("--skip-unit-check", action="store_true")
    a = ap.parse_args()

    cons = pd.read_parquet(CONS_PATH)
    cons["date"] = pd.to_datetime(cons["date"])
    covered = set(cons["code"].astype(str).str.zfill(6))
    codes = universe_codes()
    since = pd.Timestamp(a.since)

    if a.incremental:
        # ── 每日增量: 所有池内股票, 从各自最新日期往后补 ──
        latest = cons.groupby(cons["code"].astype(str).str.zfill(6))["date"].max()
        new_rows, fails = [], 0
        for i, c in enumerate(codes, 1):
            cutoff = latest.get(c, since - pd.Timedelta(days=1))
            try:
                rows = fetch_page(c, 1)          # 1 页 = 最近 60 交易日, 增量足够
            except Exception as e:
                fails += 1
                if fails <= 3:
                    print(f"  失败 {c}: {e}", flush=True)
                continue
            if rows:
                df = rows_to_df(rows)
                df = df[df["date"] > cutoff]
                if len(df):
                    new_rows.append(cons_schema_row(c, df))
            time.sleep(a.sleep * random.uniform(0.5, 1.5))
            if i % 100 == 0:
                print(f"  [{i}/{len(codes)}] 增量中, 失败 {fails}", flush=True)
        print(f"增量扫描完: {len(new_rows)} 只有新数据, 失败 {fails} 只", flush=True)
        if a.dry_run:
            return
        if new_rows:
            merge_into_consolidated(cons, new_rows)
        return

    # ── 历史回拉: --refill-all 拉全池, 否则只拉 consolidated 没有的股票 ──
    missing = codes if a.refill_all else [c for c in codes if c not in covered]
    print(f"PIT 池 {len(codes)} 只 | 已覆盖 {len(covered)} | 本次拉 {len(missing)}"
          f"{' (refill-all)' if a.refill_all else ''}", flush=True)
    if a.dry_run:
        print("dry-run:", missing[:20], "...")
        return
    if not a.skip_unit_check:
        unit_check(cons, [c for c in codes if c in covered])

    FF_DIR.mkdir(parents=True, exist_ok=True)
    new_rows, cooldowns, i = [], 0, 0
    while i < len(missing):
        c = missing[i]
        raw_path = FF_DIR / f"sina_{c}.parquet"
        stale_raw = raw_path.exists() and pd.read_parquet(
            raw_path, columns=["date"])["date"].min() > since + pd.Timedelta(days=7)
        if stale_raw:                             # 旧 raw 起点比本次 since 晚 → 重拉
            raw_path.unlink()
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            df["date"] = pd.to_datetime(df["date"])
        else:
            df = pull_history(c, since, a.sleep)
            if df is None:                        # 撞墙 → 全局静默, 同一只重来
                cooldowns += 1
                wait = min(300 * cooldowns, 1800)
                print(f"  全局静默 {wait / 60:.0f} 分钟 (第 {cooldowns} 次, "
                      f"停在 {i + 1}/{len(missing)} {c})", flush=True)
                time.sleep(wait)
                continue
            cooldowns = 0
            if df.empty:                          # 真没数据 (新上市等), 跳过
                print(f"  无数据 {c} (跳过)", flush=True)
                i += 1
                continue
            df.to_parquet(raw_path, index=False)
        new_rows.append(cons_schema_row(c, df))
        i += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(missing)}]", flush=True)

    if not new_rows:
        return
    if a.refill_all:
        refill_consolidated(cons, new_rows)
    else:
        merge_into_consolidated(cons, new_rows)


if __name__ == "__main__":
    main()
