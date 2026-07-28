"""用交易所官方两融明细补齐 fundflow_history 的 mtss_balance (Mac 可用)

口径已校准 (2026-06-30, 190只):
  深交所: 直接取 '融资融券余额'            中位偏差 0.000000%
  上交所: 融资余额 + 融券余量 x 当日收盘价   中位偏差 0.0000%, 98.9% 在 0.5% 内
          (上交所明细无融券余额金额列, 故用融券余量x收盘价估算;
           本地K线为qfq但以最新日为锚, 近期qfq收盘价==实际收盘价)

东财个股资金流(main_force_net/dde_net)仍被IP封禁, 本脚本只补 mtss_balance。

用法:
  python scripts/update_mtss_exchange.py --dry-run
  python scripts/update_mtss_exchange.py
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FF = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"
KL = ROOT / "data/raw/kline"


def bar(i, n, t0, tag=""):
    el = time.time() - t0
    rate = i / el if el else 0
    k = int(28 * i / n)
    print(f"\r  [{'#'*k}{'-'*(28-k)}] {i}/{n} {100*i/n:5.1f}% | {el:4.0f}s | "
          f"ETA {(n-i)/rate/60 if rate else 0:4.1f}m  {tag:24s}", end="", flush=True)


def trading_days(start, end):
    """从K线取共识交易日"""
    k = pd.read_parquet(KL / "000063.parquet", columns=["date"])
    d = pd.to_datetime(k["date"])
    return sorted(d[(d >= start) & (d <= end)].unique())


def close_map(day, codes):
    """取指定交易日各股收盘价"""
    out = {}
    for c in codes:
        p = KL / f"{c}.parquet"
        if not p.exists():
            continue
        try:
            k = pd.read_parquet(p, columns=["date", "close"])
        except Exception:
            continue
        k["date"] = pd.to_datetime(k["date"])
        r = k[k["date"] == day]
        if len(r):
            out[c] = float(r["close"].iloc[0])
    return out


def fetch_day(ak, day, want_codes):
    """返回 DataFrame[code, mtss_balance]"""
    ds = pd.Timestamp(day).strftime("%Y%m%d")
    parts = []

    # ── 上交所 ──
    try:
        s = ak.stock_margin_detail_sse(date=ds)
    except Exception:
        s = None
    if s is not None and len(s):
        s = s.rename(columns={"标的证券代码": "code", "融资余额": "rz",
                              "融券余量": "rq_vol"})
        s["code"] = s["code"].astype(str).str.zfill(6)
        s = s[s["code"].isin(want_codes)]
        if len(s):
            for c in ("rz", "rq_vol"):
                s[c] = pd.to_numeric(s[c], errors="coerce")
            cm = close_map(day, s["code"].tolist())
            s["close"] = s["code"].map(cm)
            s["mtss_balance"] = s["rz"].fillna(0) + \
                s["rq_vol"].fillna(0) * s["close"].fillna(0)
            parts.append(s[["code", "mtss_balance"]])

    # ── 深交所 ──
    try:
        z = ak.stock_margin_detail_szse(date=ds)
    except Exception:
        z = None
    if z is not None and len(z):
        cmap = {}
        for c in z.columns:
            cs = str(c)
            if "证券代码" in cs:
                cmap[c] = "code"
            elif cs == "融资融券余额":
                cmap[c] = "mtss_balance"
        z = z.rename(columns=cmap)
        if "code" in z.columns and "mtss_balance" in z.columns:
            z["code"] = z["code"].astype(str).str.zfill(6)
            z = z[z["code"].isin(want_codes)]
            if len(z):
                z["mtss_balance"] = pd.to_numeric(z["mtss_balance"], errors="coerce")
                parts.append(z[["code", "mtss_balance"]])

    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True).dropna(subset=["mtss_balance"])
    out = out[out["mtss_balance"] > 0]
    return out if len(out) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--end", default="2026-07-27")
    a = ap.parse_args()
    import akshare as ak

    ff = pd.read_parquet(FF)
    ff["date"] = pd.to_datetime(ff["date"])
    ff["code"] = ff["code"].astype(str).str.zfill(6)
    have = ff[ff["mtss_balance"].notna()]
    mtss_max = have["date"].max()
    print(f"fundflow_history: {len(ff):,} 行, {ff['code'].nunique()} 只, "
          f"至 {ff['date'].max().date()}")
    print(f"其中 mtss_balance 至 {mtss_max.date()} ({have['code'].nunique()} 只)\n")

    want = set(ff["code"].unique())
    days = trading_days(mtss_max + pd.Timedelta(days=1), pd.Timestamp(a.end))
    if not days:
        print("无待补交易日")
        return
    print(f"待补 {len(days)} 个交易日: {pd.Timestamp(days[0]).date()} ~ "
          f"{pd.Timestamp(days[-1]).date()}\n")

    t0 = time.time()
    got = []
    for i, d in enumerate(days, 1):
        r = fetch_day(ak, d, want)
        if r is not None:
            r["date"] = pd.Timestamp(d)
            got.append(r)
            tag = f"{pd.Timestamp(d).date()} {len(r)}只"
        else:
            tag = f"{pd.Timestamp(d).date()} 无数据"
        bar(i, len(days), t0, tag)
        time.sleep(0.5)
    print("\n")

    if not got:
        print("未取到任何数据")
        return
    new = pd.concat(got, ignore_index=True)
    print(f"抓取合计 {len(new):,} 条, 覆盖 {new['code'].nunique()} 只, "
          f"{new['date'].nunique()} 个交易日")
    print(f"  日均 {len(new)/new['date'].nunique():.0f} 只")

    # 完整性闸门: 两融明细按交易所分别发布, 单边缺失会造成截面偏差 -> 整日剔除
    cnt = new.groupby("date").size()
    thr = cnt.median() * 0.8
    drop = cnt[cnt < thr]
    if len(drop):
        print(f"\n  [闸门] 剔除覆盖不足的 {len(drop)} 天 (阈值 {thr:.0f} 只):")
        for d, c in drop.items():
            print(f"    {pd.Timestamp(d).date()}  仅 {c} 只 — 疑似单边交易所未发布")
        new = new[~new["date"].isin(drop.index)]
        print(f"  剩余 {len(new):,} 条 / {new['date'].nunique()} 天")

    if a.dry_run:
        print("\n[DRY-RUN] 未写入")
        print(new.groupby("date").size().to_string())
        return

    # 合并: 已有行填充 mtss_balance, 缺失行新增
    idx = ["date", "code"]
    ff = ff.set_index(idx)
    new = new.set_index(idx)
    both = ff.index.intersection(new.index)
    only = new.index.difference(ff.index)
    ff.loc[both, "mtss_balance"] = new.loc[both, "mtss_balance"]
    if len(only):
        add = pd.DataFrame(index=only, columns=ff.columns, dtype="float64")
        add["mtss_balance"] = new.loc[only, "mtss_balance"]
        ff = pd.concat([ff, add])
    ff = ff.reset_index().sort_values(idx).reset_index(drop=True)

    tmp = FF.with_suffix(".tmp.parquet")
    ff.to_parquet(tmp, index=False)
    tmp.replace(FF)
    print(f"\n已写入 {FF}")
    print(f"  填充已有行 {len(both):,} | 新增行 {len(only):,}")
    print(f"  总行数 {len(ff):,} | mtss_balance 最新 "
          f"{ff[ff['mtss_balance'].notna()]['date'].max().date()}")


if __name__ == "__main__":
    main()
