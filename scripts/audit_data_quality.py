"""全量数据准确性体检 — 在重建训练数据之前必须通过

检查项:
  A. 历史覆盖   — 全量重拉是否截断了 2021 前的历史 (对比训练数据日期范围)
  B. OHLC 合法性 — high>=max(o,c), low<=min(o,c), 价格>0, 成交量>=0
  C. 复权连续性  — 异常日涨跌幅 (超过所属板块涨跌停限制) = 复权断裂信号
  D. 交易日历    — 与全市场共识交易日对比, 检测缺失交易日
  E. 重复/排序   — 重复日期、未排序
  F. 新鲜度      — 最新交易日分布
用法:
    python scripts/audit_data_quality.py                 # 全市场
    python scripts/audit_data_quality.py --scope universe
    python scripts/audit_data_quality.py --limit 300
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE_DIR = ROOT / "data" / "raw" / "kline"
UNIVERSE = ROOT / "data" / "universe" / "watchlist_216.json"
TRAIN = ROOT / "data" / "processed" / "training_data_v24.parquet"


def limit_pct(code):
    """所属板块单日涨跌幅限制 (留 1pct 缓冲, 不含 ST)"""
    c = str(code)[:6]
    if c.startswith(("30", "68")):      # 创业板/科创板 20%
        return 0.21
    if c.startswith(("8", "4", "9")):   # 北交所 30%
        return 0.31
    return 0.11                          # 主板 10%


def progress(i, n, t0, extra=""):
    el = time.time() - t0
    rate = i / el if el else 0
    eta = (n - i) / rate if rate else 0
    bar_n = int(30 * i / n)
    bar = "#" * bar_n + "-" * (30 - bar_n)
    print(f"\r  [{bar}] {i}/{n} {100*i/n:5.1f}% | {el:5.0f}s | "
          f"{rate:4.1f}/s | ETA {eta/60:4.1f}m {extra}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "universe"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.scope == "universe":
        w = json.loads(UNIVERSE.read_text())
        items = w.get("watchlist", w) if isinstance(w, dict) else w
        codes = [str(x["code"])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]
        files = [KLINE_DIR / f"{c}.parquet" for c in codes]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(KLINE_DIR.glob("*.parquet"))
    if a.limit:
        files = files[:a.limit]

    n = len(files)
    print(f"=== K线数据体检 | {n} 只 | scope={a.scope} ===\n")

    rows = []
    t0 = time.time()
    date_counter = Counter()
    for i, f in enumerate(files, 1):
        code = f.stem
        r = {"code": code}
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            r["read_err"] = type(e).__name__
            rows.append(r)
            continue

        need = {"date", "open", "high", "low", "close", "volume"}
        if not need.issubset(d.columns):
            r["schema_err"] = ",".join(sorted(need - set(d.columns)))
            rows.append(r)
            continue

        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        r["rows"] = len(d)
        if not len(d):
            rows.append(r)
            continue

        r["dmin"], r["dmax"] = d["date"].min(), d["date"].max()
        r["dup_dates"] = int(d["date"].duplicated().sum())
        r["unsorted"] = int(not d["date"].is_monotonic_increasing)

        d = d.sort_values("date")
        o, h, l, c = (d[x].astype(float) for x in ("open", "high", "low", "close"))
        v = d["volume"].astype(float)
        r["bad_hl"] = int(((h < np.maximum(o, c) - 1e-6) | (l > np.minimum(o, c) + 1e-6)).sum())
        r["nonpos_px"] = int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum())
        r["neg_vol"] = int((v < 0).sum())
        r["nan_px"] = int(c.isna().sum())

        # C. 复权连续性: 剔除停牌复牌(成交量为0的次日)后的异常跳空
        ret = c.pct_change()
        lim = limit_pct(code)
        suspended = (v.shift(1) == 0) | (v == 0)
        jump = (ret.abs() > lim) & (~suspended)
        r["jumps"] = int(jump.sum())
        if r["jumps"]:
            k = ret[jump].abs().idxmax()
            r["worst_jump"] = float(ret.loc[k])
            r["worst_jump_date"] = d.loc[k, "date"]

        # 零成交量天数 (长期停牌)
        r["zero_vol_days"] = int((v == 0).sum())
        date_counter.update(d["date"].tolist())
        rows.append(r)

        if i % 100 == 0 or i == n:
            progress(i, n, t0)
    print("\n")

    df = pd.DataFrame(rows)

    # ── 报告 ─────────────────────────────────────────────
    print("=" * 64)
    print("【结构性错误】")
    for col, label in [("read_err", "读取失败"), ("schema_err", "缺列")]:
        if col in df.columns:
            bad = df[df[col].notna()]
            print(f"  {label:8s}: {len(bad)} 只" + (f"  例: {bad['code'].head(5).tolist()}" if len(bad) else ""))
        else:
            print(f"  {label:8s}: 0 只")

    ok = df[df.get("rows", pd.Series(dtype=float)).notna()].copy()
    print(f"\n  正常读取: {len(ok)}/{n} 只, 共 {int(ok['rows'].sum()):,} 行")

    print("\n【A. 历史覆盖】")
    print(f"  最早日期: 全局 min={ok['dmin'].min().date()}  "
          f"中位={ok['dmin'].median().date()}  max={ok['dmin'].max().date()}")
    pre2021 = (ok["dmin"] < pd.Timestamp("2021-01-01")).sum()
    print(f"  含 2021 前历史的股票: {pre2021} 只 / {len(ok)}")
    if TRAIN.exists():
        t = pd.read_parquet(TRAIN, columns=["date"])
        tmin, tmax = pd.to_datetime(t["date"]).min(), pd.to_datetime(t["date"]).max()
        print(f"  训练数据 v24 日期范围: {tmin.date()} ~ {tmax.date()}")
        if tmin < ok["dmin"].median():
            print(f"  [!!] 训练数据起点早于 K线中位起点 -> 全量重拉可能截断了历史")
        else:
            print(f"  [OK] K线覆盖训练数据起点")

    print("\n【B. OHLC 合法性】")
    for col, label in [("bad_hl", "high/low 不自洽"), ("nonpos_px", "非正价格"),
                       ("neg_vol", "负成交量"), ("nan_px", "收盘价NaN")]:
        s = ok[col].fillna(0)
        print(f"  {label:16s}: {int(s.sum()):>6} 条, 涉及 {int((s>0).sum()):>4} 只")

    print("\n【C. 复权连续性 (超涨跌停限制的跳空)】")
    j = ok["jumps"].fillna(0)
    print(f"  异常跳空: {int(j.sum()):,} 条, 涉及 {int((j>0).sum())} 只 / {len(ok)} "
          f"({100*(j>0).mean():.1f}%)")
    print(f"  每只均值 {j.mean():.2f} 条, 中位 {j.median():.0f} 条, 最多 {int(j.max())} 条")
    worst = ok.nlargest(8, "jumps")[["code", "jumps", "worst_jump", "worst_jump_date", "rows"]]
    print("\n  跳空最多的 8 只:")
    for _, x in worst.iterrows():
        wd = x["worst_jump_date"]
        print(f"    {x['code']}  {int(x['jumps']):>4} 条  最大 {x['worst_jump']*100:+7.1f}%  "
              f"@ {wd.date() if pd.notna(wd) else '-'}  (共{int(x['rows'])}行)")

    print("\n【D. 交易日历】")
    cal = pd.Series(date_counter).sort_index()
    top = cal.max()
    real_days = cal[cal > top * 0.5]
    print(f"  共识交易日: {len(real_days)} 天 ({real_days.index.min().date()} ~ {real_days.index.max().date()})")
    thin = cal[(cal <= top * 0.5) & (cal > 0)]
    print(f"  稀疏日期(可能是脏数据): {len(thin)} 天" +
          (f"  例: {[str(x.date()) for x in thin.index[:5]]}" if len(thin) else ""))

    print("\n【E. 重复/排序】")
    print(f"  重复日期: {int(ok['dup_dates'].fillna(0).sum())} 条, "
          f"涉及 {int((ok['dup_dates'].fillna(0)>0).sum())} 只")
    print(f"  未排序:   {int(ok['unsorted'].fillna(0).sum())} 只")

    print("\n【F. 新鲜度】")
    vc = ok["dmax"].value_counts().sort_index(ascending=False).head(6)
    for dt, cnt in vc.items():
        print(f"  {dt.date()}: {cnt} 只")
    latest = ok["dmax"].max()
    stale = (ok["dmax"] < latest - pd.Timedelta(days=10)).sum()
    print(f"  落后最新交易日 10 天以上: {stale} 只 ({100*stale/len(ok):.1f}%)")

    out = ROOT / "data" / "processed" / "audit_kline_quality.csv"
    df.to_csv(out, index=False)
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
