"""用 akshare 批量补全个股历史基本面 (Mac 可用, 替代 Windows 的 iFinD)

数据源: 东财 stock_financial_abstract —— 每只股票返回全部历史报告期(约 100+ 期)的
        关键财务指标, 是少数几个能拿到完整历史的免费接口。

产出 data/raw/fundamentals/{code6}.parquet, schema 与既有 iFinD 文件保持一致:
  date(报告期截止日, str) | code | revenue | profit | eps | bps | roe | pe | pb
  | mcap | total_assets | debt_ratio | gross_margin | operate_cf

说明:
  - pe/pb/mcap 需要股价, 本接口不提供, 留空 (feature_engine 允许缺列)
  - total_assets 由 净资产 / (1 - 资产负债率) 推算
  - 报告期 -> 发布日的偏移由 feature_engine._fund_pub_date 统一处理, 本脚本不做

用法:
  python scripts/backfill_fundamentals_ak.py --watchlist watchlist_pit.json
  python scripts/backfill_fundamentals_ak.py --codes 000001,600519 --force
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/raw/fundamentals"

# akshare 指标名 -> 目标列名 (只取 '常用指标' 分组, 避免同名指标歧义)
IND_MAP = {
    "营业总收入": "revenue",
    "归母净利润": "profit",
    "基本每股收益": "eps",
    "每股净资产": "bps",
    "净资产收益率(ROE)": "roe",
    "资产负债率": "debt_ratio",
    "毛利率": "gross_margin",
    "经营现金流量净额": "operate_cf",
    "股东权益合计(净资产)": "_equity",
}
OUT_COLS = ["date", "code", "revenue", "profit", "eps", "bps", "roe", "pe", "pb",
            "mcap", "total_assets", "debt_ratio", "gross_margin", "operate_cf"]


def bar(i, n, t0, tag=""):
    pct = i / n
    filled = int(pct * 30)
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {i}/{n} ({pct*100:5.1f}%) "
          f"用时 {el/60:.1f}min ETA {eta/60:.1f}min  {tag[:20]:<20}", end="", flush=True)


def fetch_one(ak, code: str, min_year: int) -> pd.DataFrame | None:
    """抓单只股票全历史报告期, 转为长表"""
    d = ak.stock_financial_abstract(symbol=code)
    if d is None or not len(d):
        return None
    d = d[d["选项"] == "常用指标"]
    d = d[d["指标"].isin(IND_MAP)]
    if not len(d):
        return None

    period_cols = [c for c in d.columns if str(c).isdigit() and len(str(c)) == 8]
    recs = {}
    for _, row in d.iterrows():
        col = IND_MAP[row["指标"]]
        for p in period_cols:
            if int(p[:4]) < min_year:
                continue
            v = pd.to_numeric(row[p], errors="coerce")
            if pd.isna(v):
                continue
            recs.setdefault(p, {})[col] = float(v)
    if not recs:
        return None

    out = pd.DataFrame([{"period": p, **vals} for p, vals in recs.items()])
    out["date"] = pd.to_datetime(out["period"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    out["code"] = code

    # 总资产 = 净资产 / (1 - 资产负债率)  (资产负债率为百分数)
    if "_equity" in out.columns and "debt_ratio" in out.columns:
        ratio = 1 - out["debt_ratio"] / 100.0
        out["total_assets"] = (out["_equity"] / ratio.where(ratio > 0)).round(2)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = None
    return out.sort_values("date", ascending=False)[OUT_COLS].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default="watchlist_pit.json")
    ap.add_argument("--codes", default=None, help="逗号分隔的6位代码, 覆盖 watchlist")
    ap.add_argument("--min-year", type=int, default=2018)
    ap.add_argument("--force", action="store_true", help="已有文件也重抓")
    ap.add_argument("--sleep", type=float, default=0.4, help="每次请求间隔(秒)")
    ap.add_argument("--retry", type=int, default=3)
    a = ap.parse_args()

    import akshare as ak

    if a.codes:
        codes = [c.strip().zfill(6) for c in a.codes.split(",")]
    else:
        w = json.loads((ROOT / "data/universe" / a.watchlist).read_text(encoding="utf-8"))
        codes = [s["code"][:6] for s in w["watchlist"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if a.force or not (OUT_DIR / f"{c}.parquet").exists()]
    print(f"股票池 {len(codes)} 只 | 待抓 {len(todo)} 只 (已有 {len(codes)-len(todo)} 只)")

    ok = fail = 0
    failed = []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        bar(i, len(todo), t0, code)
        df = None
        for attempt in range(a.retry):
            try:
                df = fetch_one(ak, code, a.min_year)
                break
            except Exception as e:
                if attempt == a.retry - 1:
                    failed.append((code, f"{type(e).__name__}: {e}"))
                else:
                    time.sleep(1.5 * (attempt + 1))
        if df is not None and len(df):
            df.to_parquet(OUT_DIR / f"{code}.parquet", index=False)
            ok += 1
        else:
            fail += 1
        time.sleep(a.sleep)

    print(f"\n完成: 成功 {ok} 失败 {fail} | 用时 {(time.time()-t0)/60:.1f} min")
    for c, e in failed[:10]:
        print(f"  失败 {c}: {e}")

    have = sum((OUT_DIR / f"{c}.parquet").exists() for c in codes)
    print(f"股票池覆盖: {have}/{len(codes)} ({100*have/len(codes):.1f}%)")


if __name__ == "__main__":
    main()
