"""独立源交叉验证: 本地K线(新浪qfq) vs 腾讯qfq

新浪是本次更新的唯一来源, 必须用独立源验证才能确认准确性。
腾讯接口与新浪完全独立(不同厂商/不同复权算法实现)。

比对: 最近 N 个交易日的 收盘价/开盘价 相对偏差
用法: python scripts/crossval_kline_tencent.py --n 60 --days 40
"""
import argparse
import json
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINE = ROOT / "data" / "raw" / "kline"
UNIVERSE = ROOT / "data" / "universe" / "watchlist_216.json"


def tx_symbol(code):
    c = str(code)[:6]
    if c.startswith("6"):
        return "sh" + c
    if c.startswith(("4", "8", "9")):
        return "bj" + c
    return "sz" + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="抽样股票数")
    ap.add_argument("--days", type=int, default=40, help="比对最近N个交易日")
    ap.add_argument("--scope", choices=["all", "universe"], default="universe")
    a = ap.parse_args()

    import akshare as ak

    if a.scope == "universe":
        w = json.loads(UNIVERSE.read_text())
        items = w.get("watchlist", w) if isinstance(w, dict) else w
        pool = [str(x["code"])[:6] if isinstance(x, dict) else str(x)[:6] for x in items]
        pool = [c for c in pool if (KLINE / f"{c}.parquet").exists()]
    else:
        pool = [p.stem for p in KLINE.glob("*.parquet")]
    random.seed(42)
    codes = random.sample(pool, min(a.n, len(pool)))

    print(f"=== 交叉验证 本地(新浪qfq) vs 腾讯qfq ===")
    print(f"抽样 {len(codes)} 只 | 比对最近 {a.days} 个交易日\n")

    recs, fails = [], []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        try:
            loc = pd.read_parquet(KLINE / f"{code}.parquet")
            loc["date"] = pd.to_datetime(loc["date"])
            loc = loc.sort_values("date").tail(a.days + 10)

            tx = ak.stock_zh_a_hist_tx(symbol=tx_symbol(code), adjust="qfq")
            tx["date"] = pd.to_datetime(tx["date"])
            tx = tx.rename(columns={"close": "tx_close", "open": "tx_open"})

            m = loc.merge(tx[["date", "tx_close", "tx_open"]], on="date", how="inner").tail(a.days)
            if len(m) < 5:
                fails.append((code, f"重叠仅{len(m)}天"))
                continue
            dc = (pd.to_numeric(m["close"]) / pd.to_numeric(m["tx_close"]) - 1).abs()
            do = (pd.to_numeric(m["open"]) / pd.to_numeric(m["tx_open"]) - 1).abs()
            recs.append(dict(code=code, days=len(m),
                             close_max=dc.max() * 100, close_mean=dc.mean() * 100,
                             open_max=do.max() * 100,
                             loc_last=float(m["close"].iloc[-1]),
                             tx_last=float(m["tx_close"].iloc[-1]),
                             last_date=m["date"].iloc[-1].date()))
        except Exception as e:
            fails.append((code, f"{type(e).__name__}"))

        el = time.time() - t0
        rate = i / el if el else 0
        bar_n = int(30 * i / len(codes))
        print(f"\r  [{'#'*bar_n}{'-'*(30-bar_n)}] {i}/{len(codes)} {100*i/len(codes):5.1f}% | "
              f"{el:4.0f}s | ok={len(recs)} fail={len(fails)} | "
              f"ETA {(len(codes)-i)/rate/60 if rate else 0:4.1f}m", end="", flush=True)
        time.sleep(0.25)
    print("\n")

    if not recs:
        print("无有效比对结果")
        print(f"失败: {fails[:10]}")
        return

    r = pd.DataFrame(recs)
    print("=" * 62)
    print(f"有效比对 {len(r)} 只 (失败 {len(fails)} 只)\n")
    print(f"  收盘价偏差:  最大 {r['close_max'].max():.4f}%   "
          f"均值 {r['close_mean'].mean():.4f}%   中位 {r['close_mean'].median():.4f}%")
    print(f"  开盘价偏差:  最大 {r['open_max'].max():.4f}%")

    for th, lbl in [(0.01, "<0.01%"), (0.1, "<0.1%"), (1.0, "<1%")]:
        print(f"  收盘价最大偏差 {lbl:7s} 的股票: {(r['close_max'] < th).sum()}/{len(r)}")

    bad = r[r["close_max"] > 1.0].sort_values("close_max", ascending=False)
    if len(bad):
        print(f"\n  [!] 偏差 >1% 的 {len(bad)} 只:")
        for _, x in bad.head(10).iterrows():
            print(f"      {x['code']}  最大偏差 {x['close_max']:.2f}%  "
                  f"本地末值 {x['loc_last']:.2f} vs 腾讯 {x['tx_last']:.2f} @ {x['last_date']}")
    else:
        print("\n  [OK] 无偏差 >1% 的股票 — 两个独立源完全一致")

    if fails:
        print(f"\n  失败样例: {fails[:6]}")

    r.to_csv(ROOT / "data" / "processed" / "crossval_tencent.csv", index=False)
    print(f"\n明细: data/processed/crossval_tencent.csv")


if __name__ == "__main__":
    main()
