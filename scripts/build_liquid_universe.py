# -*- coding: utf-8 -*-
"""按流动性从全市场 K 线里挑出前 N 只, 生成可用于 feature_engine 的 watchlist

筛选条件 (按顺序):
  1. 有 K 线文件, 且回测起点前已上市满 `--min-listed-days` 天
  2. 剔除退市早于回测起点的 (退市在回测期内的保留, 避免幸存者偏差)
  3. 剔除北交所 (流动性过低) 和 `--exclude-st` 时的当前 ST
  4. 按回测期内日均成交额中位数降序取前 N 只

注意 (已知偏差, 结果解读时必须考虑):
  - is_st_now 是"当前是否 ST", 不是 PIT 状态, 剔除它有轻微前视
  - 按全期流动性排名选池本身有前视 (用到了回测期内的成交额)
    这会偏向选出后来变活跃的股票; 严格做法是按截止日前的流动性滚动选池

用法:
  python scripts/build_liquid_universe.py --top 1500 --out watchlist_liquid1500.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
KLINE_DIR = os.path.join(DATA_DIR, "raw", "kline")


def _median_amount(args):
    """返回 (code6, 成交额中位数, 该区间内的K线天数)"""
    code6, start = args
    path = os.path.join(KLINE_DIR, f"{code6}.parquet")
    try:
        df = pd.read_parquet(path, columns=["date", "amount"])
    except Exception:
        return code6, 0.0, 0
    df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start)]
    if df.empty:
        return code6, 0.0, 0
    return code6, float(df["amount"].median()), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1500)
    ap.add_argument("--out", default="watchlist_liquid1500.json")
    ap.add_argument("--test-start", default="2022-09-01",
                    help="回测起点, 用于上市时长和流动性统计区间")
    ap.add_argument("--min-listed-days", type=int, default=365,
                    help="回测起点前至少已上市天数")
    ap.add_argument("--min-days", type=int, default=250,
                    help="区间内至少要有的K线天数")
    ap.add_argument("--exclude-st", action="store_true", default=True)
    ap.add_argument("--exclude-bj", action="store_true", default=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()

    ts = pd.Timestamp(a.test_start)
    meta = pd.read_parquet(os.path.join(DATA_DIR, "universe", "pit_metadata.parquet"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    n0 = len(meta)

    # 1. 上市满 min_listed_days
    meta = meta[meta["list_date"].notna()]
    meta = meta[meta["list_date"] <= ts - pd.Timedelta(days=a.min_listed_days)]
    n1 = len(meta)

    # 2. 退市早于回测起点的剔除; 回测期内退市的保留
    meta = meta[meta["delist_date"].isna() | (meta["delist_date"] >= ts)]
    n2 = len(meta)

    # 3. 板块 / ST
    if a.exclude_bj:
        meta = meta[~meta["board"].astype(str).str.contains("北交", na=False)]
    n3 = len(meta)
    if a.exclude_st:
        meta = meta[~meta["is_st_now"].fillna(False)]
    n4 = len(meta)

    # 4. 必须有 K 线文件
    have = {f[:6] for f in os.listdir(KLINE_DIR) if f.endswith(".parquet")}
    meta = meta[meta["code"].isin(have)]
    n5 = len(meta)

    print(f"全市场 {n0} -> 上市满{a.min_listed_days}天 {n1} -> 未提前退市 {n2} "
          f"-> 非北交所 {n3} -> 非ST {n4} -> 有K线 {n5}")

    # 5. 并行统计流动性
    tasks = [(c, a.test_start) for c in meta["code"]]
    rows = []
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        futs = [ex.submit(_median_amount, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 500 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] {i / len(futs) * 100:.0f}% 统计流动性")

    liq = pd.DataFrame(rows, columns=["code", "med_amount", "n_days"])
    liq = liq[liq["n_days"] >= a.min_days]
    print(f"  区间K线>={a.min_days}天: {len(liq)} 只")

    liq = liq.sort_values("med_amount", ascending=False).head(a.top)
    sel = meta.merge(liq, on="code")
    sel = sel.sort_values("med_amount", ascending=False)

    # 6. 写 watchlist (feature_engine 需要 code 带后缀 + name)
    def suffix(c):
        return f"{c}.SH" if c[0] == "6" else (f"{c}.BJ" if c[0] == "8" or c[0] == "4" else f"{c}.SZ")

    items = [{"code": suffix(r["code"]), "name": r["name"], "board": r["board"]}
             for _, r in sel.iterrows()]
    out_path = os.path.join(DATA_DIR, "universe", a.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"watchlist": items}, f, ensure_ascii=False, indent=1)

    print(f"\n已写出 {len(items)} 只 -> {out_path}")
    print(f"  日均成交额中位数: 最高 ¥{sel['med_amount'].iloc[0] / 1e8:.2f}亿  "
          f"最低 ¥{sel['med_amount'].iloc[-1] / 1e8:.4f}亿")
    print(f"  板块分布: {sel['board'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
