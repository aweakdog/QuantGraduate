# -*- coding: utf-8 -*-
"""从供应商日K线包(不复权+复权因子)构建 PIT K 线库

为什么换源 (docs/方法论与数据体系.md §3.3 第八条):
  旧链路是新浪 qfq —— 以拉取日为锚整段改写历史, (1) 12 个价格单位列
  带未来分红送转信息(V24PU 实验证明剔掉反而更好), (2) 每天全量重拉不可
  追加, (3) 多源降级口径漂移导致矩阵不可复现。
  新源给不复权 OHLC + 逐日复权因子 + 扣减值: 行写下后永不变 => append-only。

输入: 供应商 zip 解包目录, 每股一个 xlsx (含活跃股与退市股)
输出: data/processed/kline_pit.parquet  (tall 表, code+date 排序)
      data/processed/kline_pit_manifest.json (来源/行数/校验摘要)

内嵌校验: 后复权价 == close*复权因子 + 扣减值 (容差 0.02, 供应商保留2位小数);
          日期严格递增; 违反即该股票记入 manifest 的 bad 名单并拒绝入库。

用法:
    python scripts/build_kline_pit.py --src ~/baidu_probe/kline_full/xxx \
        [--src2 退市股目录] [--out data/processed/kline_pit.parquet]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd

# 供应商列 -> 库内列 (丢弃 BS/缠论/均线/macd/kdj/rsi/boll/涨幅系列: 指标自己算)
KEEP = {
    "date": "date",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "amount": "amount", "volume": "volume",
    "换手率": "turnover_rate",
    "流通股本": "float_shares", "总股本": "total_shares",
    "复权因子": "adj_factor", "扣减值": "adj_deduct", "后复权价": "hfq_close",
    "红利": "div_cash", "送股数": "bonus_shares", "转增股": "conv_shares",
    "配股数": "rights_shares", "配股价": "rights_price",
}
TOL = 0.02  # 后复权价自洽容差(元): 供应商两位小数


def read_engine():
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError:
        return None  # pandas 默认 openpyxl


ENGINE = read_engine()


def parse_one(path):
    code = re.match(r"(\d{6})", os.path.basename(path))
    if not code:
        return None, "无代码前缀: " + os.path.basename(path)
    code = code.group(1)
    try:
        kw = {"engine": ENGINE} if ENGINE else {}
        df = pd.read_excel(path, **kw)
    except Exception as e:  # noqa: BLE001
        return None, f"{code} 读取失败: {e}"
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        return None, f"{code} 缺列: {missing}"
    df = df[list(KEEP)].rename(columns=KEEP)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if not df["date"].is_monotonic_increasing:
        df = df.sort_values("date", kind="stable")
    if df["date"].duplicated().any():
        # 供应商分段计算接缝会把同一天写两行: 数值一致则去重, 冲突才拒绝
        key = ["open", "high", "low", "close", "volume", "adj_factor"]
        conflict = (df.groupby("date")[key].nunique() > 1).any(axis=1)
        if conflict.any():
            return None, f"{code} 日期重复且数值冲突: {list(conflict[conflict].index[:3])}"
        df = df.drop_duplicates(subset="date", keep="last")
    # 自洽: 后复权价 = close*factor + deduct
    chk = (df["close"] * df["adj_factor"] + df["adj_deduct"] - df["hfq_close"]).abs()
    bad = float(chk.max()) if len(chk) else 0.0
    if bad > TOL:
        return None, f"{code} 后复权自洽失败 max_err={bad:.4f}"
    df.insert(0, "code", code)
    return df, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="全包解压目录(活跃股)")
    ap.add_argument("--src2", default=None, help="退市股目录(可选)")
    ap.add_argument("--out", default="data/processed/kline_pit.parquet")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--src-zip", default=None, help="记录来源zip路径(写manifest)")
    args = ap.parse_args()

    files = []
    for d in filter(None, [args.src, args.src2]):
        fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".xlsx")]
        print(f"{d}: {len(fs)} 个 xlsx")
        files += fs
    if not files:
        sys.exit("没有输入文件")

    t0 = time.time()
    parts, errors = [], []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (df, err) in enumerate(ex.map(parse_one, files, chunksize=8)):
            if err:
                errors.append(err)
            elif df is not None:
                parts.append(df)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(files)} 已解析, {time.time()-t0:.0f}s")

    big = pd.concat(parts, ignore_index=True)
    big = big.sort_values(["code", "date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    big.to_parquet(args.out, index=False)

    zip_sha = None
    if args.src_zip and os.path.exists(args.src_zip):
        h = hashlib.sha256()
        with open(args.src_zip, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        zip_sha = h.hexdigest()

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_zip": args.src_zip, "source_zip_sha256": zip_sha,
        "n_files": len(files), "n_stocks_ok": big["code"].nunique(),
        "n_rows": len(big),
        "date_min": str(big["date"].min().date()),
        "date_max": str(big["date"].max().date()),
        "hfq_check_tol": TOL,
        "n_errors": len(errors), "errors": errors[:50],
    }
    mpath = args.out.replace(".parquet", "_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n入库 {manifest['n_stocks_ok']} 只 / {manifest['n_rows']} 行, "
          f"{manifest['date_min']} ~ {manifest['date_max']}")
    print(f"拒绝 {len(errors)} 只; 明细见 {mpath}")
    for e in errors[:10]:
        print("  ", e)
    print(f"耗时 {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
