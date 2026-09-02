# -*- coding: utf-8 -*-
"""宏观/期货表历史回填 (扩窗 2017 用, 一次性)

把各表 prepend 到 2017-01 之前 (源允许的最早), 只在旧表首日**之前**插行:
  - 重叠段一律信旧表, 逐值不改写 (写盘前 equals 自检, 违反即拒写)
  - 每表备份 .bak_bf2017, 出错可整体还原

源与既有日更 (pipeline/pull_macro.py) 的关系: 复用其 fetcher, 不动其 append-only
日更逻辑。期货 4 表用新浪外盘 (futures_foreign_hist), 与 iFinD 遗留表重叠段
999 天对账 corr>0.9998 / 中位比值 1.0000 (2026-08-19 探测)。

已知缺口 (设计内, 不造数):
  - CN2Y: akshare 中债登无 2Y 期限, 2020-01 前保持 NaN
  - 道指/纳指期货 (YM/NQ): 新浪起点 2018-04, 2017 年 NaN

用法: python scripts/backfill_macro_2017.py [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import settings  # noqa: E402
from pipeline.pull_macro import (  # noqa: E402
    _pro, fetch_cn5y, fetch_usdcnh, fetch_usdjpy, fetch_usdind,
)

MACRO = Path(settings.MACRO_DIR)
SINCE = pd.Timestamp("2016-06-01")  # 留出 ma20 等窗口缓冲


def fetch_foreign(sym):
    def _f(_since):
        import akshare as ak
        d = ak.futures_foreign_hist(symbol=sym)
        d = d.rename(columns={"date": "日期", "close": "最新值"})
        d["日期"] = pd.to_datetime(d["日期"])
        d["最新值"] = pd.to_numeric(d["最新值"], errors="coerce")
        return d[["日期", "最新值"]].dropna()
    return _f


def fetch_us_tycr_seg(col):
    """us_tycr 单次返回有行数上限 (全窗口拉会被截到最近 ~2000 行),
    回填只拉 2016-06 ~ 2020-01 小段, 不触顶"""
    def _f(_since):
        d = _pro().us_tycr(start_date="20160601", end_date="20200110")
        return pd.DataFrame({
            "日期": pd.to_datetime(d["date"], errors="coerce"),
            "最新值": pd.to_numeric(d[col], errors="coerce"),
        }).dropna()
    return _f


JOBS = [
    ("标普期货", fetch_foreign("ES")),
    ("道指期货", fetch_foreign("YM")),
    ("纳指期货", fetch_foreign("NQ")),
    ("A50期货", fetch_foreign("CHA50CFD")),
    ("US2Y", fetch_us_tycr_seg("y2")),
    ("US5Y", fetch_us_tycr_seg("y5")),
    ("USDCNH", fetch_usdcnh),
    ("USDJPY", fetch_usdjpy),
    ("USDIND", fetch_usdind),
    ("CN5Y", fetch_cn5y),
]


def backfill_one(name, fetcher, dry):
    path = MACRO / f"{name}.parquet"
    old = pd.read_parquet(path)
    dc = "日期" if "日期" in old.columns else old.columns[0]
    old[dc] = pd.to_datetime(old[dc])
    old_first = old[dc].min()
    try:
        new = fetcher(SINCE.date())
    except Exception as e:
        return f"{name}: 拉取失败 {str(e)[:60]}"
    new["日期"] = pd.to_datetime(new["日期"])
    pre = new[(new["日期"] >= SINCE) & (new["日期"] < old_first)].copy()
    if not len(pre):
        return f"{name}: 无可回填行 (旧表首日 {old_first.date()}, 源最早 {new['日期'].min().date()})"

    # 列对齐: prepend 段只有 [日期, 最新值], 旧表其余列 NaN
    pre = pre.rename(columns={"日期": dc})
    merged = pd.concat([pre, old], ignore_index=True).sort_values(dc).reset_index(drop=True)

    # 自检 1: 行数 = 旧 + prepend
    assert len(merged) == len(old) + len(pre), f"{name} 行数对不上"
    # 自检 2: 旧段逐值不变
    tail = merged[merged[dc] >= old_first].reset_index(drop=True)
    old_sorted = old.sort_values(dc).reset_index(drop=True)
    if not tail[dc].equals(old_sorted[dc]):
        return f"{name}: ✗ 旧段日期序变化, 拒写"
    vcol = "最新值" if "最新值" in old.columns else None
    if vcol and not tail[vcol].fillna(-9e9).equals(old_sorted[vcol].fillna(-9e9)):
        return f"{name}: ✗ 旧段数值变化, 拒写"

    if dry:
        return (f"{name}: [dry] 可回填 {len(pre)} 行 "
                f"({pre[dc].min().date()} ~ {pre[dc].max().date()})")
    shutil.copy2(path, path.with_suffix(".parquet.bak_bf2017"))
    merged.to_parquet(path, index=False)
    return (f"{name}: ✓ prepend {len(pre)} 行 "
            f"({pre[dc].min().date()} ~ {pre[dc].max().date()}), 总 {len(merged)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for name, fetcher in JOBS:
        print(backfill_one(name, fetcher, args.dry_run), flush=True)


if __name__ == "__main__":
    main()
