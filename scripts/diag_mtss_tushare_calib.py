"""对账: tushare margin_detail 能否接续已死的 mtss_balance (thsdk 融资融券余额)

为什么必须先对账
────────────────
mtss_balance 原本来自 thsdk, 该源 2026-06-30 断更, 依赖它的 4 个入选特征
(mtss_z / mtss_z_ma5 / mtss_z_ma20 / mtss_1d_ma20) 就此失去前向数据。
tushare margin_detail 是交易所官方口径, 有 2019 至今全市场历史, 是天然的接续源。

但换源前必须验证口径一致, 否则就是在制造一段带跳变的历史 —— 这个项目在 K 线上
已经踩过一次(sina/tx 三源拼接)。判据沿用 backfill_fundflow_universe.unit_check
的做法: 在重叠 (code, date) 上比中位比值, 偏离 1 超过 5% 就不能直接混用。

同时检查 tushare 侧的两个候选列:
  rzrqye = 融资融券余额 (融资 + 融券), 概念上最接近"融资融券余额"
  rzye   = 融资余额 (仅融资), 若旧源实际只统计融资, 这一列才是对的

用法:
  python scripts/diag_mtss_tushare_calib.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FF = ROOT / "data" / "raw" / "fund_flow_full" / "fundflow_history.parquet"
TS_DIR = ROOT / "data" / "raw" / "tushare" / "margin_detail"


def main():
    ff = pd.read_parquet(FF, columns=["code", "date", "mtss_balance"])
    ff = ff.dropna(subset=["mtss_balance"]).copy()
    ff["code"] = ff["code"].astype(str).str.zfill(6)
    ff["date"] = pd.to_datetime(ff["date"])
    ff["mtss_balance"] = ff["mtss_balance"].astype(float)
    print(f"旧 mtss_balance: {len(ff):,} 行 | {ff['code'].nunique()} 只 | "
          f"{ff['date'].min():%F} ~ {ff['date'].max():%F}")

    files = sorted(glob.glob(str(TS_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"ERROR: 没有 tushare margin_detail 数据于 {TS_DIR}")
    ts = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts["code"] = ts["ts_code"].astype(str).str[:6]
    ts["date"] = pd.to_datetime(ts["trade_date"], format="%Y%m%d")
    print(f"tushare margin_detail: {len(ts):,} 行 | {ts['code'].nunique()} 只 | "
          f"{ts['date'].min():%F} ~ {ts['date'].max():%F}")
    print(f"  可用列: {[c for c in ts.columns if c not in ('ts_code', 'trade_date', 'code', 'date')]}")

    cands = [c for c in ("rzrqye", "rzye") if c in ts.columns]
    m = ff.merge(ts[["code", "date"] + cands], on=["code", "date"], how="inner")
    print(f"\n重叠样本: {len(m):,} 行 | {m['code'].nunique()} 只 | "
          f"{m['date'].min():%F} ~ {m['date'].max():%F}" if len(m) else "\n重叠样本: 0 —— 无法对账")
    if not len(m):
        return

    print("\n全样本对账 (mtss_balance / tushare列):")
    best = None
    for c in cands:
        s = m[c].astype(float).replace(0, pd.NA)
        r = (m["mtss_balance"] / s).dropna()
        corr = m[["mtss_balance", c]].corr().iloc[0, 1]
        print(f"  vs {c:7s} 中位比值={r.median():.6f}  "
              f"[25%={r.quantile(.25):.4f}, 75%={r.quantile(.75):.4f}]  相关={corr:.6f}")
        if best is None or abs(r.median() - 1) < abs(best[1] - 1):
            best = (c, r.median(), corr)

    col, ratio, corr = best
    print(f"\n最接近的是 {col}: 中位比值 {ratio:.6f}, 相关 {corr:.6f}")

    # 逐股中位比值的分布 —— 全样本中位数可能掩盖个股层面的系统偏差
    per = (m.assign(r=m["mtss_balance"] / m[col].astype(float).replace(0, pd.NA))
             .dropna(subset=["r"])
             .groupby("code")["r"].median())
    print(f"\n逐股中位比值分布 ({len(per)} 只):")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"  {q:>5.0%}: {per.quantile(q):.6f}")
    off = per[(per - 1).abs() > 0.05]
    print(f"  偏离 1 超过 5% 的股票: {len(off)}/{len(per)} ({len(off) / len(per):.1%})")

    verdict = "可以直接接续" if abs(ratio - 1) <= 0.05 and corr > 0.99 else "不能直接接续, 需换算或改口径"
    print(f"\n结论: {verdict}")
    if abs(ratio - 1) > 0.05:
        print(f"  中位比值 {ratio:.4f} 偏离 1 超过 5%, 说明两源统计口径不同")
    if corr <= 0.99:
        print(f"  相关 {corr:.4f} 不足 0.99, 说明不是同一个量")


if __name__ == "__main__":
    main()
