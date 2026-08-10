"""用 tushare margin_detail 接续 mtss_balance (融资融券余额)

背景
────
mtss_balance 原本来自 thsdk, 2026-06-30 断更, 依赖它的 4 个入选特征
(mtss_z / mtss_z_ma5 / mtss_z_ma20 / mtss_1d_ma20) 就此没有前向数据源。

对账结论 (scripts/diag_mtss_tushare_calib.py)
  mtss_balance 与 tushare margin_detail.rzrqye 在 289,521 个重叠样本上
  中位比值 1.000000、相关 1.000000、246 只股票无一偏离 5%
  —— 旧源那一列本来就是交易所口径的融资融券余额, 零换算直接替代。

所以这里把 tushare 当作该列的唯一权威源, 整列重写而不是拼接:
  · 覆盖从 247 只 / 2020-01 扩到 tushare 的 5,146 只 / 2019-01
  · 单一源, 没有拼接跳变, 不重演 K 线三源混用的问题

只改 mtss_balance 一列, 不增删行 —— 新增行会让 main_force_net 在那些日子变成
散点缺失, 又被 fillna(0) 填成假的"零流入", 那是另一个坑。

用法:
  python scripts/fill_mtss_from_tushare.py --dry-run
  python scripts/fill_mtss_from_tushare.py
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONS = ROOT / "data" / "raw" / "fund_flow_full" / "fundflow_history.parquet"
TS_DIR = ROOT / "data" / "raw" / "tushare" / "margin_detail"
TS_COL = "rzrqye"
MAX_RATIO_DEV = 0.05   # 与旧值的中位比值允许偏离 1 的幅度, 同 unit_check 的判据


def load_tushare():
    files = sorted(glob.glob(str(TS_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"ERROR: 没有 tushare margin_detail 数据于 {TS_DIR}")
    ts = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if TS_COL not in ts.columns:
        raise SystemExit(f"ERROR: tushare 数据缺列 {TS_COL}, 实际列: {list(ts.columns)}")
    ts["code"] = ts["ts_code"].astype(str).str[:6]
    ts["date"] = pd.to_datetime(ts["trade_date"], format="%Y%m%d")
    ts = ts[["code", "date", TS_COL]].dropna(subset=[TS_COL])
    ts[TS_COL] = ts[TS_COL].astype(float)
    return ts.drop_duplicates(["code", "date"], keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-recon", action="store_true",
                    help="重叠样本不足时仍允许换源(默认硬拒)。"
                         "只有在确认旧源本来就一直为空时才该用")
    a = ap.parse_args()

    cur = pd.read_parquet(CONS)
    cur["code"] = cur["code"].astype(str).str.zfill(6)
    cur["date"] = pd.to_datetime(cur["date"])
    have = cur["mtss_balance"].notna()
    before = int(have.sum())
    print(f"现役表: {len(cur):,} 行 | {cur['code'].nunique()} 只 | "
          f"{cur['date'].min():%F} ~ {cur['date'].max():%F}")
    if before:
        print(f"  mtss_balance 非空: {before:,} "
              f"({cur.loc[have, 'code'].nunique()} 只, "
              f"至 {cur.loc[have, 'date'].max():%F})")
    else:
        print("  mtss_balance 非空: 0 (整列空)")

    ts = load_tushare()
    print(f"tushare {TS_COL}: {len(ts):,} 行 | {ts['code'].nunique()} 只 | "
          f"{ts['date'].min():%F} ~ {ts['date'].max():%F}")

    # 换源前再对一次账 —— 不信任上游, 每次写盘都自己验一遍
    chk = cur.loc[cur["mtss_balance"].notna(), ["code", "date", "mtss_balance"]].merge(
        ts, on=["code", "date"], how="inner")
    if len(chk) >= 1000:
        r = (chk["mtss_balance"].astype(float) / chk[TS_COL].replace(0, pd.NA)).dropna()
        dev = abs(r.median() - 1)
        print(f"\n对账: {len(chk):,} 个重叠样本, 中位比值 {r.median():.6f}")
        if dev > MAX_RATIO_DEV:
            raise SystemExit(
                f"ERROR: 中位比值偏离 1 达 {dev:.1%} (上限 {MAX_RATIO_DEV:.0%}), "
                f"两源口径不同, 拒绝换源")
        print("  对账通过")
    elif a.allow_no_recon:
        print(f"\n!! 重叠样本仅 {len(chk):,} 条, 无法对账 —— 已由 --allow-no-recon 放行")
    else:
        # 对账是本脚本唯一的安全闸。旧实现在重叠样本不足时只打印一行提示
        # 就继续换源, 而 "mtss_balance 整列被抹成 NaN" 恰恰会使重叠样本为 0 ->
        # 闸门静默失效。这正是 2026-08-05 那次事故的同一种失败模式, 改成硬拒。
        raise SystemExit(
            f"ERROR: 重叠样本仅 {len(chk):,} 条(需 >=1000), 对账闸门无法生效, 拒绝换源\n"
            f"  现役表 mtss_balance 非空 {before:,} 条。若为 0, 说明旧源真值已被抹掉,\n"
            f"  请先跑 scripts/restore_fundflow_legacy.py 把历史真值接回来, 再跑本脚本。\n"
            f"  确实需要跳过对账时显式加 --allow-no-recon。")

    out = cur.drop(columns=["mtss_balance"]).merge(ts, on=["code", "date"], how="left")
    out = out.rename(columns={TS_COL: "mtss_balance"})
    out = out.reindex(columns=list(cur.columns))
    after = int(out["mtss_balance"].notna().sum())

    print(f"\nmtss_balance 非空: {before:,} -> {after:,} "
          f"(+{after - before:,})")
    print(f"  覆盖股票: {cur.loc[have, 'code'].nunique()} -> "
          f"{out.loc[out['mtss_balance'].notna(), 'code'].nunique()} 只")
    _old_max = f"{cur.loc[have, 'date'].max():%F}" if before else "无"
    print(f"  最新日期: {_old_max} -> "
          f"{out.loc[out['mtss_balance'].notna(), 'date'].max():%F}")

    if len(out) != len(cur):
        raise SystemExit(f"ERROR: 行数变了 {len(cur):,} -> {len(out):,}, 拒绝写盘")
    if after < before:
        raise SystemExit(f"ERROR: 非空数反而减少 {before:,} -> {after:,}, 拒绝写盘")
    for c in cur.columns:
        if c == "mtss_balance":
            continue
        o, n = int(cur[c].notna().sum()), int(out[c].notna().sum())
        if o != n:
            raise SystemExit(f"ERROR: 非目标列 {c} 非空数被改动 {o:,} -> {n:,}, 拒绝写盘")

    if a.dry_run:
        print("\ndry-run: 未写盘")
        return
    bak = CONS.with_name(f"fundflow_history.premtss_{time.strftime('%Y%m%d_%H%M%S')}.parquet")
    CONS.replace(bak)
    print(f"\n现役表已备份: {bak.name}")
    out.to_parquet(CONS, index=False)
    print(f"已写入 {CONS}")
    print("\n下一步: 需要一次全量特征重建才会让 mtss_* 特征真正带上这些值")


if __name__ == "__main__":
    main()
