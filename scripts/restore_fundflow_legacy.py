"""一次性修复: 把 2026-08-05 被抹掉的旧源独有列从备份接回 consolidated 资金流表

事故经过
────────
scripts/pull_fundflow_shard.py 的 do_merge() 名为"合并"实为整表覆盖: 它把三个
新浪分片 concat 起来直接 to_parquet 到 fundflow_history.parquet, 旧表只备份、
不回接。新浪源拿不到 dde_net / mtss_balance / fund_flow (在 pull_fundflow_sina
里就是 pd.NA 占位), 于是这三列的真值被整段历史抹掉, 非空率 84%/71%/85% -> 0%。

紧接着 feature_engine.calc_fund_features 的 `s = df[src].fillna(0)` 把"整列全
NaN"伪造成"值为 0", 连缺失都看不出来。结果是线上 80 个入选特征里 11 个在整个
训练集上恒为常量 0 (6 个 con_dde_net_* / 4 个 mtss_* / 1 个 fund_flow_z_ma20),
模型名义 80 特征、实际只有 69 个, 且全程没有任何报错。

那次覆盖不只丢列, 还丢行: 备份里 347,081 条带真值的记录, 只有 198,990 条在现役表
里还找得到对应 (code, date), 另外 145,265 条(257 只, 其中 122 只整只消失)所在的行
被整体抹掉了 —— 新浪分片按当前 PIT 池拉数, 覆盖不到已退市/已移出池的股票, 而训练集
是按 PIT 池取历史的, 这些历史行仍然算数。

所以本脚本做两件事:
  1. 把独有列的真值按 (code, date) 从备份接回现役表;
  2. 把只存在于备份的行整行补回。

第 2 步会让 main_force_net/pct 这一列在不同行上来自不同源(现役=新浪, 补回=thsdk)。
不装作同源: backfill_fundflow_universe 的 unit_check 在重叠样本上比过中位比值,
偏离 1 在 5% 以内才允许混入, 这里沿用同一结论。相比丢掉 43% 的真值, 这个代价更小。

dde_net 源数据本身在 2026-06-30 就断了, 所以恢复后 06-30 之后仍然是 NaN ——
这是事实, 不该再用 0 掩盖。对线上模型的结论不变: 这类没有前向数据源的特征
应当从特征集剔除, 而不是留着充数。恢复历史的意义在于让训练矩阵不再被污染,
使重训和 A/B 比较重新有效。

用法:
  python scripts/restore_fundflow_legacy.py --dry-run   # 只看会恢复多少, 不写盘
  python scripts/restore_fundflow_legacy.py             # 实际修复(先备份)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FF_DIR = ROOT / "data" / "raw" / "fund_flow_full"
CONS = FF_DIR / "fundflow_history.parquet"
LEGACY_COLS = ["dde_net", "mtss_balance", "fund_flow"]
KEYS = ["code", "date"]


def norm(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def pick_source():
    """在所有备份里挑独有列真值最多的那份作为恢复源。

    不假定哪份最好 —— 直接按"三列非空行数之和"取最大, 并把候选都打印出来,
    便于人工复核。
    """
    cands = []
    for p in sorted(FF_DIR.glob("fundflow_history*.parquet")):
        if p.name == CONS.name:
            continue
        try:
            d = norm(pd.read_parquet(p))
        except Exception as e:
            print(f"  跳过 {p.name}: 读取失败 {e}")
            continue
        have = [c for c in LEGACY_COLS if c in d.columns]
        if not have:
            continue
        cnt = {c: int(d[c].notna().sum()) for c in have}
        cands.append((sum(cnt.values()), p, d, cnt))
    if not cands:
        raise SystemExit("ERROR: 没有任何备份含独有列, 无法恢复")
    cands.sort(key=lambda x: -x[0])
    print("候选恢复源 (按独有列非空行数合计排序):")
    for tot, p, _, cnt in cands:
        print(f"  {p.name:52s} 合计{tot:>9,}  " +
              "  ".join(f"{c}={n:,}" for c, n in cnt.items()))
    return cands[0][1], cands[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘")
    a = ap.parse_args()

    if not CONS.exists():
        raise SystemExit(f"ERROR: 找不到 {CONS}")
    cur = norm(pd.read_parquet(CONS))
    print(f"现役表: {len(cur):,} 行 | {cur['code'].nunique()} 只 | "
          f"{cur['date'].min():%F} ~ {cur['date'].max():%F}")
    print("现役表独有列非空: " +
          "  ".join(f"{c}={int(cur[c].notna().sum()):,}" if c in cur.columns else f"{c}=缺列"
                    for c in LEGACY_COLS))
    print()

    src_path, src = pick_source()
    print(f"\n选用: {src_path.name}")

    have = [c for c in LEGACY_COLS if c in src.columns]
    add = (src[KEYS + have]
           .dropna(subset=have, how="all")
           .drop_duplicates(KEYS, keep="last"))
    print(f"备份中有真值的行: {len(add):,} ({add['code'].nunique()} 只, "
          f"{add['date'].min():%F} ~ {add['date'].max():%F})")

    out = cur.drop(columns=[c for c in have if c in cur.columns])
    out = out.merge(add, on=KEYS, how="left")

    # 只存在于备份的行整行补回
    only_bak = src.merge(out[KEYS], on=KEYS, how="left", indicator=True)
    only_bak = only_bak[only_bak["_merge"] == "left_only"].drop(columns="_merge")
    only_bak = only_bak.dropna(subset=have, how="all")
    if len(only_bak):
        print(f"\n现役表缺失、仅备份中存在的行: {len(only_bak):,} "
              f"({only_bak['code'].nunique()} 只, "
              f"{only_bak['date'].min():%F} ~ {only_bak['date'].max():%F}) -> 补回")
        out = pd.concat([out, only_bak.reindex(columns=out.columns)], ignore_index=True)
        out = (out.sort_values(KEYS)
                  .drop_duplicates(KEYS, keep="first")
                  .reset_index(drop=True))
    # 列顺序还原成现役表的样子, 避免下游按位置取列的代码受影响
    out = out.reindex(columns=[c for c in cur.columns if c in out.columns] +
                              [c for c in out.columns if c not in cur.columns])

    print("\n恢复效果:")
    ok = True
    for c in LEGACY_COLS:
        before = int(cur[c].notna().sum()) if c in cur.columns else 0
        after = int(out[c].notna().sum()) if c in out.columns else 0
        print(f"  {c:14s} 非空 {before:>9,} -> {after:>9,}")
        if after < before:
            ok = False
    print(f"  {'行数':14s} 合计 {len(cur):>9,} -> {len(out):>9,}")
    if len(out) < len(cur):
        raise SystemExit(f"ERROR: 行数反而减少 {len(cur):,} -> {len(out):,}, 拒绝写盘")
    if not ok:
        raise SystemExit("ERROR: 有列反而更空了, 拒绝写盘")
    # 其余列只允许增加(来自补回的行), 不允许减少
    for c in cur.columns:
        if c in LEGACY_COLS or c not in out.columns:
            continue
        o, n = int(cur[c].notna().sum()), int(out[c].notna().sum())
        if n < o:
            raise SystemExit(f"ERROR: 非目标列 {c} 非空数减少 {o:,} -> {n:,}, 拒绝写盘")
    # 现役表原有的每一行都必须还在
    chk = cur[KEYS].merge(out[KEYS], on=KEYS, how="left", indicator=True)
    lost = int((chk["_merge"] == "left_only").sum())
    if lost:
        raise SystemExit(f"ERROR: 现役表有 {lost:,} 行在结果中丢失, 拒绝写盘")

    if a.dry_run:
        print("\ndry-run: 未写盘")
        return
    bak = CONS.with_name(f"fundflow_history.prerestore_{time.strftime('%Y%m%d_%H%M%S')}.parquet")
    CONS.replace(bak)
    print(f"\n现役表已备份: {bak.name}")
    out.to_parquet(CONS, index=False)
    print(f"已写入 {CONS}")
    print("\n注意: 训练矩阵的历史部分仍是被污染的 0 值, 必须跑一次全量特征重建")
    print("      (feature_engine --no-incremental) 才会真正修正。")


if __name__ == "__main__":
    main()
