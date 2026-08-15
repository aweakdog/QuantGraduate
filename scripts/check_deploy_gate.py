# -*- coding: utf-8 -*-
"""部署前闸门: 检查候选特征集能不能上线

上线一份特征集之前必须过四关, 任何一关不过就不能部署:
  1. 一致性 —— 20 个种子跑出来的 selected_features 必须完全相同。
     若不同, 说明特征筛选不可复现(历史上出过: 同一份数据两次重建回测收益 24.7% vs 171.3%)。
  2. 无覆盖掩码 —— 入选列的"有值"集合不能高度落在人工名单里 (docs §8.4)。
     这是 V24A 那套 75 特征的病根。
  3. 无零方差 —— 入选列在训练矩阵上不能是常量 (历史事故: 名义 80 特征实际只有 69 个活的)。
  4. 覆盖率 —— 入选列的股票覆盖率不能太低。

用法
────
    python scripts/check_deploy_gate.py V24B
    python scripts/check_deploy_gate.py V24B --matrix data/processed/training_data_pit_v24.parquet
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed"
WATCHLIST = ROOT / "data/universe/watchlist_216.json"
CONTAINMENT = 0.85      # 覆盖集落在名单内的比例上限
MIN_COVERAGE = 0.50     # 股票覆盖率下限

# 这些前缀是 live_signal 在运行时现算的市场级/隔夜特征, 本来就不进训练矩阵,
# 所以"矩阵里没有"是正常的, 不算缺列。真值定义在 scripts/daily_rebuild.py:56
RUNTIME_FEAT_PREFIXES = ("ovn_", "overnight_", "intraday_", "mkt_")


def load_watchlist(path: Path):
    """watchlist_216.json 的真实结构是 {"watchlist": [{"code","name","theme"}]}"""
    if not path.exists():
        return set()
    raw = json.load(open(path))
    if isinstance(raw, dict):
        items = raw.get("watchlist") or raw.get("codes") or []
    else:
        items = raw
    out = set()
    for x in items:
        s = str(x.get("code", "") if isinstance(x, dict) else x)
        d6 = "".join(ch for ch in s if ch.isdigit())[:6]
        if len(d6) == 6:
            out.add(d6)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "V24B"
    mpath = ROOT / "data/processed/training_data_pit_v24.parquet"
    if "--matrix" in sys.argv:
        mpath = Path(sys.argv[sys.argv.index("--matrix") + 1])

    fs = sorted(glob.glob(str(BASE / f"wf_daily_{tag}_s*.json")))
    if not fs:
        sys.exit(f"没有 {tag} 的结果")

    # ---------- 1. 一致性 ----------
    sets, per_file = {}, {}
    for f in fs:
        d = json.load(open(f))
        sel = d.get("selected_features")
        if not sel:
            continue
        key = tuple(sorted(sel))
        sets.setdefault(key, []).append(Path(f).name)
        per_file[Path(f).name] = sel
    print(f"===== 1. 一致性 =====\n{tag}: {len(fs)} 个结果文件, "
          f"{len(sets)} 种不同的特征集")
    if len(sets) != 1:
        for i, (k, v) in enumerate(sets.items(), 1):
            print(f"  变体{i}: {len(k)} 列, {len(v)} 个文件, 例 {v[0]}")
        ks = list(sets)
        print(f"  变体1 vs 变体2 差异: 只在1里 {sorted(set(ks[0]) - set(ks[1]))[:8]}")
        print("  ⚠ 不通过 —— 特征筛选不可复现, 不能部署")
    else:
        print("  ✓ 通过 —— 20 个种子的特征集完全相同")
    feats = list(next(iter(sets.keys())))
    print(f"  特征数 {len(feats)}")

    # ---------- 加载矩阵 ----------
    if not mpath.exists():
        sys.exit(f"找不到矩阵 {mpath}")
    m = pd.read_parquet(mpath)
    m["c6"] = m["code"].astype(str).str.extract(r"(\d{6})")[0]
    feats_all = list(feats)          # 完整清单, 部署要用这个
    absent = [c for c in feats if c not in m.columns]
    runtime = [c for c in absent if c.startswith(RUNTIME_FEAT_PREFIXES)]
    miss = [c for c in absent if c not in runtime]
    feats = [c for c in feats if c in m.columns]   # 只有这些能在矩阵上体检
    print(f"\n矩阵 {mpath.name}: {m.shape[0]} 行 x {m.shape[1]} 列, "
          f"股票 {m['c6'].nunique()} 只")
    print(f"  入选列在矩阵中: {len(feats)}  运行时现算(正常): {len(runtime)}")
    if runtime:
        print(f"    {sorted(runtime)}")
    if miss:
        print(f"  ⚠ 真缺失 {len(miss)} 列: {miss[:8]}")

    wl = load_watchlist(WATCHLIST)
    print(f"人工名单 watchlist_216: {len(wl)} 只")

    # ---------- 2/3/4 逐列体检 ----------
    tot = m["c6"].nunique()
    bad_mask, bad_zero, bad_cov = [], [], []
    for c in feats:
        s = pd.to_numeric(m[c], errors="coerce")
        ok = s.notna()
        codes = set(m.loc[ok, "c6"].unique())
        cov = len(codes) / tot if tot else 0
        cont = (len(codes & wl) / len(codes)) if codes and wl else 0
        if ok.sum() and s[ok].std(skipna=True) == 0:
            bad_zero.append(c)
        if cov < MIN_COVERAGE:
            bad_cov.append((c, cov, cont))
        elif cont >= CONTAINMENT and cov < 0.95:
            bad_mask.append((c, cov, cont))

    print(f"\n===== 2. 覆盖掩码 (覆盖集 >={CONTAINMENT:.0%} 落在人工名单内) =====")
    if bad_mask:
        for c, cov, cont in bad_mask:
            print(f"  ⚠ {c:<28} 覆盖 {cov:.1%}  名单内 {cont:.1%}")
        print(f"  ⚠ 不通过 —— {len(bad_mask)} 列是掩码")
    else:
        print("  ✓ 通过 —— 无掩码列")

    print("\n===== 3. 零方差 =====")
    if bad_zero:
        print(f"  ⚠ 不通过 —— {len(bad_zero)} 列是常量: {bad_zero}")
    else:
        print("  ✓ 通过 —— 无常量列")

    print(f"\n===== 4. 覆盖率 (下限 {MIN_COVERAGE:.0%}) =====")
    if bad_cov:
        for c, cov, cont in bad_cov:
            print(f"  ⚠ {c:<28} 覆盖 {cov:.1%}  名单内 {cont:.1%}")
        print(f"  ⚠ 不通过 —— {len(bad_cov)} 列覆盖过低")
    else:
        print("  ✓ 通过 —— 所有列覆盖率达标")

    cov_all = [pd.to_numeric(m[c], errors="coerce").notna().mean() for c in feats]
    print(f"\n入选列非空率: 中位 {np.median(cov_all):.1%}  "
          f"最低 {min(cov_all):.1%}  最高 {max(cov_all):.1%}")

    gate = not (len(sets) != 1 or bad_mask or bad_zero or bad_cov or miss)
    print(f"\n{'=' * 46}\n闸门结论: {'✓ 可以部署' if gate else '✗ 不可部署'}")
    if gate:
        out = BASE / f"features_{tag}.json"
        json.dump(feats_all, open(out, "w"), ensure_ascii=False, indent=1)
        print(f"特征清单已写出: {out}  ({len(feats_all)} 列, "
              f"其中 {len(runtime)} 列由 live_signal 运行时现算)")


if __name__ == "__main__":
    main()
