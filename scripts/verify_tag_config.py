# -*- coding: utf-8 -*-
"""逐项对比两个实验 tag 的回测配置, 确认差异只在预期的那几项

为什么需要这个
──────────────
配对实验的全部说服力都建立在"两臂只差一个变量"上。回测脚本有几十个参数,
默认值和显式值混用时极易漂移 —— 比如 --tranche-n 默认 2 而参照组用的是 3,
不显式传就会得到一个看似合理、实际不可比的结果, 且不会有任何报错。
这个脚本把两个 tag 的 config 逐项 diff, 差异项必须与预期完全一致才能采信实验。

用法
────
    python scripts/verify_tag_config.py V24B V24T1 --expect train_file
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed"
SKIP = {"daily", "trades", "summary", "stability", "selected_features",
        "feat_importance_top120", "period", "n_days"}


def cfg(tag):
    fs = sorted(glob.glob(str(BASE / f"wf_daily_{tag}_s*.json")))
    if not fs:
        sys.exit(f"没有 {tag} 的结果")
    d = json.load(open(fs[0]))
    return {k: v for k, v in d.items() if k not in SKIP}, Path(fs[0]).name, d


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.split("用法")[1])
    ref, new = sys.argv[1], sys.argv[2]
    expect = []
    if "--expect" in sys.argv:
        expect = sys.argv[sys.argv.index("--expect") + 1:]
        expect = [e for e in expect if not e.startswith("--")]

    a, fa, da = cfg(ref)
    b, fb, db = cfg(new)
    print(f"参照 {ref}: {fa}\n实验 {new}: {fb}\n")

    keys = sorted(set(a) | set(b))
    diff = [k for k in keys if a.get(k, "<缺>") != b.get(k, "<缺>")]
    print(f"===== 配置差异 ({len(diff)} 项) =====")
    for k in diff:
        flag = "  <- 预期" if k in expect else "  ⚠ 非预期"
        print(f"  {k}:\n      {ref} = {a.get(k, '<缺>')}\n      {new} = {b.get(k, '<缺>')}{flag}")
    if not diff:
        print("  无差异")

    unexpected = [k for k in diff if k not in expect]
    missing = [k for k in expect if k not in diff]
    print(f"\n===== 结论 =====")
    if missing:
        print(f"  ⚠ 预期会变但实际没变: {missing}")
    if unexpected:
        print(f"  ⚠ 非预期差异 {len(unexpected)} 项: {unexpected}")
        print("  => 两臂不可比, 不要采信这次实验")
    else:
        print("  ✓ 差异全部在预期内, 两臂可比")

    # 特征集差异 (信息用, 不作判据: 换了矩阵本来就会选出不同特征)
    sa, sb = set(da.get("selected_features", [])), set(db.get("selected_features", []))
    if sa and sb:
        print(f"\n特征集: {ref} {len(sa)} 个, {new} {len(sb)} 个, 交集 {len(sa & sb)}")
        only_b = sorted(sb - sa)
        tk = [c for c in only_b if c.startswith("tk_")]
        print(f"  {new} 新入选 {len(only_b)} 个, 其中逐笔列 {len(tk)} 个")
        if tk:
            print(f"    {tk}")
        other = [c for c in only_b if not c.startswith("tk_")]
        if other:
            print(f"  非逐笔的新入选(被挤进来的): {other}")


if __name__ == "__main__":
    main()
