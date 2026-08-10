"""列出某变体在各窗口锁定的特征集, 并按类别汇总重要性。

用途: 回答"我们现在到底在用哪些特征"、"两个窗口选出来的东西一不一样"。
后者尤其重要 —— 如果 A/B 选出的特征重合度低, 说明所谓"有效特征"本身就不
跨期稳定, 那么在任一窗口上做的特征结论都不该外推。

重要性口径是 LightGBM 的分裂增益占比, 只反映模型用了多少, 不代表赚了多少钱
(融券余额那次就是重要性排第1但端到端无效)。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grid import PROC, WINDOWS, VARIANTS, features_json  # noqa: E402

PREFIX_CAT = [
    (("vol_", "atr"), "波动率/量能"),
    (("con_", "leader_"), "概念主题聚合"),
    (("mf_", "dde_", "fund_flow"), "个股资金流"),
    (("ma5", "ma20", "macd", "rsi", "ret_", "pos_", "intraday"), "趋势动量"),
    (("ovn_", "overnight"), "隔夜"),
    (("mtss_", "rq_"), "融资融券"),
    (("tev_", "ev_", "days_since_ann", "ann_"), "事件"),
]


def cat(f):
    for pres, name in PREFIX_CAT:
        if f.startswith(pres):
            return name
    return "基本面"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mb_dmw", choices=list(VARIANTS))
    ap.add_argument("--list", action="store_true", help="逐列打印")
    a = ap.parse_args()

    sels = {}
    for w in WINDOWS:
        d = json.load(open(PROC / features_json(w, a.variant)))
        sels[w] = (d["selected_features"],
                   {x["feature"]: x["importance"]
                    for x in d["feat_importance_top120"]})

    wins = list(WINDOWS)
    sa, sb = set(sels[wins[0]][0]), set(sels[wins[1]][0])
    print(f"变体 {a.variant}")
    print(f"{wins[0]} 选中 {len(sa)} 列, {wins[1]} 选中 {len(sb)} 列, "
          f"交集 {len(sa & sb)} 列 (重合度 {len(sa & sb) / len(sb) * 100:.0f}%)\n")

    for w in wins:
        sel, imp = sels[w]
        agg = {}
        for f in sel:
            c = agg.setdefault(cat(f), [0, 0.0])
            c[0] += 1
            c[1] += imp.get(f, 0.0)
        tot = sum(v[1] for v in agg.values()) or 1.0
        print(f"窗口 {w} ({WINDOWS[w]['desc']})")
        print(f"{'类别':<16}{'列数':>6}{'重要性占比':>12}")
        for c, (n, s) in sorted(agg.items(), key=lambda x: -x[1][1]):
            print(f"{c:<16}{n:>6}{100 * s / tot:>11.1f}%")
        print()

    if a.list:
        for w in wins:
            sel, imp = sels[w]
            other = sb if w == wins[0] else sa
            print(f"\n窗口 {w} 全部 {len(sel)} 列 (按重要性)")
            print(f"{'#':<4}{'特征':<34}{'类别':<16}{'重要性':>10}{'另一窗也选':>10}")
            for i, f in enumerate(sorted(sel, key=lambda x: -imp.get(x, 0)), 1):
                print(f"{i:<4}{f:<34}{cat(f):<16}{imp.get(f, 0):>10.5f}"
                      f"{('是' if f in other else ''):>10}")


if __name__ == "__main__":
    main()
