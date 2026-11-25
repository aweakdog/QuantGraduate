"""组合层去极值 & 波动率倒数加权研究 (PO, 2026-11-23 预注册, 见 docs/experiment_board.md)。纯回放、零训练。

  python scripts/port_opt_study.py run --root /home/yliog/quant-research-20260912-n2 --workers 12
  python scripts/port_opt_study.py summary --root /home/yliog/quant-research-20260912-n2

复用 N7_Q5 (qmt10w 现行: 10万/n5/主板/T1B/PG1) 的 40 个 purge6 预测缓存, 仅改组合执行层:
  ST1      : 买入跳过当日第 1 名 (检验 10-09「第 1 名最弱」规律)
  ST2      : 买入跳过前 2 名 (邻域一致性)
  IV20     : 近 20 日波动率倒数加权 (0.5x~2.0x, 严格 PIT)
  IV60     : 近 60 日波动率倒数加权 (邻域一致性)
  ST1_IV20 : 两者叠加 (描述)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402
import run_overnight_research as ro  # noqa: E402

VARIANTS = {
    "ST1": ["--skip-top", "1"],
    "ST2": ["--skip-top", "2"],
    "IV20": ["--weight-mode", "inv_vol", "--vol-window", "20"],
    "IV60": ["--weight-mode", "inv_vol", "--vol-window", "60"],
    "ST1_IV20": ["--skip-top", "1", "--weight-mode", "inv_vol", "--vol-window", "20"],
}
NEIGHBOR = {"ST1": "ST2", "IV20": "IV60"}
GATE = {"ret_pp": 5.0, "pos_min": 26, "sharpe": 0.08, "dd_pp": 2.0}


def command(root, var, seed):
    args = ro.common_args("B", rg.END, "training_data_pit_v24_tick1.parquet", None, rg.TEST_START,
                          profile=ro.EXEC_POINTS["Q5"])
    args += list(VARIANTS[var])
    return [sys.executable, "-u", str(Path(root) / "scripts/wf_v35_breadth_alpha.py"), *args,
            "--lgb-seed", str(seed), "--label-alignment", "purge6",
            "--tag", f"PO_{var}", "--load-preds", ro.n6_caches(seed)[1]]


def compare_port(base, var):
    """不依赖空仓段差异(状态序列完全相同), 用全期/前后两半/N4·N5分批/近126日评判。"""
    seeds = sorted(set(base) & set(var))
    n4_set = set(ro.N6_SEEDS[:20])
    rows = []
    for s in seeds:
        b = base[s]["daily_ret"]
        v = var[s]["daily_ret"].reindex(b.index).fillna(0.0)
        sb, sv = rg.stats(b), rg.stats(v)
        h1, h2 = b.index < rg.SPLIT, b.index >= rg.SPLIT
        r126 = b.index[-126:]
        rows.append({
            "seed": s, "batch": "N4" if s in n4_set else "N5",
            "ret_b": sb["ret"], "ret_v": sv["ret"],
            "d_ret": sv["ret"] - sb["ret"],
            "d_sharpe": sv["sharpe"] - sb["sharpe"],
            "d_dd": sv["dd"] - sb["dd"],
            "d_h1": rg.stats(v[h1])["ret"] - rg.stats(b[h1])["ret"],
            "d_h2": rg.stats(v[h2])["ret"] - rg.stats(b[h2])["ret"],
            "d_r126": rg.stats(v.loc[r126])["ret"] - rg.stats(b.loc[r126])["ret"],
        })
    t = pd.DataFrame(rows)
    med = t.median(numeric_only=True)
    b1 = t[t["batch"] == "N4"]
    b2 = t[t["batch"] == "N5"]
    res = {
        "n": len(seeds),
        "var_med_ret": round(float(t["ret_v"].median()), 2),
        "var_worst_ret": round(float(t["ret_v"].min()), 2),
        "med_d_ret": round(float(med["d_ret"]), 2),
        "pos": int((t["d_ret"] > 0).sum()),
        "med_d_sharpe": round(float(med["d_sharpe"]), 3),
        "med_d_dd": round(float(med["d_dd"]), 2),
        "med_d_h1": round(float(med["d_h1"]), 2),
        "med_d_h2": round(float(med["d_h2"]), 2),
        "med_d_r126": round(float(med["d_r126"]), 2),
        "n4_med_d_ret": round(float(b1["d_ret"].median()), 2) if len(b1) else None,
        "n4_pos": int((b1["d_ret"] > 0).sum()),
        "n5_med_d_ret": round(float(b2["d_ret"].median()), 2) if len(b2) else None,
        "n5_pos": int((b2["d_ret"] > 0).sum()),
    }
    res["pass_core"] = bool(
        res["n"] == 40
        and res["med_d_ret"] >= GATE["ret_pp"]
        and res["pos"] >= GATE["pos_min"]
        and res["med_d_sharpe"] >= GATE["sharpe"]
        and res["med_d_dd"] >= -GATE["dd_pp"]
        and res["med_d_h1"] > 0 and res["med_d_h2"] > 0
        and (res["n4_med_d_ret"] or -1) > 0 and (res["n5_med_d_ret"] or -1) > 0
    )
    return res


def summary(root):
    base = {s: rg.daily(rg.out_path(root, "N7_Q5", s)) for s in ro.N6_SEEDS if rg.out_path(root, "N7_Q5", s).exists()}
    res = {
        "baseline": "N7_Q5 (equal weight, rank 1..5)",
        "n_base": len(base),
        "base_median": {k: round(float(np.median([rg.stats(d["daily_ret"])[k] for d in base.values()])), 2)
                        for k in ("ret", "sharpe", "dd")},
        "base_worst_ret": round(float(min(rg.stats(d["daily_ret"])["ret"] for d in base.values())), 2),
    }
    for v in VARIANTS:
        var = {s: rg.daily(rg.out_path(root, f"PO_{v}", s)) for s in ro.N6_SEEDS
               if rg.out_path(root, f"PO_{v}", s).exists()}
        if var:
            res[v] = compare_port(base, var)
    for v in VARIANTS:
        if v not in res:
            continue
        nb = NEIGHBOR.get(v)
        res[v]["pass"] = bool(res[v]["pass_core"] and (nb is None or res.get(nb, {}).get("med_d_ret", -1) >= 0))
    (Path(root) / "summary_po.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    if a.phase == "run":
        rg.run(a.root, a.workers, VARIANTS, "PO", command)
    else:
        print(json.dumps(summary(a.root), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
