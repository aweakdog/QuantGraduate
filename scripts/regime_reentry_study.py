"""快速回场 (RF, 11-04 预注册, 见 docs/experiment_board.md)。纯回放、零训练。

  python scripts/regime_reentry_study.py run --root /home/yliog/quant-research-20260912-n2 --workers 12
  python scripts/regime_reentry_study.py summary --root /home/yliog/quant-research-20260912-n2
空仓中广度单日暴涨时, 要不要不等 2 天确认就回场。复用 N7_Q5 (qmt10w 现行) 的 40 个 purge6 预测缓存,
只改回场规则: F1 (回场确认 1 天) / T15、T20 (广度单日涨 ≥15/20 个点且收在 40% 以上当天回场),
对照已有 N7_Q5 (40% 阈值、进出都确认 2 天)。比较口径与 RG 完全相同 (regime_gate_study.compare)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_gate_study as rg  # noqa: E402
import run_overnight_research as ro  # noqa: E402

VARIANTS = {"F1": {"--regime-confirm-on": "1"}, "T15": {"--regime-thrust": "0.15"},
            "T20": {"--regime-thrust": "0.20"}}
NEIGHBOR = {"T15": "T20", "T20": "T15"}     # 推力阈值要邻域一致: 一个过, 另一个也不能差于现行


def command(root, var, seed):
    args = ro.common_args("B", rg.END, "training_data_pit_v24_tick1.parquet", None, rg.TEST_START,
                          profile=ro.EXEC_POINTS["Q5"])
    for k, v in VARIANTS[var].items():
        args += [k, v]
    return [sys.executable, "-u", str(Path(root) / "scripts/wf_v35_breadth_alpha.py"), *args, "--lgb-seed", str(seed),
            "--label-alignment", "purge6", "--tag", f"RF_{var}", "--load-preds", ro.n6_caches(seed)[1]]


def judge(res):
    for v in VARIANTS:
        if v in res:
            nb = res.get(NEIGHBOR.get(v, ""), {})
            res[v]["pass"] = bool(res[v]["pass_core"] and (v not in NEIGHBOR or nb.get("med_d_ret", -1) >= 0))
    return res


def summary(root):
    base = {s: rg.daily(rg.out_path(root, "N7_Q5", s)) for s in ro.N6_SEEDS if rg.out_path(root, "N7_Q5", s).exists()}
    res = {"baseline": "N7_Q5 (B40, 进出确认 2 天)", "n_base": len(base),
           "base_median": {k: round(float(np.median([rg.stats(d["daily_ret"])[k] for d in base.values()])), 2)
                           for k in ("ret", "sharpe", "dd")},
           "cash_share_base": round(float(next(iter(base.values()))["in_cash"].mean()), 3)}
    for v in VARIANTS:
        var = {s: rg.daily(rg.out_path(root, f"RF_{v}", s)) for s in ro.N6_SEEDS
               if rg.out_path(root, f"RF_{v}", s).exists()}
        if var:
            res[v] = rg.compare(base, var) | {"cash_share": round(float(next(iter(var.values()))["in_cash"].mean()), 3)}
    res = judge(res)
    (Path(root) / "summary_rf.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.phase == "run":
        rg.run(a.root, a.workers, VARIANTS, "RF", command)
    else:
        print(json.dumps(summary(a.root), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
