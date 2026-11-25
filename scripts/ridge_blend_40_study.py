"""E6: DART × 异质线性 Ridge 40 种子邻域验证 (RB40, 2026-11-23 预注册, 见 docs/experiment_board.md)。

零训练树模型, 在 041 ~/quant-research-20260912-n2 同冻结快照上:
  1) 导出同口径矩阵并训练确定性 Ridge (alpha=1000, 10000, 100000; 约 12 秒)
  2) 对 40 个已有 N7_Q5 purge6 预测缓存 (N4 20 种子 + N5 20 全新种子) 合成混合缓存:
       R100K_10 / R100K_15 / R100K_20 / R100K_25 (主邻域序列, alpha=100000)
       R10K_15  / R10K_20                        (跨 alpha 稳健对照, alpha=10000)
       R1K_15   / R1K_20                         (跨 alpha 稳健对照, alpha=1000)
  3) 同步合成「生产 5 种子集成 (42,7,123,2024,31337)」纯 DART vs 叠 Ridge 15%/20% 逐字对照
  4) 按 40 种子预注册门汇总输出 summary_rb40.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blend_preds as bp  # noqa: E402
import port_opt_study as po  # noqa: E402
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402
import run_overnight_research as ro  # noqa: E402

ARMS = {
    "R100K_10": ("preds_RIDGE_a100000.pkl", 0.10),
    "R100K_15": ("preds_RIDGE_a100000.pkl", 0.15),
    "R100K_20": ("preds_RIDGE_a100000.pkl", 0.20),
    "R100K_25": ("preds_RIDGE_a100000.pkl", 0.25),
    "R10K_15": ("preds_RIDGE_a10000.pkl", 0.15),
    "R10K_20": ("preds_RIDGE_a10000.pkl", 0.20),
    "R1K_15": ("preds_RIDGE_a1000.pkl", 0.15),
    "R1K_20": ("preds_RIDGE_a1000.pkl", 0.20),
}
NEIGHBORS = {
    "R100K_15": ("R100K_10", "R100K_20", "R10K_15"),
    "R100K_20": ("R100K_15", "R100K_25", "R10K_20"),
    "R10K_15": ("R100K_15", "R10K_20"),
    "R10K_20": ("R100K_20", "R10K_15"),
    "R1K_15": ("R10K_15", "R1K_20"),
    "R1K_20": ("R10K_20", "R1K_15"),
}
PROD_SEEDS = [42, 7, 123, 2024, 31337]


def build_prod_ens_cache(root: Path, out_path: Path):
    """按生产 5 种子等权平均预测合成逐字等价缓存。"""
    pdir = root / "data/processed"
    objs = [pickle.load(open(pdir / ro.n6_caches(s)[1], "rb")) for s in PROD_SEEDS]
    by_d = [{dp["date"]: dict(zip(dp["ranked"], dp["pred_vals"])) for dp in o["preds"]} for o in objs]
    dates = sorted(set.intersection(*(set(m.keys()) for m in by_d)))
    preds = []
    for d in dates:
        codes = sorted(set.intersection(*(set(m[d].keys()) for m in by_d)))
        vals = np.mean([[m[d][c] for c in codes] for m in by_d], axis=0)
        order = np.argsort(-vals, kind="mergesort")
        preds.append({
            "date": d,
            "ranked": [codes[k] for k in order],
            "pred_vals": [round(float(vals[k]), 6) for k in order],
            "ic": None,
        })
    meta = dict(objs[0]["meta"])
    meta["model"] = "prod_ens5_purge6"
    with open(out_path, "wb") as fh:
        pickle.dump({"meta": meta, "preds": preds}, fh)


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prep", "run", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    root = Path(a.root)
    pdir = root / "data/processed"
    env = os.environ.copy()
    env["QUANT_DATA_DIR"] = str(root / "data")
    env["QUANT_MODE"] = "backtest"

    if a.phase == "prep":
        mat = pdir / "rb40_export_matrix.parquet"
        if not mat.exists():
            args = ro.common_args("B", rg.END, "training_data_pit_v24_tick1.parquet", None, rg.TEST_START,
                                  profile=ro.EXEC_POINTS["Q5"])
            cmd = [sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
                   *args, "--label-alignment", "purge6", "--export-matrix", mat.name]
            subprocess.run(cmd, cwd=root, env=env, check=True)
        cmd_r = [sys.executable, "-u", str(root / "scripts/train_ridge_preds.py"),
                 "--matrix", str(mat), "--out-dir", str(pdir), "--alphas", "1000,10000,100000", "--winsor", "0.02"]
        subprocess.run(cmd_r, cwd=root, env=env, check=True)
        print("RB40_PREP_DONE", flush=True)
        return 0

    if a.phase == "run":
        jobs = []
        corr_map = {k: [] for k in ARMS}
        for arm, (ridge_name, w) in ARMS.items():
            rp = pdir / ridge_name
            for s in ro.N6_SEEDS:
                _, pg_c, _ = ro.n6_caches(s)
                dst_c = f"preds_RB40_{arm}_s{s}.pkl"
                if not (pdir / dst_c).exists():
                    info = bp.blend_two(pdir / pg_c, rp, w, pdir / dst_c, f"rb40_{arm}")
                    if info["corr_mean"] is not None:
                        corr_map[arm].append(info["corr_mean"])
                if not rg.out_path(root, f"RB40_{arm}", s).exists():
                    jobs.append((f"RB40_{arm}", dst_c, s))
        # 生产 5 种子集成对照
        ens_base = pdir / "preds_RB40_ENS5_BASE.pkl"
        if not ens_base.exists():
            build_prod_ens_cache(root, ens_base)
        if not rg.out_path(root, "RB40_ENS5_BASE", 42).exists():
            jobs.append(("RB40_ENS5_BASE", ens_base.name, 42))
        for tag, rname, w in (
            ("RB40_ENS5_R100K_15", "preds_RIDGE_a100000.pkl", 0.15),
            ("RB40_ENS5_R100K_20", "preds_RIDGE_a100000.pkl", 0.20),
            ("RB40_ENS5_R10K_20", "preds_RIDGE_a10000.pkl", 0.20),
            ("RB40_ENS5_R1K_20", "preds_RIDGE_a1000.pkl", 0.20),
        ):
            dst_c = f"preds_{tag}.pkl"
            if not (pdir / dst_c).exists():
                bp.blend_two(ens_base, pdir / rname, w, pdir / dst_c, tag.lower())
            if not rg.out_path(root, tag, 42).exists():
                jobs.append((tag, dst_c, 42))

        if any(corr_map.values()):
            (root / "rb40_corr.json").write_text(json.dumps(
                {k: round(float(np.mean(v)), 4) for k, v in corr_map.items() if v}, indent=1))
        print(f"RB40: {len(jobs)} 个纯回放待跑 (workers={a.workers})", flush=True)

        def one(job):
            tag, cname, s = job
            args = ro.common_args("B", rg.END, "training_data_pit_v24_tick1.parquet", None, rg.TEST_START,
                                  profile=ro.EXEC_POINTS["Q5"])
            cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
                   *args, "--lgb-seed", str(s), "--label-alignment", "purge6",
                   "--tag", tag, "--load-preds", cname]
            log = root / "logs" / f"{tag.lower()}_s{s}.log"
            with log.open("w") as fh:
                rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            return tag, s, rc

        errs = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for tag, s, rc in ex.map(one, jobs):
                if rc != 0:
                    errs += 1
                    print(f"  ERR {tag} s{s}: rc={rc}", flush=True)
        print(f"RB40_RUN_DONE errs={errs}", flush=True)
        return 1 if errs else 0

    base = {s: rg.daily(rg.out_path(root, "N7_Q5", s)) for s in ro.N6_SEEDS if rg.out_path(root, "N7_Q5", s).exists()}
    res = {
        "baseline": "N7_Q5 (40 seeds, pure DART+L2+PG1)",
        "n_base": len(base),
        "base_median": {k: round(float(np.median([rg.stats(d["daily_ret"])[k] for d in base.values()])), 2)
                        for k in ("ret", "sharpe", "dd")},
        "base_worst_ret": round(float(min(rg.stats(d["daily_ret"])["ret"] for d in base.values())), 2),
        "arms": {},
        "prod_ens5": {},
    }
    for arm in ARMS:
        var = {s: rg.daily(rg.out_path(root, f"RB40_{arm}", s)) for s in ro.N6_SEEDS
               if rg.out_path(root, f"RB40_{arm}", s).exists()}
        if var:
            res["arms"][arm] = po.compare_port(base, var)
    for arm, info in res["arms"].items():
        nbs = NEIGHBORS.get(arm, ())
        nb_ok = all(res["arms"].get(nb, {}).get("med_d_ret", -1) >= 0 for nb in nbs)
        info["pass"] = bool(info["pass_core"] and nb_ok)
    for tag in ("RB40_ENS5_BASE", "RB40_ENS5_R100K_15", "RB40_ENS5_R100K_20", "RB40_ENS5_R10K_20", "RB40_ENS5_R1K_20"):
        p = rg.out_path(root, tag, 42)
        if p.exists():
            d = rg.daily(p)
            st = rg.stats(d["daily_ret"])
            h1, h2 = d.index < rg.SPLIT, d.index >= rg.SPLIT
            res["prod_ens5"][tag] = {
                "ret": round(st["ret"], 2), "sharpe": round(st["sharpe"], 3), "dd": round(st["dd"], 2),
                "h1": round(rg.stats(d["daily_ret"][h1])["ret"], 2),
                "h2": round(rg.stats(d["daily_ret"][h2])["ret"], 2),
                "r126": round(rg.stats(d["daily_ret"].iloc[-126:])["ret"], 2),
            }
    (root / "summary_rb40.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
