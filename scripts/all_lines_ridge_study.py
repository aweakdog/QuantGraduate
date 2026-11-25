"""全站 12 条线最优配置验证脚本 (ALR, 2026-11-23)。

两个子任务:
  1. adapt-check: 在 ~/quant-research-20261123 已有 40 种子 M0 (T1B+PG1) 缓存上,
     对比 固定 alpha=10000 vs 样本数自适应 alpha = c * N (c=0.03, 0.05, 0.10, 0.25),
     覆盖 Q5 / S5W / FYF 40 种子 + 生产 5 种子集成 ENS5。
  2. run-fam / eval-fams: 验证剩余三个模型族 (各 20 种子, 同 2026-11-20 冻结快照):
       - T1A_MB : 主板 + T1A 84 特征 (legacy) -> 验证 aggr10w (10万/n3+cap2) / aggr5w (5万/n3+cap2)
       - B80_MB : 主板 + 基线 80 特征 (legacy) -> 验证 aggr2w (2万/n2+lot-flex) / bench10m (1000万/n8)
       - B80_FM : 全市场 + 基线 80 特征 (legacy) -> 验证 steady2w (2万/n3+cap2+lot-flex) / bench10m_fm (1000万/n8)
     每个族产出基线 20 种子 preds 缓存, 然后用共用纯函数 ridge_blend.py 在同快照矩阵上
     跑确定性 Ridge 并合成 20% 混合, 按各线逐字生产配置做 20 种子配对重放!
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blend_preds as bp  # noqa: E402
import mf_confirm_study as cf  # noqa: E402
import model_family_study as mf  # noqa: E402
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402
import ridge_blend as rb  # noqa: E402

B80_FEAT = "wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json"

FAMILIES = {
    "T1A_MB": {
        "features_from": "features_V24PUT_T1A.json",
        "skip_boards": "30,688",
        "align": "legacy",
        "horizon": 5,
        "train_cap": "100000", "train_n": "3", "train_extra": ["--ind-cap", "2"],
        "eval_points": {
            "aggr10w": {"cap": "100000", "n": "3", "extra": ["--ind-cap", "2"]},
            "aggr5w": {"cap": "50000", "n": "3", "extra": ["--ind-cap", "2"]},
        },
    },
    "B80_MB": {
        "features_from": B80_FEAT,
        "skip_boards": "30,688",
        "align": "legacy",
        "horizon": 5,
        "train_cap": "10000000", "train_n": "8", "train_extra": [],
        "eval_points": {
            "bench10m": {"cap": "10000000", "n": "8", "extra": []},
            "aggr2w": {"cap": "20000", "n": "2", "extra": ["--lot-flex", "0.5"]},
        },
    },
    "B80_FM": {
        "features_from": B80_FEAT,
        "skip_boards": "",
        "align": "legacy",
        "horizon": 5,
        "train_cap": "10000000", "train_n": "8", "train_extra": [],
        "eval_points": {
            "bench10m_fm": {"cap": "10000000", "n": "8", "extra": []},
            "steady2w": {"cap": "20000", "n": "3", "extra": ["--lot-flex", "0.5", "--ind-cap", "2"]},
        },
    },
}


def fam_args(fam_name: str, cap: str, n_pos: str, extra: list[str]) -> list[str]:
    cfg = FAMILIES[fam_name]
    args = [
        "--train-file", "training_data_pit_v24_tick1.parquet",
        "--pit-universe", "universe_pit.parquet",
        "--features-from", cfg["features_from"],
        "--label", "5d", "--objective", "l2", "--label-alignment", cfg["align"],
        "--hold-days", "5", "--tranche-n", n_pos, "--initial-capital", cap,
        "--portfolio-mode", "periodic", "--exec-mode", "t1close", "--slippage", "0.002",
        "--regime-filter", "breadth", "--regime-ma", "20",
        "--regime-breadth", "0.40", "--regime-confirm", "2",
        "--min-pred", "0.002", "--fill-daily", "--roll-rank", "8",
        "--test-start", mf.TEST_START, "--test-end", mf.END,
    ]
    if cfg["skip_boards"]:
        args += ["--skip-boards", cfg["skip_boards"]]
    args += list(extra)
    return args


def adapt_check(root: Path, workers: int):
    pdir = root / "data/processed"
    env = mf.run_env(root)
    mat_p = pdir / "mf_export_matrix.parquet"
    meta_in = json.loads(Path(str(mat_p) + ".meta.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(mat_p)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)

    cfgs = [("AR_003", 0.0, 0.03), ("AR_005", 0.0, 0.05), ("AR_010", 0.0, 0.10), ("AR_025", 0.0, 0.25)]
    print("运行自适应 alpha=c*N 走窗 Ridge (4 档, 共用 ridge_blend.py)...", flush=True)
    wf = rb.walkforward_ridge_preds(
        df, meta_in["features"], meta_in["label_raw"], meta_in["label"],
        meta_in["test_start"], meta_in["test_end"], int(meta_in["label_horizon"]), int(meta_in["min_train_days"]),
        cfgs, winsor=0.02,
    )
    meta_base = {
        "train_file": meta_in["train_file"], "pit_universe": meta_in["pit_universe"],
        "label": "5d", "objective": "l2", "label_alignment": "purge6", "label_horizon": int(meta_in["label_horizon"]),
        "label_panel_sha256": None, "selected_features": meta_in["features"],
        "train_start": None, "test_start": meta_in["test_start"], "test_end": meta_in["test_end"],
        "neutralize_style": False, "n_features": 80, "exclude_feats": "",
        "n_selected": len(meta_in["features"]), "feat_cutoff": meta_in["first_pred"],
    }
    for tag, _, ar in cfgs:
        cp = pdir / f"preds_RIDGE_{tag}.pkl"
        m = dict(meta_base, model=f"ridge_{tag.lower()}")
        pickle.dump({"meta": m, "preds": wf[tag]}, open(cp, "wb"))
        ics = [dp["ic"] for dp in wf[tag] if np.isfinite(dp["ic"])]
        print(f"  {tag} (c={ar}): days={len(wf[tag])} IC={np.mean(ics):+.4f}", flush=True)

    # 合成 40 种子 + ENS5 混合缓存并回放
    jobs = []
    ens_m0 = pdir / "preds_CF_ENS5_M0.pkl"
    for tag, _, _ in cfgs:
        rp = pdir / f"preds_RIDGE_{tag}.pkl"
        for s in cf.ALL_SEEDS_40:
            m0_p = mf.pred_pkl(root, "M0", s)
            out_c = pdir / f"preds_AD_{tag}_s{s}.pkl"
            if not out_c.exists():
                bp.blend_two(m0_p, rp, 0.20, out_c, f"ad_{tag.lower()}")
            if not cf.out_path_prof(root, f"AD_{tag}", s, "100000").exists():
                jobs.append((f"AD_{tag}", out_c.name, s, "100000", "5"))
        # ENS5
        ec = pdir / f"preds_AD_ENS5_{tag}.pkl"
        if not ec.exists():
            bp.blend_two(ens_m0, rp, 0.20, ec, f"ad_ens5_{tag.lower()}")
        if not cf.out_path_prof(root, f"AD_ENS5_{tag}", 42, "100000").exists():
            jobs.append((f"AD_ENS5_{tag}", ec.name, 42, "100000", "5"))

    print(f"[ADAPT] {len(jobs)} 个纯回放待跑 (workers={workers})", flush=True)

    def one(j):
        tag, cname, s, cap, n_pos = j
        args = mf.base_args()
        args[args.index("--initial-capital") + 1] = cap
        args[args.index("--tranche-n") + 1] = n_pos
        cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
               *args, "--lgb-seed", str(s), "--tag", tag, "--load-preds", cname]
        log = root / "logs" / f"{tag.lower()}_s{s}.log"
        with log.open("w") as fh:
            return tag, s, subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(workers) as ex:
        errs = sum(1 for _, _, rc in ex.map(one, jobs) if rc != 0)
    print(f"ADAPT_REPLAY_DONE errs={errs}", flush=True)

    base_q5 = {s: rg.daily(mf.out_json(root, "M0", s)) for s in cf.ALL_SEEDS_40}
    res = {"fixed_R10K_20": {}, "adapt": {}, "ens5": {}}
    r10 = {s: rg.daily(cf.out_path_prof(root, "CF_R10K_20", s, "100000")) for s in cf.ALL_SEEDS_40}
    res["fixed_R10K_20"] = {
        "all40": cf.eval_pair(base_q5, r10, cf.ALL_SEEDS_40),
        "fresh20": cf.eval_pair(base_q5, r10, cf.FRESH_SEEDS),
    }
    for tag, _, _ in cfgs:
        vm = {s: rg.daily(cf.out_path_prof(root, f"AD_{tag}", s, "100000")) for s in cf.ALL_SEEDS_40}
        res["adapt"][tag] = {
            "all40": cf.eval_pair(base_q5, vm, cf.ALL_SEEDS_40),
            "fresh20": cf.eval_pair(base_q5, vm, cf.FRESH_SEEDS),
        }
        d = rg.daily(cf.out_path_prof(root, f"AD_ENS5_{tag}", 42, "100000"))
        st = rg.stats(d["daily_ret"])
        h1, h2 = d.index < mf.SPLIT, d.index >= mf.SPLIT
        res["ens5"][tag] = {
            "ret": round(st["ret"], 2), "sharpe": round(st["sharpe"], 3), "dd": round(st["dd"], 2),
            "h1": round(rg.stats(d["daily_ret"][h1])["ret"], 2),
            "h2": round(rg.stats(d["daily_ret"][h2])["ret"], 2),
            "oos": round(rg.stats(d["daily_ret"].loc[mf.OOS_START:])["ret"], 2),
        }
    (root / "summary_adapt_alpha.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def run_fam(root: Path, fam: str, workers: int):
    cfg = FAMILIES[fam]
    env = mf.run_env(root)
    pdir = root / "data/processed"
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    jobs = [s for s in mf.SEEDS_20 if not (pdir / f"preds_ALR_{fam}_s{s}.pkl").exists()]
    print(f"[ALR {fam}] {len(jobs)}/20 个种子待训 (workers={workers})", flush=True)

    def one(s: int):
        args = fam_args(fam, cfg["train_cap"], cfg["train_n"], cfg["train_extra"])
        cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
               *args, "--lgb-seed", str(s), "--tag", f"ALR_{fam}_BASE",
               "--save-preds", f"preds_ALR_{fam}_s{s}.pkl"]
        log = log_dir / f"alr_{fam.lower()}_s{s}.log"
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return s, rc

    with ThreadPoolExecutor(workers) as ex:
        for s, rc in ex.map(one, jobs):
            print(f"  ALR_{fam} s{s}: rc={rc}", flush=True)


def eval_fams(root: Path, alpha_ratio: float, fixed_alpha: float, workers: int):
    pdir = root / "data/processed"
    env = mf.run_env(root)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    summary = {"alpha_ratio": alpha_ratio, "fixed_alpha": fixed_alpha, "families": {}}

    for fam, cfg in FAMILIES.items():
        if not all((pdir / f"preds_ALR_{fam}_s{s}.pkl").exists() for s in mf.SEEDS_20):
            print(f"[SKIP {fam}] preds incomplete", flush=True)
            continue
        # 1. 导出该族矩阵 + 训同口径确定性 Ridge
        mat_p = pdir / f"alr_matrix_{fam}.parquet"
        if not mat_p.exists():
            args = fam_args(fam, cfg["train_cap"], cfg["train_n"], cfg["train_extra"])
            subprocess.run([sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
                            *args, "--export-matrix", mat_p.name], cwd=root, env=env, check=True)
        meta_in = json.loads(Path(str(mat_p) + ".meta.json").read_text(encoding="utf-8"))
        df = pd.read_parquet(mat_p)
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str)
        r_tag = f"ALR_RIDGE_{fam}"
        rp = pdir / f"preds_{r_tag}.pkl"
        if not rp.exists():
            wf = rb.walkforward_ridge_preds(
                df, meta_in["features"], meta_in["label_raw"], meta_in["label"],
                meta_in["test_start"], meta_in["test_end"], int(meta_in["label_horizon"]),
                int(meta_in["min_train_days"]), [(r_tag, fixed_alpha, alpha_ratio)], winsor=0.02,
            )
            meta_r = {
                "model": r_tag.lower(),
                "train_file": meta_in["train_file"], "pit_universe": meta_in["pit_universe"],
                "label": "5d", "objective": "l2", "label_alignment": cfg["align"],
                "label_horizon": int(meta_in["label_horizon"]), "label_panel_sha256": None,
                "selected_features": meta_in["features"], "train_start": None,
                "test_start": meta_in["test_start"], "test_end": meta_in["test_end"],
                "neutralize_style": False, "n_features": 80, "exclude_feats": "",
                "n_selected": len(meta_in["features"]), "feat_cutoff": meta_in["first_pred"],
            }
            pickle.dump({"meta": meta_r, "preds": wf[r_tag]}, open(rp, "wb"))
            ics = [dp["ic"] for dp in wf[r_tag] if np.isfinite(dp["ic"])]
            print(f"[{fam}] Ridge IC={np.mean(ics):+.4f}", flush=True)

        # 2. 合成 20 种子混合缓存 + ENS5 缓存, 逐点回放 (base vs blend20)
        jobs = []
        corrs = []
        for s in mf.SEEDS_20:
            b_c = pdir / f"preds_ALR_{fam}_s{s}.pkl"
            m_c = pdir / f"preds_ALR_{fam}_R20_s{s}.pkl"
            if not m_c.exists():
                info = bp.blend_two(b_c, rp, 0.20, m_c, f"alr_{fam.lower()}_r20")
                if info["corr_mean"] is not None:
                    corrs.append(info["corr_mean"])
            for pt_name, pt in cfg["eval_points"].items():
                for kind, cname in (("BASE", b_c.name), ("R20", m_c.name)):
                    tag = f"ALR_{pt_name}_{kind}"
                    if not cf.out_path_prof(root, tag, s, pt["cap"]).exists():
                        jobs.append((fam, pt["cap"], pt["n"], pt["extra"], tag, cname, s))
        # ENS5
        e_base = pdir / f"preds_ALR_{fam}_ENS5_BASE.pkl"
        e_r20 = pdir / f"preds_ALR_{fam}_ENS5_R20.pkl"
        if not e_base.exists():
            cf.avg_caches([pdir / f"preds_ALR_{fam}_s{s}.pkl" for s in cf.PROD_SEEDS], e_base, f"alr_{fam.lower()}_ens5")
        if not e_r20.exists():
            bp.blend_two(e_base, rp, 0.20, e_r20, f"alr_{fam.lower()}_ens5_r20")
        for pt_name, pt in cfg["eval_points"].items():
            for kind, cp in (("ENS5_BASE", e_base), ("ENS5_R20", e_r20)):
                tag = f"ALR_{pt_name}_{kind}"
                if not cf.out_path_prof(root, tag, 42, pt["cap"]).exists():
                    jobs.append((fam, pt["cap"], pt["n"], pt["extra"], tag, cp.name, 42))

        print(f"[{fam}] {len(jobs)} 个纯回放待跑 (workers={workers})", flush=True)

        def one(j):
            fn, cap, n_pos, extra, tag, cname, s = j
            args = fam_args(fn, cap, n_pos, extra)
            cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
                   *args, "--lgb-seed", str(s), "--tag", tag, "--load-preds", cname]
            log = log_dir / f"{tag.lower()}_s{s}.log"
            with log.open("w") as fh:
                return tag, s, subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode

        with ThreadPoolExecutor(workers) as ex:
            errs = sum(1 for _, _, rc in ex.map(one, jobs) if rc != 0)
        print(f"[{fam}] replays done errs={errs}", flush=True)

        f_out = {"pred_corr": round(float(np.mean(corrs)), 4) if corrs else None, "points": {}}
        for pt_name, pt in cfg["eval_points"].items():
            bm = {s: rg.daily(cf.out_path_prof(root, f"ALR_{pt_name}_BASE", s, pt["cap"])) for s in mf.SEEDS_20}
            vm = {s: rg.daily(cf.out_path_prof(root, f"ALR_{pt_name}_R20", s, pt["cap"])) for s in mf.SEEDS_20}
            b_rets = [rg.stats(d["daily_ret"])["ret"] for d in bm.values()]
            b_sh = [rg.stats(d["daily_ret"])["sharpe"] for d in bm.values()]
            b_dd = [rg.stats(d["daily_ret"])["dd"] for d in bm.values()]
            b_oos = [rg.stats(d["daily_ret"].loc[mf.OOS_START:])["ret"] for d in bm.values()]
            pair = cf.eval_pair(bm, vm, mf.SEEDS_20)
            eb = rg.stats(rg.daily(cf.out_path_prof(root, f"ALR_{pt_name}_ENS5_BASE", 42, pt["cap"]))["daily_ret"])
            er = rg.stats(rg.daily(cf.out_path_prof(root, f"ALR_{pt_name}_ENS5_R20", 42, pt["cap"]))["daily_ret"])
            f_out["points"][pt_name] = {
                "base_med": {"ret": round(float(np.median(b_rets)), 2), "sh": round(float(np.median(b_sh)), 3),
                             "dd": round(float(np.median(b_dd)), 2), "oos": round(float(np.median(b_oos)), 2)},
                "r20_vs_base": pair,
                "ens5_base": {k: round(v, 2) for k, v in eb.items()},
                "ens5_r20": {k: round(v, 2) for k, v in er.items()},
            }
        summary["families"][fam] = f_out

    (root / "summary_all_lines_ridge.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["adapt-check", "run-fam", "eval-fams"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--fam", choices=list(FAMILIES), default="T1A_MB")
    ap.add_argument("--alpha-ratio", type=float, default=0.05)
    ap.add_argument("--fixed-alpha", type=float, default=10000.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.phase == "adapt-check":
        adapt_check(root, a.workers)
    elif a.phase == "run-fam":
        run_fam(root, a.fam, a.workers)
    else:
        eval_fams(root, a.alpha_ratio, a.fixed_alpha, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
