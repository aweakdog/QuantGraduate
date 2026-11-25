"""模型族 / 损失函数 / 异质集成研究编排器 (MF, 2026-11-23 预注册, 见 docs/experiment_board.md)。

全部统一冻结快照 ~/quant-research-20261123 (矩阵与 K 线均到 2026-11-20), qmt10w 现行执行口径:
  10 万 / n5 / 仅主板 / T1B 84 特征 / PG1 (purge6) / FDRR8 / min-pred 0.002 / breadth 40% (confirm 2)。
测试窗: 2023-09-20 ~ 2026-11-20 (含 2026-09-05 ~ 2026-11-20 真样本外 2.5 个月)。
20 种子: 42,7,123,2024,31337,1,2,3,5,11,17,23,55,77,99,202,314,512,888,1234。

臂:
  M0        : 现行基线 (DART + L2, 同夜同矩阵基线, 存预测缓存)
  HUBER     : DART + Huber 损失 (压制 5 日极端肥尾梯度, 存预测缓存)
  WINSOR2   : DART + L2 + 训练标签每日截面 [2%, 98%] 缩尾 (同收益量纲, 存预测缓存)
  GBDT_REG  : 非 DART 强正则 GBDT (n_estimators=200, lr=0.02, reg_lambda=10, min_child_samples=100, 存预测缓存)
  RIDGE     : 确定性异质线性模型 Ridge(alpha=1000, 每日截面秩标准化特征, 2% 缩尾标签)
  BL_R20    : 0.80 M0 + 0.20 RIDGE (按日对齐均值/标准差后混合, 零训练纯回放)
  BL_R30    : 0.70 M0 + 0.30 RIDGE (邻域一致性)
  BL_G30    : 0.70 M0 + 0.30 GBDT_REG (同种子混合, 零训练纯回放)
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blend_preds as bp  # noqa: E402
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402

TEST_START, END = "2023-09-20", "2026-11-20"
SPLIT = "2025-04-21"
OOS_START = "2026-09-05"
SEEDS_20 = [42, 7, 123, 2024, 31337, 1, 2, 3, 5, 11, 17, 23, 55, 77, 99, 202, 314, 512, 888, 1234]

ARM_FLAGS = {
    "M0": [],
    "HUBER": ["--lgb-loss", "huber", "--huber-alpha", "0.05"],
    "WINSOR2": ["--label-winsor", "0.02"],
    "GBDT_REG": ["--boosting-type", "gbdt", "--n-estimators", "200", "--learning-rate", "0.02",
                 "--reg-lambda", "10.0", "--min-child-samples", "100"],
}
GATE = {"ret_pp": 5.0, "pos_min": 13, "sharpe": 0.08, "dd_pp": 2.0, "r126_pp": -5.0}


def base_args():
    return [
        "--train-file", "training_data_pit_v24_tick1.parquet",
        "--pit-universe", "universe_pit.parquet",
        "--features-from", "features_V24PUT_T1B.json",
        "--label", "5d", "--objective", "l2", "--label-alignment", "purge6",
        "--hold-days", "5", "--tranche-n", "5", "--initial-capital", "100000",
        "--portfolio-mode", "periodic", "--exec-mode", "t1close", "--slippage", "0.002",
        "--skip-boards", "30,688",
        "--regime-filter", "breadth", "--regime-ma", "20",
        "--regime-breadth", "0.40", "--regime-confirm", "2",
        "--min-pred", "0.002", "--fill-daily", "--roll-rank", "8",
        "--test-start", TEST_START, "--test-end", END,
    ]


def out_json(root: Path, arm: str, seed: int) -> Path:
    return root / "data/processed" / f"wf_daily_MF_{arm}_s{seed}_ts{TEST_START}_te{END}_cap100000.json"


def pred_pkl(root: Path, arm: str, seed: int) -> Path:
    return root / "data/processed" / f"preds_MF_{arm}_s{seed}.pkl"


def run_env(root: Path) -> dict:
    env = os.environ.copy()
    env["QUANT_DATA_DIR"] = str(root / "data")
    env["QUANT_MODE"] = "backtest"
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    return env


def run_arm(root: Path, arm: str, seeds: list[int], workers: int):
    jobs = [s for s in seeds if not (out_json(root, arm, s).exists() and pred_pkl(root, arm, s).exists())]
    print(f"[{arm}] {len(jobs)}/{len(seeds)} 个种子待跑 (workers={workers})", flush=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    env = run_env(root)

    def one(s: int):
        cmd = [
            "nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
            *base_args(), *ARM_FLAGS[arm],
            "--lgb-seed", str(s),
            "--tag", f"MF_{arm}",
            "--save-preds", pred_pkl(root, arm, s).name,
        ]
        log = log_dir / f"mf_{arm.lower()}_s{s}.log"
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return s, rc

    with ThreadPoolExecutor(workers) as ex:
        for s, rc in ex.map(one, jobs):
            print(f"  MF_{arm} s{s}: rc={rc}", flush=True)


def run_replay(root: Path, tag: str, cache_name: str, seed: int, env: dict) -> int:
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    cmd = [
        "nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
        *base_args(), "--lgb-seed", str(seed),
        "--tag", f"MF_{tag}", "--load-preds", cache_name,
    ]
    log = log_dir / f"mf_{tag.lower()}_s{seed}.log"
    with log.open("w") as fh:
        return subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def run_blends(root: Path, workers: int):
    env = run_env(root)
    pdir = root / "data/processed"
    # 1. 纯 Ridge 独立回放 (alpha 300/1000/3000/10000)
    ridge_jobs = []
    for al in (300, 1000, 3000, 10000):
        cname = f"preds_RIDGE_a{al}.pkl"
        if (pdir / cname).exists() and not out_json(root, f"RIDGE_a{al}", 42).exists():
            ridge_jobs.append((f"RIDGE_a{al}", cname, 42))

    # 2. 合成混合缓存 & 排队回放
    corr_stats = {"BL_R20": [], "BL_R30": [], "BL_R100K_20": [], "BL_R100K_30": [], "BL_G30": [], "BL_H50": []}
    blend_jobs = []
    ridge_main = pdir / "preds_RIDGE_a1000.pkl"
    ridge_hi = pdir / "preds_RIDGE_a100000.pkl"
    for s in SEEDS_20:
        m0_p = pred_pkl(root, "M0", s)
        if not m0_p.exists():
            continue
        if ridge_main.exists():
            for tag, w in (("BL_R20", 0.20), ("BL_R30", 0.30)):
                out_c = pdir / f"preds_MF_{tag}_s{s}.pkl"
                if not out_c.exists():
                    info = bp.blend_two(m0_p, ridge_main, w, out_c, f"blend_dart_ridge_{int(w*100)}")
                    if info["corr_mean"] is not None:
                        corr_stats[tag].append(info["corr_mean"])
                if not out_json(root, tag, s).exists():
                    blend_jobs.append((tag, out_c.name, s))
        if ridge_hi.exists():
            for tag, w in (("BL_R100K_20", 0.20), ("BL_R100K_30", 0.30)):
                out_c = pdir / f"preds_MF_{tag}_s{s}.pkl"
                if not out_c.exists():
                    info = bp.blend_two(m0_p, ridge_hi, w, out_c, f"blend_dart_ridge100k_{int(w*100)}")
                    if info["corr_mean"] is not None:
                        corr_stats[tag].append(info["corr_mean"])
                if not out_json(root, tag, s).exists():
                    blend_jobs.append((tag, out_c.name, s))
        gbdt_p = pred_pkl(root, "GBDT_REG", s)
        if gbdt_p.exists():
            out_c = pdir / f"preds_MF_BL_G30_s{s}.pkl"
            if not out_c.exists():
                info = bp.blend_two(m0_p, gbdt_p, 0.30, out_c, "blend_dart_gbdt_30")
                if info["corr_mean"] is not None:
                    corr_stats["BL_G30"].append(info["corr_mean"])
            if not out_json(root, "BL_G30", s).exists():
                blend_jobs.append(("BL_G30", out_c.name, s))
        huber_p = pred_pkl(root, "HUBER", s)
        if huber_p.exists():
            out_c = pdir / f"preds_MF_BL_H50_s{s}.pkl"
            if not out_c.exists():
                info = bp.blend_two(m0_p, huber_p, 0.50, out_c, "blend_dart_huber_50")
                if info["corr_mean"] is not None:
                    corr_stats["BL_H50"].append(info["corr_mean"])
            if not out_json(root, "BL_H50", s).exists():
                blend_jobs.append(("BL_H50", out_c.name, s))

    all_jobs = ridge_jobs + blend_jobs
    print(f"[BLEND/RIDGE] {len(all_jobs)} 个纯回放待跑 (workers={workers})", flush=True)
    if any(corr_stats.values()):
        (root / "blend_corr.json").write_text(json.dumps(
            {k: round(float(np.mean(v)), 4) for k, v in corr_stats.items() if v}, indent=1))

    with ThreadPoolExecutor(workers) as ex:
        for (tag, _, s), rc in zip(all_jobs, ex.map(lambda j: run_replay(root, j[0], j[1], j[2], env), all_jobs)):
            print(f"  MF_{tag} s{s}: rc={rc}", flush=True)


def rank_buckets_for_arm(root: Path, arm: str, y_lookup: dict, seeds: list[int]) -> dict:
    r1s, r25s, r68s, ics = [], [], [], []
    for s in seeds:
        p = pred_pkl(root, arm, s)
        if not p.exists():
            continue
        dps = pickle.load(open(p, "rb"))["preds"]
        d1, d25, d68, dic = [], [], [], []
        for dp in dps:
            d = dp["date"]
            if dp.get("ic") is not None and np.isfinite(dp["ic"]):
                dic.append(float(dp["ic"]))
            ys = [y_lookup.get((d, c), np.nan) for c in dp["ranked"][:8]]
            if len(ys) >= 8:
                if np.isfinite(ys[0]):
                    d1.append(ys[0])
                v25 = [x for x in ys[1:5] if np.isfinite(x)]
                if v25:
                    d25.append(float(np.mean(v25)))
                v68 = [x for x in ys[5:8] if np.isfinite(x)]
                if v68:
                    d68.append(float(np.mean(v68)))
        if d1:
            r1s.append(float(np.mean(d1)) * 100)
            r25s.append(float(np.mean(d25)) * 100)
            r68s.append(float(np.mean(d68)) * 100)
            ics.append(float(np.mean(dic)))
    if not r1s:
        return {}
    return {
        "ic_med": round(float(np.median(ics)), 4),
        "r1_excess_pct": round(float(np.median(r1s)), 3),
        "r2_5_excess_pct": round(float(np.median(r25s)), 3),
        "r6_8_excess_pct": round(float(np.median(r68s)), 3),
    }


def summarize(root: Path) -> dict:
    base = {s: rg.daily(out_json(root, "M0", s)) for s in SEEDS_20 if out_json(root, "M0", s).exists()}
    if not base:
        return {"error": "no M0 baseline outputs"}

    # 加载 y_target 供排名分层诊断
    y_lookup = {}
    exp_m = root / "data/processed/mf_export_matrix.parquet"
    if exp_m.exists():
        mdf = pd.read_parquet(exp_m, columns=["date", "code", "y_target"])
        mdf["date"] = pd.to_datetime(mdf["date"])
        mdf["code"] = mdf["code"].astype(str)
        y_lookup = dict(zip(zip(mdf["date"], mdf["code"]), mdf["y_target"]))

    def _dist(dmap):
        rets = [rg.stats(d["daily_ret"])["ret"] for d in dmap.values()]
        sh = [rg.stats(d["daily_ret"])["sharpe"] for d in dmap.values()]
        dds = [rg.stats(d["daily_ret"])["dd"] for d in dmap.values()]
        oos = [rg.stats(d["daily_ret"].loc[OOS_START:])["ret"] for d in dmap.values()]
        return {
            "n": len(dmap),
            "ret_med": round(float(np.median(rets)), 2),
            "ret_std": round(float(np.std(rets)), 2),
            "ret_min": round(float(np.min(rets)), 2),
            "ret_max": round(float(np.max(rets)), 2),
            "sharpe_med": round(float(np.median(sh)), 3),
            "dd_med": round(float(np.median(dds)), 2),
            "oos_0905_1120_med": round(float(np.median(oos)), 2),
        }

    res = {
        "window": f"{TEST_START} ~ {END}",
        "oos_window": f"{OOS_START} ~ {END}",
        "M0": _dist(base) | (rank_buckets_for_arm(root, "M0", y_lookup, SEEDS_20) if y_lookup else {}),
    }

    # 纯 Ridge 单点结果
    for al in (300, 1000, 3000, 10000):
        p = out_json(root, f"RIDGE_a{al}", 42)
        if p.exists():
            d = rg.daily(p)
            st = rg.stats(d["daily_ret"])
            oos_r = rg.stats(d["daily_ret"].loc[OOS_START:])["ret"]
            res[f"RIDGE_a{al}"] = {
                "ret": round(st["ret"], 2), "sharpe": round(st["sharpe"], 3),
                "dd": round(st["dd"], 2), "oos_0905_1120": round(oos_r, 2),
            }

    for arm in ("HUBER", "WINSOR2", "GBDT_REG", "BL_R20", "BL_R30", "BL_R100K_20", "BL_R100K_30", "BL_G30", "BL_H50"):
        var = {s: rg.daily(out_json(root, arm, s)) for s in SEEDS_20 if out_json(root, arm, s).exists()}
        if not var:
            continue
        seeds = sorted(set(base) & set(var))
        rows = []
        for s in seeds:
            b = base[s]["daily_ret"]
            v = var[s]["daily_ret"].reindex(b.index).fillna(0.0)
            sb, sv = rg.stats(b), rg.stats(v)
            h1, h2 = b.index < SPLIT, b.index >= SPLIT
            r126 = b.index[-126:]
            rows.append({
                "seed": s,
                "d_ret": sv["ret"] - sb["ret"],
                "d_sharpe": sv["sharpe"] - sb["sharpe"],
                "d_dd": sv["dd"] - sb["dd"],
                "d_h1": rg.stats(v[h1])["ret"] - rg.stats(b[h1])["ret"],
                "d_h2": rg.stats(v[h2])["ret"] - rg.stats(b[h2])["ret"],
                "d_r126": rg.stats(v.loc[r126])["ret"] - rg.stats(b.loc[r126])["ret"],
                "d_oos": rg.stats(v.loc[OOS_START:])["ret"] - rg.stats(b.loc[OOS_START:])["ret"],
            })
        t = pd.DataFrame(rows)
        med = t.median(numeric_only=True)
        info = _dist(var) | {
            "paired_n": len(seeds),
            "med_d_ret": round(float(med["d_ret"]), 2),
            "pos": int((t["d_ret"] > 0).sum()),
            "med_d_sharpe": round(float(med["d_sharpe"]), 3),
            "med_d_dd": round(float(med["d_dd"]), 2),
            "med_d_h1": round(float(med["d_h1"]), 2),
            "med_d_h2": round(float(med["d_h2"]), 2),
            "med_d_r126": round(float(med["d_r126"]), 2),
            "med_d_oos": round(float(med["d_oos"]), 2),
            "oos_pos": int((t["d_oos"] > 0).sum()),
        }
        info["pass_gate"] = bool(
            info["paired_n"] == 20
            and info["med_d_ret"] >= GATE["ret_pp"]
            and info["pos"] >= GATE["pos_min"]
            and info["med_d_sharpe"] >= GATE["sharpe"]
            and info["med_d_dd"] >= -GATE["dd_pp"]
            and info["med_d_h1"] > 0 and info["med_d_h2"] > 0
            and info["med_d_r126"] >= GATE["r126_pp"]
        )
        if y_lookup and arm in ("HUBER", "WINSOR2", "GBDT_REG", "BL_R20", "BL_R30", "BL_G30", "BL_H50"):
            info |= rank_buckets_for_arm(root, arm, y_lookup, seeds)
        res[arm] = info

    (root / "summary_mf.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run-arm", "run-blend", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", choices=list(ARM_FLAGS), default="M0")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_20))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.phase == "run-arm":
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        run_arm(root, a.arm, seeds, a.workers)
    elif a.phase == "run-blend":
        run_blends(root, a.workers)
    else:
        print(json.dumps(summarize(root), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
