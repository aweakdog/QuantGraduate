"""E7: 异质融合全新 20 种子复核 + 40 种子终审 + 跨点位泛化 (MF_CONFIRM, 2026-11-23 预注册)。

统一冻结快照 ~/quant-research-20261123 (2023-09-20 ~ 2026-11-20, 含 09-05~11-20 真样本外)。
新增 20 个全新不相交种子 FRESH_SEEDS 跑 M0 (DART+L2) 与 HUBER (DART+Huber alpha=0.05),
与首批 SEEDS_20 拼成完整 40 种子面板, 零训练纯回放验证:
  主点位 Q5 (qmt10w: 10万/n5):
    - R10K_20  : 0.80 M0 + 0.20 Ridge_a10k
    - R100K_20 : 0.80 M0 + 0.20 Ridge_a100k
    - BL_H50   : 0.50 M0 + 0.50 HUBER
    - TRI_442  : 0.40 M0 + 0.40 HUBER + 0.20 Ridge_a10k
    - TRI_532  : 0.50 M0 + 0.30 HUBER + 0.20 Ridge_a10k (邻域权重)
  跨点位 (40 种子纯回放, M0 vs R10K_20 vs TRI_442):
    - FYF      : 100万 / n8 (fyf100w 口径 + PG1)
    - S5W      : 5万 / n5 (steady5w 口径)
  生产 5 种子集成 (42,7,123,2024,31337 逐字对照到 2026-11-20):
    - ENS5_M0 / ENS5_R10K_20 / ENS5_R100K_20 / ENS5_TRI_442 / ENS5_TRI_532
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
import model_family_study as mf  # noqa: E402
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402

FRESH_SEEDS = [8, 13, 19, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
ALL_SEEDS_40 = mf.SEEDS_20 + FRESH_SEEDS
PROD_SEEDS = [42, 7, 123, 2024, 31337]

EXTRA_PROFILES = {
    "FYF": {"cap": "1000000", "n": "8"},
    "S5W": {"cap": "50000", "n": "5"},
}


def blend_three(m0_p: Path, hb_p: Path, rg_p: Path, w_m: float, w_h: float, w_r: float, out_p: Path, tag: str):
    """把 HUBER 和 RIDGE 各自按日仿射对齐到 M0 均值/标准差后再按 (w_m, w_h, w_r) 线性加权。"""
    m_obj = pickle.load(open(m0_p, "rb"))
    h_by_d = {dp["date"]: dp for dp in pickle.load(open(hb_p, "rb"))["preds"]}
    r_by_d = {dp["date"]: dp for dp in pickle.load(open(rg_p, "rb"))["preds"]}
    out = []
    for dp in m_obj["preds"]:
        d = dp["date"]
        codes = dp["ranked"]
        pm = np.array(dp["pred_vals"], dtype=np.float64)

        def _align(aux_dp):
            if aux_dp is None or not aux_dp.get("pred_vals"):
                return pm
            am = dict(zip(aux_dp["ranked"], aux_dp["pred_vals"]))
            pa = np.array([am.get(c, np.nan) for c in codes], dtype=np.float64)
            v = np.isfinite(pa) & np.isfinite(pm)
            if v.sum() <= 5:
                return pm
            m_mu, m_sd = float(pm[v].mean()), float(pm[v].std())
            a_mu, a_sd = float(pa[v].mean()), float(pa[v].std())
            if a_sd > 1e-12 and m_sd > 1e-12:
                al = (pa - a_mu) / a_sd * m_sd + m_mu
                return np.where(np.isfinite(al), al, pm)
            return np.full_like(pm, m_mu)

        ph = _align(h_by_d.get(d))
        pr = _align(r_by_d.get(d))
        p_new = w_m * pm + w_h * ph + w_r * pr
        order = np.argsort(-p_new, kind="mergesort")
        ic = dp.get("ic")
        out.append({
            "date": d,
            "ranked": [codes[k] for k in order],
            "pred_vals": [round(float(p_new[k]), 6) for k in order],
            "ic": float("nan") if ic is None else ic,
        })
    meta = dict(m_obj["meta"])
    meta["model"] = tag
    with open(out_p, "wb") as fh:
        pickle.dump({"meta": meta, "preds": out}, fh)


def avg_caches(paths: list[Path], out_p: Path, tag: str):
    objs = [pickle.load(open(p, "rb")) for p in paths]
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
            "ic": float("nan"),
        })
    meta = dict(objs[0]["meta"])
    meta["model"] = tag
    with open(out_p, "wb") as fh:
        pickle.dump({"meta": meta, "preds": preds}, fh)


def out_path_prof(root: Path, tag: str, seed: int, cap: str = "100000") -> Path:
    return root / "data/processed" / f"wf_daily_{tag}_s{seed}_ts{mf.TEST_START}_te{mf.END}_cap{cap}.json"


def run_replays(root: Path, workers: int):
    pdir = root / "data/processed"
    env = mf.run_env(root)
    r10k = pdir / "preds_RIDGE_a10000.pkl"
    r100k = pdir / "preds_RIDGE_a100000.pkl"

    jobs = []  # (tag, cache_name, seed, cap, n_pos)
    for s in ALL_SEEDS_40:
        m0_p = mf.pred_pkl(root, "M0", s)
        hb_p = mf.pred_pkl(root, "HUBER", s)
        if not (m0_p.exists() and hb_p.exists()):
            continue
        # 合成混合缓存 (40 种子全齐)
        c_r10 = pdir / f"preds_CF_R10K_20_s{s}.pkl"
        if not c_r10.exists():
            bp.blend_two(m0_p, r10k, 0.20, c_r10, "cf_r10k_20")
        c_r100 = pdir / f"preds_CF_R100K_20_s{s}.pkl"
        if not c_r100.exists():
            bp.blend_two(m0_p, r100k, 0.20, c_r100, "cf_r100k_20")
        c_h50 = pdir / f"preds_CF_BL_H50_s{s}.pkl"
        if not c_h50.exists():
            bp.blend_two(m0_p, hb_p, 0.50, c_h50, "cf_bl_h50")
        c_t442 = pdir / f"preds_CF_TRI_442_s{s}.pkl"
        if not c_t442.exists():
            blend_three(m0_p, hb_p, r10k, 0.40, 0.40, 0.20, c_t442, "cf_tri_442")
        c_t532 = pdir / f"preds_CF_TRI_532_s{s}.pkl"
        if not c_t532.exists():
            blend_three(m0_p, hb_p, r10k, 0.50, 0.30, 0.20, c_t532, "cf_tri_532")

        # 主点位 Q5 (10万/n5)
        for tag, cname in (
            ("CF_R10K_20", c_r10.name),
            ("CF_R100K_20", c_r100.name),
            ("CF_BL_H50", c_h50.name),
            ("CF_TRI_442", c_t442.name),
            ("CF_TRI_532", c_t532.name),
        ):
            if not out_path_prof(root, tag, s, "100000").exists():
                jobs.append((tag, cname, s, "100000", "5"))

        # 跨点位 FYF (100万/n8) & S5W (5万/n5): M0 基线 + R10K_20 + TRI_442
        for pname, pcfg in EXTRA_PROFILES.items():
            for arm_k, cname in (("M0", m0_p.name), ("R10K_20", c_r10.name), ("TRI_442", c_t442.name)):
                tag = f"CF_{pname}_{arm_k}"
                if not out_path_prof(root, tag, s, pcfg["cap"]).exists():
                    jobs.append((tag, cname, s, pcfg["cap"], pcfg["n"]))

    # 生产 5 种子集成逐字对照 (2023-09-20 ~ 2026-11-20)
    ens_m0 = pdir / "preds_CF_ENS5_M0.pkl"
    ens_hb = pdir / "preds_CF_ENS5_HUBER.pkl"
    if all(mf.pred_pkl(root, "M0", s).exists() and mf.pred_pkl(root, "HUBER", s).exists() for s in PROD_SEEDS):
        if not ens_m0.exists():
            avg_caches([mf.pred_pkl(root, "M0", s) for s in PROD_SEEDS], ens_m0, "cf_ens5_m0")
        if not ens_hb.exists():
            avg_caches([mf.pred_pkl(root, "HUBER", s) for s in PROD_SEEDS], ens_hb, "cf_ens5_huber")
        ens_r10 = pdir / "preds_CF_ENS5_R10K_20.pkl"
        if not ens_r10.exists():
            bp.blend_two(ens_m0, r10k, 0.20, ens_r10, "cf_ens5_r10k_20")
        ens_r100 = pdir / "preds_CF_ENS5_R100K_20.pkl"
        if not ens_r100.exists():
            bp.blend_two(ens_m0, r100k, 0.20, ens_r100, "cf_ens5_r100k_20")
        ens_t442 = pdir / "preds_CF_ENS5_TRI_442.pkl"
        if not ens_t442.exists():
            blend_three(ens_m0, ens_hb, r10k, 0.40, 0.40, 0.20, ens_t442, "cf_ens5_tri_442")
        ens_t532 = pdir / "preds_CF_ENS5_TRI_532.pkl"
        if not ens_t532.exists():
            blend_three(ens_m0, ens_hb, r10k, 0.50, 0.30, 0.20, ens_t532, "cf_ens5_tri_532")
        for tag, cp in (
            ("CF_ENS5_M0", ens_m0), ("CF_ENS5_R10K_20", ens_r10), ("CF_ENS5_R100K_20", ens_r100),
            ("CF_ENS5_TRI_442", ens_t442), ("CF_ENS5_TRI_532", ens_t532),
        ):
            if not out_path_prof(root, tag, 42, "100000").exists():
                jobs.append((tag, cp.name, 42, "100000", "5"))

    print(f"[CF_REPLAY] {len(jobs)} 个纯回放待跑 (workers={workers})", flush=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    def one(j):
        tag, cname, s, cap, n_pos = j
        args = mf.base_args()
        args[args.index("--initial-capital") + 1] = cap
        args[args.index("--tranche-n") + 1] = n_pos
        cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
               *args, "--lgb-seed", str(s), "--tag", tag, "--load-preds", cname]
        log = log_dir / f"{tag.lower()}_s{s}.log"
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        return tag, s, rc

    errs = 0
    with ThreadPoolExecutor(workers) as ex:
        for tag, s, rc in ex.map(one, jobs):
            if rc != 0:
                errs += 1
                print(f"  ERR {tag} s{s}: rc={rc}", flush=True)
    print(f"CF_REPLAY_DONE errs={errs}", flush=True)
    return 1 if errs else 0


def eval_pair(base: dict, var: dict, seeds: list[int]) -> dict:
    cs = [s for s in seeds if s in base and s in var]
    if not cs:
        return {}
    rows = []
    for s in cs:
        b = base[s]["daily_ret"]
        v = var[s]["daily_ret"].reindex(b.index).fillna(0.0)
        sb, sv = rg.stats(b), rg.stats(v)
        h1, h2 = b.index < mf.SPLIT, b.index >= mf.SPLIT
        rows.append({
            "seed": s,
            "ret_v": sv["ret"], "sh_v": sv["sharpe"], "dd_v": sv["dd"],
            "oos_v": rg.stats(v.loc[mf.OOS_START:])["ret"],
            "d_ret": sv["ret"] - sb["ret"], "d_sh": sv["sharpe"] - sb["sharpe"], "d_dd": sv["dd"] - sb["dd"],
            "d_h1": rg.stats(v[h1])["ret"] - rg.stats(b[h1])["ret"],
            "d_h2": rg.stats(v[h2])["ret"] - rg.stats(b[h2])["ret"],
            "d_r126": rg.stats(v.iloc[-126:])["ret"] - rg.stats(b.iloc[-126:])["ret"],
            "d_oos": rg.stats(v.loc[mf.OOS_START:])["ret"] - rg.stats(b.loc[mf.OOS_START:])["ret"],
        })
    t = pd.DataFrame(rows)
    med = t.median(numeric_only=True)
    return {
        "n": len(cs),
        "ret_med": round(float(med["ret_v"]), 2),
        "ret_std": round(float(t["ret_v"].std()), 2),
        "ret_min": round(float(t["ret_v"].min()), 2),
        "sh_med": round(float(med["sh_v"]), 3),
        "dd_med": round(float(med["dd_v"]), 2),
        "oos_med": round(float(med["oos_v"]), 2),
        "d_ret": round(float(med["d_ret"]), 2),
        "pos": int((t["d_ret"] > 0).sum()),
        "d_sh": round(float(med["d_sh"]), 3),
        "d_dd": round(float(med["d_dd"]), 2),
        "d_h1": round(float(med["d_h1"]), 2),
        "d_h2": round(float(med["d_h2"]), 2),
        "d_r126": round(float(med["d_r126"]), 2),
        "d_oos": round(float(med["d_oos"]), 2),
        "oos_pos": int((t["d_oos"] > 0).sum()),
    }


def summarize(root: Path) -> dict:
    base_q5 = {s: rg.daily(mf.out_json(root, "M0", s)) for s in ALL_SEEDS_40 if mf.out_json(root, "M0", s).exists()}
    def _base_dist(dmap, seeds):
        ds = [dmap[s]["daily_ret"] for s in seeds if s in dmap]
        if not ds:
            return {}
        return {
            "n": len(ds),
            "ret_med": round(float(np.median([rg.stats(r)["ret"] for r in ds])), 2),
            "ret_min": round(float(min(rg.stats(r)["ret"] for r in ds)), 2),
            "sh_med": round(float(np.median([rg.stats(r)["sharpe"] for r in ds])), 3),
            "dd_med": round(float(np.median([rg.stats(r)["dd"] for r in ds])), 2),
            "oos_med": round(float(np.median([rg.stats(r.loc[mf.OOS_START:])["ret"] for r in ds])), 2),
        }

    res = {
        "window": f"{mf.TEST_START} ~ {mf.END}",
        "oos_window": f"{mf.OOS_START} ~ {mf.END}",
        "base_Q5": {
            "all40": _base_dist(base_q5, ALL_SEEDS_40),
            "batch1_20": _base_dist(base_q5, mf.SEEDS_20),
            "fresh_20": _base_dist(base_q5, FRESH_SEEDS),
        },
        "Q5": {},
        "cross_profiles": {},
        "prod_ens5": {},
    }

    for arm in ("R10K_20", "R100K_20", "BL_H50", "TRI_442", "TRI_532"):
        var = {s: rg.daily(out_path_prof(root, f"CF_{arm}", s, "100000")) for s in ALL_SEEDS_40
               if out_path_prof(root, f"CF_{arm}", s, "100000").exists()}
        b1 = eval_pair(base_q5, var, mf.SEEDS_20)
        b2 = eval_pair(base_q5, var, FRESH_SEEDS)
        a40 = eval_pair(base_q5, var, ALL_SEEDS_40)
        fresh_pass = bool(
            b2 and b2["n"] == 20 and b2["d_ret"] >= 5.0 and b2["pos"] >= 13
            and b2["d_sh"] >= 0.08 and b2["d_dd"] >= -2.0 and b2["d_r126"] >= -5.0 and b2["d_oos"] >= 0.0
        )
        all40_pass_strict = bool(
            a40 and a40["n"] == 40 and a40["d_ret"] >= 5.0 and a40["pos"] >= 26
            and a40["d_sh"] >= 0.08 and a40["d_dd"] >= -2.0
            and a40["d_h1"] > 0 and a40["d_h2"] > 0
            and b1.get("d_ret", -1) > 0 and b2.get("d_ret", -1) > 0
        )
        res["Q5"][arm] = {
            "all40": a40, "batch1_20": b1, "fresh_20": b2,
            "fresh20_pass": fresh_pass, "all40_pass_strict": all40_pass_strict,
        }

    for pname, pcfg in EXTRA_PROFILES.items():
        bm = {s: rg.daily(out_path_prof(root, f"CF_{pname}_M0", s, pcfg["cap"])) for s in ALL_SEEDS_40
              if out_path_prof(root, f"CF_{pname}_M0", s, pcfg["cap"]).exists()}
        res["cross_profiles"][pname] = {"base_all40": _base_dist(bm, ALL_SEEDS_40)}
        for arm in ("R10K_20", "TRI_442"):
            vm = {s: rg.daily(out_path_prof(root, f"CF_{pname}_{arm}", s, pcfg["cap"])) for s in ALL_SEEDS_40
                  if out_path_prof(root, f"CF_{pname}_{arm}", s, pcfg["cap"]).exists()}
            res["cross_profiles"][pname][arm] = {
                "all40": eval_pair(bm, vm, ALL_SEEDS_40),
                "fresh_20": eval_pair(bm, vm, FRESH_SEEDS),
            }

    for tag in ("CF_ENS5_M0", "CF_ENS5_R10K_20", "CF_ENS5_R100K_20", "CF_ENS5_TRI_442", "CF_ENS5_TRI_532"):
        p = out_path_prof(root, tag, 42, "100000")
        if p.exists():
            d = rg.daily(p)
            st = rg.stats(d["daily_ret"])
            h1, h2 = d.index < mf.SPLIT, d.index >= mf.SPLIT
            res["prod_ens5"][tag] = {
                "ret": round(st["ret"], 2), "sharpe": round(st["sharpe"], 3), "dd": round(st["dd"], 2),
                "h1": round(rg.stats(d["daily_ret"][h1])["ret"], 2),
                "h2": round(rg.stats(d["daily_ret"][h2])["ret"], 2),
                "r126": round(rg.stats(d["daily_ret"].iloc[-126:])["ret"], 2),
                "oos_0905_1120": round(rg.stats(d["daily_ret"].loc[mf.OOS_START:])["ret"], 2),
            }

    (root / "summary_mf_confirm.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["replay", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.phase == "replay":
        return run_replays(root, a.workers)
    summarize(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
