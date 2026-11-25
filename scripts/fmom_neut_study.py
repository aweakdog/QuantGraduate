"""E5: 因子动量中性化因果检验 (FMOM_NEUT, 2026-11-23, 接 E4 PG1_MECH 发现)。纯回放、零训练。

E4 证实 PG1 (purge6) 使模型预测与「上周刚奏效的 76 维因子收益方向 fmom_score(D)」的相关性
在 40/40 种子上系统下降 (+0.1683 -> +0.1234, 降幅 26.7%)。
本实验做直接因果干预 (对照 N6-B):
  在信号日 D 上, 令 f = rank_pct(fmom_score(D)) - 0.5, 对预测 pred 做 OLS:
      pred' = pred - k * beta * f
  1) 在 LEGACY (CURRENT) 40 种子缓存上试 k=0.27 (对齐 PG1 降幅) 与 k=0.50: 看能否复现 PG1 增益?
  2) 在 PURGE6 40 种子缓存上试 k=0.27 与 k=0.50: 看在 PG1 基础上继续压低因子动量是否有增量?
全套复用 041 ~/quant-research-20260912-n2 的 40 对预测缓存, 严格同口径回放。
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import port_opt_study as po  # noqa: E402
import proctitle  # noqa: E402
import regime_gate_study as rg  # noqa: E402
import run_overnight_research as ro  # noqa: E402

ARMS = {
    "LEG_FM27": ("legacy", 0.27),
    "LEG_FM50": ("legacy", 0.50),
    "PG_FM27": ("purge6", 0.27),
    "PG_FM50": ("purge6", 0.50),
}


def build_fmom_scores(root: Path) -> dict:
    mat_path = root / "data/processed/training_data_pit_v24_tick1.parquet"
    uni_path = root / "data/universe/universe_pit.parquet"
    feat_path = root / "data/processed/features_V24PUT_T1B.json"
    feats_all = json.loads(feat_path.read_text(encoding="utf-8"))["selected_features"]

    df = pd.read_parquet(mat_path)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    df = df[~df["code"].str.startswith(("30", "688"))]
    u = pd.read_parquet(uni_path)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    u["code6"] = u["code"].astype(str).str.zfill(6)
    eff = pd.DatetimeIndex(sorted(u["effective_date"].unique()))
    members = {d: set(g["code6"]) for d, g in u.groupby("effective_date")}
    code6 = df["code"].str[:6]
    period = eff.searchsorted(pd.DatetimeIndex(df["date"]), side="right") - 1
    keep = np.zeros(len(df), dtype=bool)
    for i, d in enumerate(eff):
        m = period == i
        if m.any():
            keep[m] = code6[m].isin(members[pd.Timestamp(d)]).values
    df = df[keep].reset_index(drop=True)

    feats = [f for f in feats_all if f in df.columns]
    df["y"] = df.groupby("date")["fwd_5d_ret"].transform(lambda x: x - x.mean())
    all_dates = sorted(df["date"].unique())

    by_day, day_ic = {}, {}
    for d, g in df.groupby("date", sort=True):
        X = (g[feats].rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)
        by_day[d] = (g["code"].to_numpy(), X)
        m = np.isfinite(y)
        if m.sum() > 20:
            yc = y[m] - y[m].mean()
            yn = np.linalg.norm(yc)
            Xm = X[m] - X[m].mean(axis=0, keepdims=True)
            xn = np.linalg.norm(Xm, axis=0)
            day_ic[d] = np.where((xn > 1e-12) & (yn > 1e-12), (Xm.T @ yc) / (xn * yn + 1e-12), 0.0)

    fmom = {}
    for gpos, d in enumerate(all_dates):
        if gpos < 6:
            continue
        gv = day_ic.get(all_dates[gpos - 6])
        if gv is None:
            continue
        codes, X = by_day[d]
        sc = pd.Series(X @ gv, index=codes)
        rk = sc.rank(pct=True) - 0.5
        fmom[d] = rk.to_dict()
    return fmom


def neutralize_cache(src: Path, dst: Path, fmom: dict, k: float, model_tag: str) -> tuple[float, float]:
    obj = pickle.load(open(src, "rb"))
    out_preds = []
    eb, ea = [], []
    for dp in obj["preds"]:
        d = dp["date"]
        fm = fmom.get(d)
        pv = dp.get("pred_vals")
        if fm is None or not pv:
            out_preds.append(dp)
            continue
        codes = dp["ranked"]
        p = np.array(pv, dtype=np.float64)
        f = np.array([fm.get(c, 0.0) for c in codes], dtype=np.float64)
        m = np.isfinite(p) & np.isfinite(f)
        if m.sum() > 20 and np.var(f[m]) > 1e-12:
            fc = f[m] - f[m].mean()
            beta = float(np.dot(fc, p[m] - p[m].mean()) / np.dot(fc, fc))
            p_new = p - k * beta * (f - f[m].mean())
            cb = spearmanr(p[m], f[m])[0]
            ca = spearmanr(p_new[m], f[m])[0]
            if np.isfinite(cb) and np.isfinite(ca):
                eb.append(float(cb))
                ea.append(float(ca))
        else:
            p_new = p
        order = np.argsort(-p_new, kind="mergesort")
        out_preds.append({
            "date": d,
            "ranked": [codes[idx] for idx in order],
            "pred_vals": [round(float(p_new[idx]), 6) for idx in order],
            "ic": dp.get("ic"),
        })
    meta = dict(obj["meta"])
    meta["model"] = model_tag
    with open(dst, "wb") as fh:
        pickle.dump({"meta": meta, "preds": out_preds}, fh)
    return (float(np.mean(eb)) if eb else 0.0), (float(np.mean(ea)) if ea else 0.0)


def main(argv=None):
    proctitle.lowkey("mltask/grid")
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    root = Path(a.root)
    pdir = root / "data/processed"

    if a.phase == "run":
        print("构建上周因子动量面板...", flush=True)
        fmom = build_fmom_scores(root)
        env = os.environ.copy()
        env["QUANT_DATA_DIR"] = str(root / "data")
        env["QUANT_MODE"] = "backtest"
        jobs = []
        expo_log = {}
        # 同时确保有 LEGACY Q5 基线 (如果还没有 N7_Q5_LEG, 一并回放)
        for s in ro.N6_SEEDS:
            leg_c, pg_c, _ = ro.n6_caches(s)
            if not rg.out_path(root, "FM_LEG_BASE", s).exists():
                jobs.append(("FM_LEG_BASE", "legacy", leg_c, s))
            for arm, (src_kind, k) in ARMS.items():
                src_c = leg_c if src_kind == "legacy" else pg_c
                dst_c = f"preds_FM_{arm}_s{s}.pkl"
                if not (pdir / dst_c).exists():
                    b, a_v = neutralize_cache(pdir / src_c, pdir / dst_c, fmom, k, f"fmom_{arm}")
                    expo_log.setdefault(arm, []).append((b, a_v))
                if not rg.out_path(root, f"FM_{arm}", s).exists():
                    jobs.append((f"FM_{arm}", src_kind, dst_c, s))
        if expo_log:
            for arm, pairs in expo_log.items():
                print(f"  {arm} expo(pred, fmom): {np.mean([x[0] for x in pairs]):+.4f} -> {np.mean([x[1] for x in pairs]):+.4f}", flush=True)
        print(f"{len(jobs)} 个纯回放待跑 (workers={a.workers})", flush=True)

        def one(job):
            tag, align, cname, s = job
            args = ro.common_args("B", rg.END, "training_data_pit_v24_tick1.parquet", None, rg.TEST_START,
                                  profile=ro.EXEC_POINTS["Q5"])
            cmd = ["nice", "-n", "10", sys.executable, "-u", str(root / "scripts/wf_v35_breadth_alpha.py"),
                   *args, "--lgb-seed", str(s), "--label-alignment", align,
                   "--tag", tag, "--load-preds", cname]
            log = root / "logs" / f"{tag.lower()}_s{s}.log"
            with log.open("w") as fh:
                rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            return tag, s, rc

        with ThreadPoolExecutor(a.workers) as ex:
            for tag, s, rc in ex.map(one, jobs):
                if rc != 0:
                    print(f"  ERR {tag} s{s}: rc={rc}", flush=True)
        print("FMOM_NEUT_RUN_DONE", flush=True)
    else:
        base_pg = {s: rg.daily(rg.out_path(root, "N7_Q5", s)) for s in ro.N6_SEEDS if rg.out_path(root, "N7_Q5", s).exists()}
        base_leg = {s: rg.daily(rg.out_path(root, "FM_LEG_BASE", s)) for s in ro.N6_SEEDS if rg.out_path(root, "FM_LEG_BASE", s).exists()}
        res = {
            "base_pg_med": {k: round(float(np.median([rg.stats(d["daily_ret"])[k] for d in base_pg.values()])), 2) for k in ("ret", "sharpe", "dd")},
            "base_leg_med": {k: round(float(np.median([rg.stats(d["daily_ret"])[k] for d in base_leg.values()])), 2) for k in ("ret", "sharpe", "dd")} if base_leg else {},
        }
        for arm, (src_kind, _) in ARMS.items():
            var = {s: rg.daily(rg.out_path(root, f"FM_{arm}", s)) for s in ro.N6_SEEDS if rg.out_path(root, f"FM_{arm}", s).exists()}
            if var:
                ref = base_leg if src_kind == "legacy" else base_pg
                res[arm] = po.compare_port(ref, var)
        (root / "summary_fmom_neut.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
