"""广度空仓开关是否过于保守 (RG, 11-02 预注册, 见 docs/experiment_board.md)。纯回放、零训练。

  python scripts/regime_gate_study.py run --root /home/yliog/quant-research-20260912-n2 --workers 8
  python scripts/regime_gate_study.py summary --root /home/yliog/quant-research-20260912-n2
复用 N7_Q5 (qmt10w 现行) 的 40 个 purge6 预测缓存, 只改执行层开关: OFF / B30 / B35 / B45, 对照已有 N7_Q5 (B40)。
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_overnight_research as ro  # noqa: E402

TEST_START, END = "2023-09-20", "2026-09-11"
SPLIT = "2025-03-17"
VARIANTS = {"OFF": {"--regime-filter": "off"}, "B30": {"--regime-breadth": "0.30"},
            "B35": {"--regime-breadth": "0.35"}, "B45": {"--regime-breadth": "0.45"}}
GATE = {"ret_pp": 5.0, "pos_min": 26, "sharpe": 0.10, "dd_pp": 3.0, "spell_share": 0.6}


def out_path(root, tag, seed):
    return Path(root) / "data/processed" / f"wf_daily_{tag}_s{seed}_ts{TEST_START}_te{END}_cap100000.json"


def command(root, var, seed):
    args = ro.common_args("B", END, "training_data_pit_v24_tick1.parquet", None, TEST_START, profile=ro.EXEC_POINTS["Q5"])
    for k, v in VARIANTS[var].items():
        args[args.index(k) + 1] = v
    return [sys.executable, "-u", str(Path(root) / "scripts/wf_v35_breadth_alpha.py"), *args, "--lgb-seed", str(seed),
            "--label-alignment", "purge6", "--tag", f"RG_{var}", "--load-preds", ro.n6_caches(seed)[1]]


def run(root, workers):
    jobs = [(v, s) for v in VARIANTS for s in ro.N6_SEEDS if not out_path(root, f"RG_{v}", s).exists()]
    print(f"{len(jobs)} 个回放待跑", flush=True)

    def one(job):
        v, s = job
        log = Path(root) / "logs" / f"rg_{v}_s{s}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("w") as f:
            rc = subprocess.run(["nice", "-n", "10", *command(root, v, s)], cwd=root, stdout=f, stderr=subprocess.STDOUT).returncode
        return v, s, rc
    with ThreadPoolExecutor(workers) as ex:
        for v, s, rc in ex.map(one, jobs):
            print(f"  RG_{v} s{s}: rc={rc}", flush=True)


def daily(path):
    d = pd.DataFrame(json.loads(Path(path).read_text())["daily"])
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")[["daily_ret", "in_cash"]].astype({"daily_ret": float, "in_cash": bool})


def stats(r):
    eq = (1 + r).cumprod()
    return {"ret": float((eq.iloc[-1] - 1) * 100), "sharpe": float(r.mean() / r.std() * np.sqrt(244)) if r.std() > 0 else 0.0,
            "dd": float((eq / eq.cummax() - 1).min() * 100)}


def segments(mask):
    out, start, prev = [], None, None
    for d, m in mask.items():
        if m and start is None:
            start = d
        if not m and start is not None:
            out.append((start, prev))
            start = None
        prev = d
    if start is not None:
        out.append((start, prev))
    return out


def compare(base, var):
    """base/var: {seed: DataFrame}。返回中位差、正数个数、两半、按段统计。"""
    seeds = sorted(set(base) & set(var))
    rows = []
    for s in seeds:
        b, v = base[s]["daily_ret"], var[s]["daily_ret"].reindex(base[s].index).fillna(0)
        sb, sv = stats(b), stats(v)
        h1, h2 = b.index < SPLIT, b.index >= SPLIT
        rows.append({"seed": s, "d_ret": sv["ret"] - sb["ret"], "d_sharpe": sv["sharpe"] - sb["sharpe"], "d_dd": sv["dd"] - sb["dd"],
                     "d_h1": stats(v[h1])["ret"] - stats(b[h1])["ret"], "d_h2": stats(v[h2])["ret"] - stats(b[h2])["ret"]})
    t = pd.DataFrame(rows)
    s0 = seeds[0]
    diff = base[s0]["in_cash"] != var[s0]["in_cash"].reindex(base[s0].index).fillna(True)
    segs = []
    for a, e in segments(diff):
        per = [float((1 + var[s]["daily_ret"].loc[a:e]).prod() / (1 + base[s]["daily_ret"].loc[a:e]).prod() - 1) for s in seeds]
        segs.append({"start": str(a.date()), "end": str(e.date()), "days": int(diff.loc[a:e].sum()), "median_pct": round(float(np.median(per)) * 100, 2)})
    drop = None
    if segs:
        top = max(segs, key=lambda x: x["median_pct"])
        a, e = pd.Timestamp(top["start"]), pd.Timestamp(top["end"])
        dd = []
        for s in seeds:
            v = var[s]["daily_ret"].reindex(base[s].index).fillna(0).copy()
            v.loc[a:e] = base[s]["daily_ret"].loc[a:e]
            dd.append(stats(v)["ret"] - stats(base[s]["daily_ret"])["ret"])
        drop = round(float(np.median(dd)), 2)
    med = t.median(numeric_only=True)
    res = {"n": len(seeds), "med_d_ret": round(float(med["d_ret"]), 2), "pos": int((t["d_ret"] > 0).sum()),
           "med_d_sharpe": round(float(med["d_sharpe"]), 3), "med_d_dd": round(float(med["d_dd"]), 2),
           "med_d_h1": round(float(med["d_h1"]), 2), "med_d_h2": round(float(med["d_h2"]), 2),
           "segments": segs, "seg_pos_share": round(sum(x["median_pct"] > 0 for x in segs) / len(segs), 3) if segs else None,
           "med_d_ret_drop_top_seg": drop}
    res["pass_core"] = bool(res["med_d_ret"] >= GATE["ret_pp"] and res["pos"] >= GATE["pos_min"] and res["med_d_sharpe"] >= GATE["sharpe"]
                            and res["med_d_h1"] > 0 and res["med_d_h2"] > 0 and res["med_d_dd"] >= -GATE["dd_pp"]
                            and (res["seg_pos_share"] or 0) >= GATE["spell_share"] and (drop or -1) >= 0)
    return res


def summary(root):
    base = {s: daily(out_path(root, "N7_Q5", s)) for s in ro.N6_SEEDS if out_path(root, "N7_Q5", s).exists()}
    res = {"baseline": "N7_Q5 (B40)", "n_base": len(base),
           "base_median": {k: round(float(np.median([stats(d["daily_ret"])[k] for d in base.values()])), 2) for k in ("ret", "sharpe", "dd")},
           "cash_share_base": round(float(next(iter(base.values()))["in_cash"].mean()), 3)}
    for v in VARIANTS:
        var = {s: daily(out_path(root, f"RG_{v}", s)) for s in ro.N6_SEEDS if out_path(root, f"RG_{v}", s).exists()}
        if var:
            res[v] = compare(base, var) | {"cash_share": round(float(next(iter(var.values()))["in_cash"].mean()), 3)}
    if "B30" in res and "B35" in res:      # 邻域一致: B30 过要求 B35 也不差于 B40
        res["B30"]["pass"] = res["B30"]["pass_core"] and res["B35"]["med_d_ret"] >= 0
    for v in ("OFF", "B35", "B45"):
        if v in res:
            res[v]["pass"] = res[v]["pass_core"]
    (Path(root) / "summary_rg.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "summary"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.phase == "run":
        run(a.root, a.workers)
    else:
        print(json.dumps(summary(a.root), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
