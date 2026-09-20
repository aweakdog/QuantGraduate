import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SEEDS = [1, 42, 123, 888, 2024, 7, 31337, 2, 3, 5, 11, 17, 23, 55, 77, 99, 202, 314, 512, 1234]
ENSEMBLE = [42, 7, 123, 2024, 31337]
PROFILES = {
    "A": {"name": "aggr10w", "capital": 100000, "positions": 3, "ind_cap": 2, "features": "features_V24PUT_T1A.json"},
    "B": {"name": "steady5w", "capital": 50000, "positions": 5, "ind_cap": 0, "features": "features_V24PUT_T1B.json"},
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest(root):
    path = root / "snapshot_manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    matrix = root / "data/processed/training_data_pit_v24_tick1.parquet"
    keys = pd.read_parquet(matrix, columns=["date", "code"])
    end = pd.Timestamp(keys.date.max()).strftime("%Y-%m-%d")
    files = [matrix, root / "data/universe/universe_pit.parquet", root / "data/raw/tushare/sw_member/sw_member.parquet"]
    files += sorted((root / "data/raw/kline").glob("*.parquet"))
    files += sorted((root / "data/processed").glob("features_V24PUT_T1*.json"))
    files += sorted((root / "data/processed").glob("features_N2_*.json"))
    files += [p for p in [root / "data/processed/label_alignment_panel.parquet",
                          root / "data/processed/label_alignment_panel.meta.json", root / "n2_inputs.json"] if p.exists()]
    hashes = {str(p.relative_to(root)): digest(p) for p in files}
    test_start = "2023-09-19"
    if (root / "n2_inputs.json").exists():
        from scripts.research_labels import common_test_start
        base = pd.read_parquet(matrix, columns=["date", "code", "fwd_5d_ret"])
        panel = pd.read_parquet(root / "data/processed/label_alignment_panel.parquet")
        test_start = common_test_start(base, panel)
    value = {"created_at": stamp(), "test_start": test_start, "test_end": end,
             "matrix_rows": len(keys), "files": hashes, "code_sha256": {p.name: digest(p) for p in sorted((root / "scripts").glob("*.py"))}}
    write_json(path, value)
    return value


def common_args(model, end, matrix, features=None, test_start="2023-09-19"):
    p = PROFILES[model]
    return ["--train-file", matrix, "--pit-universe", "universe_pit.parquet", "--label", "5d", "--objective", "l2",
            "--features-from", features or p["features"], "--hold-days", "5", "--portfolio-mode", "periodic",
            "--exec-mode", "t1close", "--slippage", "0.002", "--regime-filter", "breadth", "--regime-ma", "20",
            "--regime-breadth", "0.40", "--regime-confirm", "2", "--min-pred", "0.002", "--fill-daily", "--roll-rank", "8",
            "--skip-boards", "30,688", "--tranche-n", str(p["positions"]), "--ind-cap", str(p["ind_cap"]),
            "--initial-capital", str(p["capital"]), "--test-start", test_start, "--test-end", end]


def make_tasks(root, phase, end, test_start="2023-09-19"):
    tasks = []
    engine = str(root / "scripts/wf_v35_breadth_alpha.py")
    if phase not in {"corr", "t2a", "label", "prune", "n3", "n4"}:
        raise ValueError("unknown study phase")
    matrix = "training_data_pit_v24_tick1_t2a.parquet" if phase == "t2a" else "training_data_pit_v24_tick1.parquet"

    def add(name, command, deps, output, priority=1):
        tasks.append({"id": name, "command": command, "deps": deps, "output": str(output), "priority": priority})

    def result(name, model, seed):
        return root / "data/processed" / (f"wf_daily_{name}_s{seed}_ts{test_start}_te{end}_cap{PROFILES[model]['capital']}.json")

    if phase == "n4":
        # N4 (2026-09-20): N3 拆解发现 CONTROL 的增益几乎全来自 purge6 (B 点 +45.4pp, 8/10)。
        #  1) purge6 20 种子确认: 补 SEEDS[10:20]; 基线用 N3 已有的 N2_F_*_FULL 同种子结果
        #     (FULL 与 CURRENT 是同一 legacy 模型配置, 同种子逐位相同, 不必重训)。
        #  2) 剂量-反应: purge7 / purge8 × SEEDS[:10], 与 N2 CURRENT 配对。
        for seed in SEEDS[10:20]:
            for model in PROFILES:
                tag = f"N2_L_{model}_PURGE6"
                base = [sys.executable, "-u", engine, *common_args(model, end, matrix, None, test_start), "--lgb-seed", str(seed)]
                add(f"{tag}_s{seed}", [*base, "--label-alignment", "purge6", "--tag", tag, "--save-preds", f"preds_{tag}_s{seed}.pkl"],
                    [], result(tag, model, seed))
        for seed in SEEDS[:10]:
            for model in PROFILES:
                for arm, mode in [("PURGE7", "purge7"), ("PURGE8", "purge8")]:
                    tag = f"N2_L_{model}_{arm}"
                    base = [sys.executable, "-u", engine, *common_args(model, end, matrix, None, test_start), "--lgb-seed", str(seed)]
                    add(f"{tag}_s{seed}", [*base, "--label-alignment", mode, "--tag", tag, "--save-preds", f"preds_{tag}_s{seed}.pkl"],
                        [], result(tag, model, seed), 2)
        return tasks
    if phase == "n3":
        # N3 (2026-09-17): 两个接力实验, 同一队列。
        #  1) NOEXO 20 种子确认: 只补 SEEDS[10:20] 的 FULL/NOEXO, 文件名与 N2 完全同构,
        #     汇总时与 N2 已有的 10 种子拼成 20 配对 (同快照同 K线, 允许拼接)。
        #  2) CONTROL 拆解: purge6(只多截断一天) / common5(只筛共同行), 各 10 种子,
        #     与 N2 已有的 CURRENT/CONTROL 结果配对, 把 CONTROL-CURRENT 的差异归因。
        for seed in SEEDS[10:20]:
            for model in PROFILES:
                for arm in ["FULL", "NOEXO"]:
                    features = f"features_N2_{model}_{arm}.json" if arm != "FULL" else None
                    tag = f"N2_F_{model}_{arm}"
                    base = [sys.executable, "-u", engine, *common_args(model, end, matrix, features, test_start), "--lgb-seed", str(seed)]
                    add(f"{tag}_s{seed}", [*base, "--tag", tag, "--save-preds", f"preds_{tag}_s{seed}.pkl"], [], result(tag, model, seed))
        for seed in SEEDS[:10]:
            for model in PROFILES:
                for arm, mode in [("PURGE6", "purge6"), ("COMMON5", "common5")]:
                    tag = f"N2_L_{model}_{arm}"
                    base = [sys.executable, "-u", engine, *common_args(model, end, matrix, None, test_start), "--lgb-seed", str(seed)]
                    add(f"{tag}_s{seed}", [*base, "--label-alignment", mode, "--tag", tag, "--save-preds", f"preds_{tag}_s{seed}.pkl"],
                        [], result(tag, model, seed), 2)
        return tasks
    if phase in {"label", "prune"}:
        modes = {"CURRENT": "legacy", "CONTROL": "common", "ALIGNED": "t1close"}
        arms = list(modes) if phase == "label" else ["FULL", "NOEXO", "NOMACRO"]
        prefix = "N2_L" if phase == "label" else "N2_F"
        for seed in SEEDS[:10]:
            for model in PROFILES:
                for arm in arms:
                    features = f"features_N2_{model}_{arm}.json" if phase == "prune" and arm != "FULL" else None
                    tag = f"{prefix}_{model}_{arm}"
                    base = [sys.executable, "-u", engine, *common_args(model, end, matrix, features, test_start), "--lgb-seed", str(seed)]
                    if phase == "label":
                        base += ["--label-alignment", modes[arm]]
                    add(f"{tag}_s{seed}", [*base, "--tag", tag, "--save-preds", f"preds_{tag}_s{seed}.pkl"],
                        [], result(tag, model, seed))
        return tasks
    if phase == "corr":
        for seed in SEEDS:
            for model in PROFILES:
                train = f"N1_C_{model}_TRAIN_s{seed}"
                cache = f"preds_N1_C_{model}_s{seed}.pkl"
                base = [sys.executable, "-u", engine, *common_args(model, end, matrix), "--lgb-seed", str(seed)]
                add(train, [*base, "--tag", train, "--save-preds", cache], [], root / "data/processed" / cache, 2)
                for cap in [0, 0.8, 0.7, 0.6]:
                    tag = f"N1_C_{model}_C{round(cap * 100):02d}"
                    add(f"{tag}_s{seed}", [*base, "--tag", tag, "--load-preds", cache, "--corr-cap", str(cap)],
                        [train], result(tag, model, seed), 0)
        for model in PROFILES:
            name = f"N1_C_{model}_ENSEMBLE"
            cache = f"preds_N1_C_{model}_E5.pkl"
            add(name, [sys.executable, "-u", str(root / "scripts/ensemble_pred_caches.py"), "--inputs",
                       *[f"preds_N1_C_{model}_s{s}.pkl" for s in ENSEMBLE], "--matrix", matrix, "--output", cache],
                [f"N1_C_{model}_TRAIN_s{s}" for s in ENSEMBLE], root / "data/processed" / cache, 0)
            for cap in [0, 0.8, 0.7, 0.6]:
                tag = f"N1_C_{model}_E5_C{round(cap * 100):02d}"
                base = [sys.executable, "-u", engine, *common_args(model, end, matrix), "--lgb-seed", "42"]
                add(tag, [*base, "--tag", tag, "--load-preds", cache, "--corr-cap", str(cap)], [name], result(tag, model, 42), 0)
    else:
        for seed in SEEDS[:5]:
            for model in PROFILES:
                for arm in ["BASE", "T2A"]:
                    features = PROFILES[model]["features"] if arm == "BASE" else f"features_V24PUT_T1{model}_T2A.json"
                    tag = f"N1_T2_{model}_{arm}"
                    cache = f"preds_{tag}_s{seed}.pkl"
                    base = [sys.executable, "-u", engine, *common_args(model, end, matrix, features, test_start), "--lgb-seed", str(seed)]
                    add(f"{tag}_s{seed}", [*base, "--tag", tag, "--save-preds", cache], [], result(tag, model, seed))
    return tasks


def resources():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.strip().split()[0])
    return os.getloadavg()[0], memory["MemAvailable"] / 1024**2


def compare(a, b):
    da, db = pd.DataFrame(a["daily"]).set_index("date"), pd.DataFrame(b["daily"]).set_index("date")
    if not da.index.equals(db.index):
        raise ValueError("paired result dates differ")
    sa, sb = a["summary"], b["summary"]
    fields = ["total_return_pct", "max_dd_pct", "avg_deployed_pct", "avg_holdings", "n_trades", "total_cost_pct"]
    out = {key: float(sa[key] - sb[key]) for key in fields}
    out["arm_return"] = sa["total_return_pct"]
    out["base_return"] = sb["total_return_pct"]
    out["n_corr_skip"] = sa.get("n_corr_skip", 0)
    out["corr_unavailable"] = sa.get("corr_unavailable", 0)
    out["worst_day_delta_pp"] = float((da.daily_ret.min() - db.daily_ret.min()) * 100)
    out["es5_delta_pp"] = float((da.daily_ret.nsmallest(max(1, len(da) // 20)).mean() - db.daily_ret.nsmallest(max(1, len(db) // 20)).mean()) * 100)
    out["yearly_delta_pp"] = {}
    for year in sorted({str(d)[:4] for d in da.index}):
        mask = da.index.astype(str).str.startswith(year)
        out["yearly_delta_pp"][year] = float(((1 + da.loc[mask, "daily_ret"]).prod() - (1 + db.loc[mask, "daily_ret"]).prod()) * 100)
    for size in [63, 126]:
        out[f"recent_{size}_delta_pp"] = float(((1 + da.daily_ret.tail(size)).prod() - (1 + db.daily_ret.tail(size)).prod()) * 100)
    keep = da.index.astype(str) != "2026-08-19"
    out["excluding_0819_delta_pp"] = float(((1 + da.loc[keep, "daily_ret"]).prod() - (1 + db.loc[keep, "daily_ret"]).prod()) * 100)
    out["event_0819_delta_pp"] = float((da.loc["2026-08-19", "daily_ret"] - db.loc["2026-08-19", "daily_ret"]) * 100) if "2026-08-19" in da.index else None
    return out


def summarize_n2(root, phase, end, test_start="2023-09-19"):
    prefix = "N2_L" if phase == "label" else "N2_F"
    comparisons = [("CONTROL", "CURRENT"), ("ALIGNED", "CONTROL"), ("ALIGNED", "CURRENT")] if phase == "label" else [("NOEXO", "FULL"), ("NOMACRO", "FULL")]
    rows = []
    for model in PROFILES:
        for arm, base in comparisons:
            pairs = []
            for seed in SEEDS[:10]:
                suffix = f"_s{seed}_ts{test_start}_te{end}_cap{PROFILES[model]['capital']}.json"
                pa = root / "data/processed" / f"wf_daily_{prefix}_{model}_{arm}{suffix}"
                pb = root / "data/processed" / f"wf_daily_{prefix}_{model}_{base}{suffix}"
                if pa.exists() and pb.exists():
                    pairs.append({"seed": seed, **compare(json.loads(pa.read_text()), json.loads(pb.read_text()))})
            if not pairs:
                continue
            median = {k: float(np.median([p[k] for p in pairs])) for k in pairs[0]
                      if k not in {"seed", "yearly_delta_pp"} and pairs[0][k] is not None}
            positive = sum(p["total_return_pct"] > 0 for p in pairs)
            primary = phase == "prune" or (arm, base) == ("ALIGNED", "CONTROL")
            screen = primary and len(pairs) == 10 and positive >= 7 and median["total_return_pct"] >= 3 and median["max_dd_pct"] >= -2
            rows.append({"model": model, "profile": PROFILES[model], "arm": arm + "-" + base,
                         "n_pairs": len(pairs), "expected_pairs": 10, "median": median,
                         "positive_return_differences": positive, "primary": primary,
                         "passes_exploration_screen": screen, "adoption_ready": False, "pairs": pairs})
    out = {"updated_at": stamp(), "phase": phase, "test_end": end, "rows": rows,
           "interpretation": "Ten-seed screening only, no automatic promotion or extra variants. LAB1 ALIGNED-CONTROL isolates target change; CONTROL-CURRENT measures common-sample/purge effects; ALIGNED-CURRENT checks practical benefit. Features and execution fixed within each contrast. Seeds are not independent market histories."}
    write_json(root / f"summary_{phase}.json", out)
    return out


N3_GATE = {"stage": "twenty_seed_confirmation", "arm": "NOEXO-FULL", "requires_complete_pairs": 20,
           "return_delta_pp_min": 5, "positive_pairs_min": 13, "drawdown_delta_pp_min": -2,
           "recent_126_delta_pp_min": -5, "joint_policy": "both operating points must pass; no per-profile cherry-picking",
           "automatic_promotion": False,
           "dissection": "PURGE6-CURRENT and COMMON5-CURRENT attribute CONTROL-CURRENT; attribution only, no gate"}


def _paired_rows(root, prefix, model, arm, base, seeds, end, test_start, fallback_base=None):
    """fallback_base: 基线文件缺失时可用的等价基线 (prefix, arm), 如 N2_F FULL 之于 N2_L CURRENT ——
    两者是同一 legacy 模型配置, 同种子结果逐位相同, 只是标签不同。"""
    pairs = []
    for seed in seeds:
        suffix = f"_s{seed}_ts{test_start}_te{end}_cap{PROFILES[model]['capital']}.json"
        pa = root / "data/processed" / f"wf_daily_{prefix}_{model}_{arm}{suffix}"
        pb = root / "data/processed" / f"wf_daily_{prefix}_{model}_{base}{suffix}"
        if not pb.exists() and fallback_base is not None:
            pb = root / "data/processed" / f"wf_daily_{fallback_base[0]}_{model}_{fallback_base[1]}{suffix}"
        if pa.exists() and pb.exists():
            pairs.append({"seed": seed, **compare(json.loads(pa.read_text()), json.loads(pb.read_text()))})
    if not pairs:
        return None
    median = {k: float(np.median([p[k] for p in pairs])) for k in pairs[0]
              if k not in {"seed", "yearly_delta_pp"} and pairs[0][k] is not None}
    years = sorted({y for p in pairs for y in (p.get("yearly_delta_pp") or {})})
    yearly = {y: float(np.median([p["yearly_delta_pp"][y] for p in pairs if y in p.get("yearly_delta_pp", {})])) for y in years}
    return {"model": model, "profile": PROFILES[model], "arm": f"{arm}-{base}", "n_pairs": len(pairs),
            "median": median, "yearly_median_delta_pp": yearly,
            "positive_return_differences": sum(p["total_return_pct"] > 0 for p in pairs),
            "recent_126_positive": sum(p["recent_126_delta_pp"] > 0 for p in pairs), "pairs": pairs}


def summarize_n3(root, end, test_start="2023-09-19"):
    rows, verdict = [], {}
    for model in PROFILES:
        row = _paired_rows(root, "N2_F", model, "NOEXO", "FULL", SEEDS[:20], end, test_start)
        if row is not None:
            m = row["median"]
            complete = row["n_pairs"] == N3_GATE["requires_complete_pairs"]
            row["expected_pairs"] = 20
            row["passes_gate"] = bool(complete and m["total_return_pct"] >= N3_GATE["return_delta_pp_min"]
                                      and row["positive_return_differences"] >= N3_GATE["positive_pairs_min"]
                                      and m["max_dd_pct"] >= N3_GATE["drawdown_delta_pp_min"]
                                      and m["recent_126_delta_pp"] >= N3_GATE["recent_126_delta_pp_min"])
            row["complete"] = complete
            verdict[model] = row["passes_gate"] if complete else None
            rows.append(row)
        for arm in ["PURGE6", "COMMON5", "CONTROL"]:
            row = _paired_rows(root, "N2_L", model, arm, "CURRENT", SEEDS[:10], end, test_start)
            if row is not None:
                row["expected_pairs"] = 10
                row["role"] = "dissection" if arm != "CONTROL" else "reference"
                rows.append(row)
    out = {"updated_at": stamp(), "phase": "n3", "test_end": end, "gate": N3_GATE,
           # 联合判定要求两个操作点都齐 20 配对; 任一点缺席或未满就是 None, 不得用单点宣布通过
           "noexo_both_points_pass": (all(verdict[m] for m in PROFILES)
                                      if all(verdict.get(m) is not None for m in PROFILES) else None),
           "adoption_ready": False, "rows": rows,
           "interpretation": "NOEXO-FULL rows pool N2's 10 seeds with N3's 10 new seeds (same frozen snapshot/K-lines). "
                             "Dissection rows share N2 CURRENT baselines; PURGE6 + COMMON5 need not add up to CONTROL. "
                             "Passing the gate proposes, never performs, a production feature-set change."}
    write_json(root / "summary_n3.json", out)
    return out


N4_GATE = {**N3_GATE, "arm": "PURGE6-CURRENT",
           "dose_response": "PURGE7/PURGE8-CURRENT (10 seeds) read alongside PURGE6: monotone = systematic; spike at 6 only = boundary chaos",
           "dissection": None}


def summarize_n4(root, end, test_start="2023-09-19"):
    rows, verdict = [], {}
    for model in PROFILES:
        row = _paired_rows(root, "N2_L", model, "PURGE6", "CURRENT", SEEDS[:20], end, test_start, fallback_base=("N2_F", "FULL"))
        if row is not None:
            m = row["median"]
            complete = row["n_pairs"] == N4_GATE["requires_complete_pairs"]
            row.update(expected_pairs=20, complete=complete, role="gate",
                       passes_gate=bool(complete and m["total_return_pct"] >= N4_GATE["return_delta_pp_min"]
                                        and row["positive_return_differences"] >= N4_GATE["positive_pairs_min"]
                                        and m["max_dd_pct"] >= N4_GATE["drawdown_delta_pp_min"]
                                        and m["recent_126_delta_pp"] >= N4_GATE["recent_126_delta_pp_min"]))
            verdict[model] = row["passes_gate"] if complete else None
            rows.append(row)
        for arm in ["PURGE7", "PURGE8", "COMMON5", "CONTROL"]:
            row = _paired_rows(root, "N2_L", model, arm, "CURRENT", SEEDS[:10], end, test_start)
            if row is not None:
                row.update(expected_pairs=10, role="dose_response" if arm.startswith("PURGE") else "reference")
                rows.append(row)
    out = {"updated_at": stamp(), "phase": "n4", "test_end": end, "gate": N4_GATE,
           "purge6_both_points_pass": (all(verdict[m] for m in PROFILES)
                                       if all(verdict.get(m) is not None for m in PROFILES) else None),
           "adoption_ready": False, "rows": rows,
           "interpretation": "PURGE6 pools N3's 10 seeds with N4's 10 new seeds; new-seed baselines are N3 FULL runs (identical legacy model). "
                             "Passing the gate proposes, never performs, a change of the live training cutoff; a mechanism must be stated first."}
    write_json(root / "summary_n4.json", out)
    return out


def summarize(root, phase, end, test_start="2023-09-19"):
    if phase == "n4":
        return summarize_n4(root, end, test_start)
    if phase == "n3":
        return summarize_n3(root, end, test_start)
    if phase in {"label", "prune"}:
        return summarize_n2(root, phase, end, test_start)
    proc = root / "data/processed"
    rows = []
    for model in PROFILES:
        arms = ["C80", "C70", "C60"] if phase == "corr" else ["T2A"]
        for arm in arms:
            pairs = []
            for seed in SEEDS if phase == "corr" else SEEDS[:5]:
                prefix = f"N1_C_{model}" if phase == "corr" else f"N1_T2_{model}"
                base_arm = "C00" if phase == "corr" else "BASE"
                suffix = f"_s{seed}_ts2023-09-19_te{end}_cap{PROFILES[model]['capital']}.json"
                pa = proc / f"wf_daily_{prefix}_{arm}{suffix}"
                pb = proc / f"wf_daily_{prefix}_{base_arm}{suffix}"
                if pa.exists() and pb.exists():
                    pairs.append({"seed": seed, **compare(json.loads(pa.read_text()), json.loads(pb.read_text()))})
            if not pairs:
                continue
            median = {key: float(np.median([p[key] for p in pairs])) for key in pairs[0] if key not in ["seed", "yearly_delta_pp"] and pairs[0][key] is not None}
            loss_a = sum(p["arm_return"] < 0 for p in pairs)
            loss_b = sum(p["base_return"] < 0 for p in pairs)
            passed = len(pairs) == 20 and median["total_return_pct"] >= -2 and median["max_dd_pct"] >= 2 and loss_a <= loss_b
            rows.append({"model": model, "profile": PROFILES[model], "arm": arm, "n_pairs": len(pairs), "median": median,
                         "positive_return_differences": sum(p["total_return_pct"] > 0 for p in pairs),
                         "loss_seeds_arm_base": [loss_a, loss_b], "passes_corr_numeric_gate": passed if phase == "corr" else None,
                         "pairs": pairs})
    ensembles = []
    if phase == "corr":
        for model in PROFILES:
            suffix = f"_s42_ts2023-09-19_te{end}_cap{PROFILES[model]['capital']}.json"
            base = proc / f"wf_daily_N1_C_{model}_E5_C00{suffix}"
            for arm in ["C80", "C70", "C60"]:
                other = proc / f"wf_daily_N1_C_{model}_E5_{arm}{suffix}"
                if base.exists() and other.exists():
                    ensembles.append({"model": model, "arm": arm, **compare(json.loads(other.read_text()), json.loads(base.read_text()))})
    out = {"updated_at": stamp(), "phase": phase, "test_end": end, "rows": rows, "production_ensemble": ensembles,
           "interpretation": "Historical research only; no auto promotion. Seeds are not independent market histories. T2a five-seed output is exploratory, not adoption evidence. Excluding 08-19 removes its realized daily return, not its subsequent portfolio path effects."}
    write_json(root / f"summary_{phase}.json", out)
    return out


def run(root, phase, workers, max_load, min_memory):
    info = manifest(root)
    end = info["test_end"]
    if phase == "corr" and not (root / "smoke_verified.json").exists():
        raise RuntimeError("original-vs-new smoke verification missing")
    if phase in {"label", "prune"}:
        qc = json.loads((root / "n2_inputs.json").read_text())
        smoke = root / "n2_smoke_verified.json"
        if not qc[phase + "_ready"] or qc["test_end"] != end or not smoke.exists():
            raise RuntimeError("N2 input QC/date or smoke verification failed")
    if phase in {"n3", "n4"}:
        qc = json.loads((root / "n2_inputs.json").read_text())
        if not (qc["label_ready"] and qc["prune_ready"]) or qc["test_end"] != end or not (root / "n2_smoke_verified.json").exists():
            raise RuntimeError("N3/N4 need the verified N2 inputs (label + prune) on this snapshot")
        # 接力前提: 配对所需的基线结果都在本快照里, 否则拼不成配对
        needed = ([("N2_L", ["CURRENT", "CONTROL"], SEEDS[:10]), ("N2_F", ["FULL", "NOEXO"], SEEDS[:10])] if phase == "n3"
                  else [("N2_L", ["CURRENT", "PURGE6"], SEEDS[:10]), ("N2_F", ["FULL"], SEEDS[10:20])])
        if phase == "n4" and not (root / "n3_smoke_verified.json").exists():
            raise RuntimeError("N4 needs the N3 engine smoke verification")
        for prefix, arms, seeds in needed:
            for model in PROFILES:
                for arm in arms:
                    for seed in seeds:
                        p = root / "data/processed" / f"wf_daily_{prefix}_{model}_{arm}_s{seed}_ts{info['test_start']}_te{end}_cap{PROFILES[model]['capital']}.json"
                        if not p.is_file() or p.stat().st_size == 0:
                            raise RuntimeError(f"{phase.upper()} baseline missing: " + p.name)
    if phase == "t2a":
        qc = json.loads((root / "t2a_qc.json").read_text())
        if not qc["ok"] or qc["test_end"] != end:
            raise RuntimeError("T2a coverage check failed")
    for name, expected in info["files"].items():
        if digest(root / name) != expected:
            raise RuntimeError("snapshot changed: " + name)
    code_hashes = {p.name: digest(p) for p in sorted((root / "scripts").glob("*.py"))}
    extra_inputs = {}
    if phase == "t2a":
        extra_inputs = {p.name: digest(p) for p in [root / "data/processed/training_data_pit_v24_tick1_t2a.parquet", *sorted((root / "data/processed").glob("features_*_T2A.json"))]}
    tasks = make_tasks(root, phase, end, info["test_start"])
    status_path = root / f"status_{phase}.json"
    if status_path.exists() or (root / f"plan_{phase}.json").exists():
        raise RuntimeError("existing queue: inspect before restarting; no automatic duplicate launch")
    n2 = phase in {"label", "prune"}
    gate = {"n3": N3_GATE, "n4": N4_GATE}.get(phase) or ({"stage": "ten_seed_screen_only", "return_delta_pp_min": 3, "drawdown_delta_pp_min": -2,
             "positive_pairs_min": 7, "requires_complete_pairs": 10, "automatic_promotion": False,
             "label_primary": "ALIGNED-CONTROL", "label_practical_check": "ALIGNED-CURRENT must also be nonnegative",
             "joint_policy": "Both operating points must pass before proposing 20-seed confirmation; no posthoc per-profile cherry-picking"}
            if n2 else {"return_delta_pp_min": -2, "drawdown_delta_pp_min": 2, "loss_seeds_must_not_increase": True,
                        "corr_window": 20, "corr_min_periods": 15, "missing_pairs": "allow_and_count", "sell_policy": "unchanged"})
    write_json(root / f"plan_{phase}.json", {"created_at": stamp(), "code_sha256": code_hashes, "extra_inputs_sha256": extra_inputs, "workers": workers, "max_load": max_load,
               "min_memory_gb": min_memory, "seeds": SEEDS[:10] if n2 else SEEDS, "profiles": PROFILES, "tasks": tasks,
               "gate": gate})
    status = {t["id"]: {"state": "pending"} for t in tasks}
    active = {}
    env = dict(os.environ, QUANT_DATA_DIR=str(root / "data"), QUANT_MODE="backtest", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = stamp()
    last_wait = 0
    while True:
        for name, (process, log, began, task) in list(active.items()):
            rc = process.poll()
            if rc is None:
                if time.monotonic() - began > 21600:
                    status[name]["over_time_limit"] = True
                continue
            log.close()
            good = rc == 0 and Path(task["output"]).is_file() and Path(task["output"]).stat().st_size > 0
            status[name].update(state="completed" if good else "failed", returncode=rc, finished_at=stamp())
            del active[name]
            print(name, status[name]["state"], flush=True)
            summarize(root, phase, end, info["test_start"])
        for task in tasks:
            s = status[task["id"]]
            if s["state"] == "pending" and any(status[d]["state"] in ["failed", "blocked"] for d in task["deps"]):
                s.update(state="blocked", reason="dependency failed")
        load, available = resources()
        ready = sorted([t for t in tasks if status[t["id"]]["state"] == "pending" and all(status[d]["state"] == "completed" for d in t["deps"])], key=lambda t: t["priority"])
        reserve = 0
        for task in ready:
            if len(active) >= workers or load + reserve + 10 > max_load or available - reserve * 2 - 20 < min_memory:
                break
            log_path = root / "logs" / (task["id"] + ".log")
            log = log_path.open("x")
            process = subprocess.Popen(task["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[task["id"]] = (process, log, time.monotonic(), task)
            status[task["id"]].update(state="running", pid=process.pid, started_at=stamp(), log=str(log_path))
            reserve += 10
            print("START", task["id"], "pid", process.pid, flush=True)
        counts = dict(Counter(s["state"] for s in status.values()))
        write_json(status_path, {"started_at": started, "updated_at": stamp(), "phase": phase, "queue_pid": os.getpid(),
                   "counts": counts, "load": load, "available_gb": round(available, 1), "tasks": status})
        if not active and not counts.get("pending", 0):
            break
        if not active and time.monotonic() - last_wait > 60:
            print("WAIT resources", round(load, 1), round(available, 1), flush=True)
            last_wait = time.monotonic()
        time.sleep(5)
    summarize(root, phase, end, info["test_start"])
    if counts.get("failed", 0) or counts.get("blocked", 0):
        raise RuntimeError("queue finished with failures: " + str(counts))
    print("QUEUE COMPLETE", counts, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["corr", "t2a", "label", "prune", "n3", "n4", "manifest",
                                      "summary-corr", "summary-t2a", "summary-label", "summary-prune", "summary-n3", "summary-n4"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-load", type=float, default=85)
    ap.add_argument("--min-memory-gb", type=float, default=96)
    args = ap.parse_args()
    if not (ROOT / "smoke_verified.json").exists() and args.phase == "corr":
        ap.error("smoke_verified.json missing")
    # 每个训练进程 LightGBM n_jobs=10; 6 worker ≈ 60 线程, 是用户 2026-08-22 放宽后
    # 041/040 上可接受的上限 (~64 核), 再多就吃满共享机器了
    if not 1 <= args.workers <= 6:
        ap.error("workers must be between 1 and 6")
    if args.phase == "manifest":
        print(json.dumps({k: v for k, v in manifest(ROOT).items() if k not in ["files", "code_sha256"]}))
        return
    if args.phase.startswith("summary-"):
        info = manifest(ROOT)
        print(json.dumps(summarize(ROOT, args.phase.split("-", 1)[1], info["test_end"], info["test_start"]), ensure_ascii=False))
        return
    with (ROOT / f"queue_{args.phase}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(ROOT, args.phase, args.workers, args.max_load, args.min_memory_gb)


if __name__ == "__main__":
    main()
