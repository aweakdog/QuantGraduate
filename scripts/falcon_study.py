import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("temporal", "dense", "prototype")


def study_name(seed, objective=None):
    return f"seed_{seed}" + (f"_{objective}" if objective is not None else "")


def study_commands(root, output, data_root, seeds, milestones, python=sys.executable, *,
                   variants=VARIANTS, objectives=None, expected_manifest=None):
    from scripts.falcon_objectives import OBJECTIVES
    from scripts.falcon_trial import milestone_steps

    if not milestones:
        raise ValueError("milestones must not be empty")
    milestones = milestone_steps(milestones[-1], milestones)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative, nonempty and unique")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("invalid model variants")
    if objectives is not None and (not objectives or objectives[0] != "price_path" or
                                   len(set(objectives)) != len(objectives) or any(o not in OBJECTIVES for o in objectives)):
        raise ValueError("objectives must be unique and start with the price_path control")
    tasks = []
    for seed in seeds:
        for objective in objectives if objectives is not None else [None]:
            name = study_name(seed, objective)
            command = [python, "-u", str(root / "scripts/falcon_trial.py"),
                       "--data-root", str(data_root), "--output-dir", str(output / name),
                       "--mode", "budget", "--seed", str(seed), "--steps", str(milestones[-1]),
                       "--milestones", *[str(s) for s in milestones], "--device", "cuda:0", "--precision", "bf16",
                       "--variants", *variants, "--objective", objective or "price_path",
                       "--width", "64", "--heads", "4", "--temporal-layers", "2", "--spatial-layers", "2",
                       "--prototypes", "8", "--patch-size", "16", "--context", "64", "--horizon", "5",
                       "--entities", "16", "--batch-dates", "2", "--lr", "0.001", "--orthogonality-weight", "0.01",
                       "--return-scale", "0.05", "--rank-weight", "0.1", "--rank-temperature", "0.05",
                       "--validation-start", "2026-01-01", "--test-start", "2026-07-01"]
            if expected_manifest:
                command.extend(["--expected-manifest-sha256", expected_manifest])
            tasks.append({"seed": seed, "objective": objective or "price_path", "id": name,
                          "output": str(output / name), "command": command})
    return tasks


def gpu_idle(gpu):
    base = ["nvidia-smi", "-i", str(gpu)]
    process = subprocess.run([*base, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                             check=True, capture_output=True, text=True, timeout=10)
    if process.stdout.strip():
        return False
    stats = subprocess.run([*base, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=10)
    utilization, memory = next(csv.reader(io.StringIO(stats.stdout)))
    return int(utilization.strip()) <= 5 and int(memory.strip()) < 256


def resource_ready():
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip().split()[0]
              for line in Path("/proc/meminfo").read_text().splitlines()}
    return os.getloadavg()[0] <= 64 and int(fields["MemAvailable"]) / 1024**2 >= 96


def summarize_study(output, seeds, milestones):
    rows = []
    fields = ("group_rank_ic_mean", "endpoint_return_mae", "zero_return_mae", "normalized_pinball", "quantile_crossing_fraction")
    for variant in VARIANTS:
        for step in milestones:
            pairs = []
            for seed in seeds:
                path = output / f"seed_{seed}" / f"budget_{variant}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                stages = {r["step"]: r["validation"] for r in data["milestones"]}
                if step not in stages or milestones[0] not in stages:
                    continue
                current, baseline = stages[step], stages[milestones[0]]
                metrics = {k: current.get(k) for k in fields}
                delta = {k: current[k] - baseline[k] if current.get(k) is not None and baseline.get(k) is not None else None for k in fields}
                pairs.append({"seed": seed, "metrics": metrics, "paired_vs_first": delta})
            if not pairs:
                continue
            median, changes = {}, {}
            for key in fields:
                values = [p["metrics"][key] for p in pairs if p["metrics"][key] is not None]
                differences = [p["paired_vs_first"][key] for p in pairs if p["paired_vs_first"][key] is not None]
                median[key] = float(np.median(values)) if values else None
                changes[key] = float(np.median(differences)) if differences else None
            rows.append({"variant": variant, "step": step, "n_seeds": len(pairs), "expected_seeds": len(seeds),
                         "median": median, "paired_delta_median": changes, "pairs": pairs,
                         "ic_improved_count": sum(p["paired_vs_first"]["group_rank_ic_mean"] is not None and p["paired_vs_first"]["group_rank_ic_mean"] > 0 for p in pairs),
                         "beats_zero_mae_count": sum(p["metrics"]["endpoint_return_mae"] is not None and p["metrics"]["zero_return_mae"] is not None and p["metrics"]["endpoint_return_mae"] <= p["metrics"]["zero_return_mae"] for p in pairs)})
    return {"research_only": True, "test_not_evaluated": True, "adoption_ready": False, "rows": rows}


def objective_metrics(output, seed, objective, variant, step):
    path = output / study_name(seed, objective) / f"budget_{variant}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data["seed"] != seed or data["objective"] != objective or data["variant"] != variant:
        raise ValueError("objective result identity mismatch")
    return next((r["validation"] for r in data["milestones"] if r["step"] == step), None)


def summarize_objectives(output, seeds, milestones, objectives, variants):
    rows = []
    fields = ("group_rank_ic_mean", "endpoint_return_mae", "zero_return_mae",
              "endpoint_return_pinball", "endpoint_quantile_crossing_fraction")
    for variant in variants:
        for step in milestones:
            for index, objective in enumerate(objectives):
                previous = objectives[max(0, index - 1)]
                pairs = []
                for seed in seeds:
                    current = objective_metrics(output, seed, objective, variant, step)
                    baseline = objective_metrics(output, seed, objectives[0], variant, step)
                    control = objective_metrics(output, seed, previous, variant, step)
                    if current is None or baseline is None:
                        continue
                    delta = {k: current[k] - baseline[k] if current.get(k) is not None and baseline.get(k) is not None else None for k in fields}
                    prior = {k: current[k] - control[k] if control is not None and current.get(k) is not None and control.get(k) is not None else None for k in fields}
                    pairs.append({"seed": seed, "metrics": {k: current.get(k) for k in fields},
                                  "vs_price_path": delta, "vs_previous": prior})
                if not pairs:
                    continue
                aggregates = {}
                for name in ["metrics", "vs_price_path", "vs_previous"]:
                    aggregates[name] = {}
                    for key in fields:
                        values = [p[name][key] for p in pairs if p[name][key] is not None]
                        aggregates[name][key] = float(np.median(values)) if values else None
                rows.append({"variant": variant, "objective": objective, "step": step,
                             "previous_objective": previous, "n_seeds": len(pairs), "expected_seeds": len(seeds),
                             "median": aggregates["metrics"], "paired_delta_vs_price_path": aggregates["vs_price_path"],
                             "paired_delta_vs_previous": aggregates["vs_previous"], "pairs": pairs,
                             "ic_improved_count": sum(p["vs_price_path"]["group_rank_ic_mean"] is not None and p["vs_price_path"]["group_rank_ic_mean"] > 0 for p in pairs),
                             "beats_zero_mae_count": sum(p["metrics"]["endpoint_return_mae"] is not None and p["metrics"]["zero_return_mae"] is not None and p["metrics"]["endpoint_return_mae"] <= p["metrics"]["zero_return_mae"] for p in pairs)})
    return {"research_only": True, "test_not_evaluated": True, "adoption_ready": False,
            "comparison": "fixed-budget objective contrasts within the same seed and backbone", "rows": rows}


def run(args):
    from scripts.falcon_data import file_hash, verify_snapshot
    from scripts.falcon_trial import timestamp, write_json

    provenance = verify_snapshot(args.data_root)
    args.output_dir.mkdir(mode=0o700)
    tasks = study_commands(ROOT, args.output_dir, args.data_root, args.seeds, args.milestones,
                           variants=args.variants, objectives=args.objectives, expected_manifest=provenance["manifest_sha256"])
    status = {t["id"]: {"state": "pending", "seed": t["seed"], "objective": t["objective"]} for t in tasks}
    write_json(args.output_dir / "plan.json", {"created_at": timestamp(), "tasks": tasks, "gpus": args.gpus,
               "inputs": provenance, "milestones": args.milestones, "seeds": args.seeds,
               "variants": args.variants, "objectives": args.objectives,
               "source_sha256": {p.name: file_hash(p) for p in (ROOT / "scripts").glob("falcon_*.py")},
               "protocol": ("Fixed backbone, inputs, horizon, update budgets and seeds. Compare original multi-horizon normalized price loss, endpoint-only same price loss, endpoint simple-return pinball, and endpoint-return pinball plus same-date ranking. Return scale=0.05, rank weight=0.1, temperature=0.05; constants not tuned. Only full PIT validation. Compare endpoint metrics, not unsupervised early-horizon heads. Same-seed price_path and adjacent-objective paired controls; no test inference or checkpoint selection. Exploratory, no deployment."
                            if args.objectives is not None else
                            "Fixed training budgets in each same-seed trajectory. Architecture, features, target, optimizer, learning rate and training entity count unchanged. Full PIT validation only; no test inference, no best-checkpoint selection, no production deployment. Seeds are exploratory, not independent market histories."),
               "resources": "At most two GPUs; only admit observed-idle cards with no compute processes, load<=64 and >=96GiB available RAM. Each worker limits CPU threads to two. Admission is not a cluster reservation."})
    active = {}
    waiting_since = None
    try:
        while True:
            for name, (worker, _gpu, log, task) in list(active.items()):
                code = worker.poll()
                if code is None:
                    continue
                log.close()
                path = Path(task["output"]) / "status.json"
                child = json.loads(path.read_text()) if path.exists() else {}
                success = code == 0 and child.get("state") == "completed"
                status[name].update(state="completed" if success else "failed", returncode=code, finished_at=timestamp())
                del active[name]
                print(name, status[name]["state"], flush=True)
            pending = [t for t in tasks if status[t["id"]]["state"] == "pending"]
            used = {item[1] for item in active.values()}
            if pending and resource_ready():
                for gpu in args.gpus:
                    if gpu in used or not pending or not gpu_idle(gpu):
                        continue
                    task = pending.pop(0)
                    log_path = args.output_dir / f"{task['id']}.log"
                    log = log_path.open("x")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    process = subprocess.Popen(task["command"], cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    active[task["id"]] = (process, gpu, log, task)
                    status[task["id"]].update(state="running", gpu=gpu, pid=process.pid,
                                             started_at=timestamp(), log=str(log_path))
                    used.add(gpu)
                    print("START", task["id"], "GPU", gpu, "PID", process.pid, flush=True)
            counts = dict(Counter(t["state"] for t in status.values()))
            write_json(args.output_dir / "status.json", {"updated_at": timestamp(), "queue_pid": os.getpid(),
                       "counts": counts, "tasks": status})
            summary = (summarize_objectives(args.output_dir, args.seeds, args.milestones, args.objectives, args.variants)
                       if args.objectives is not None else summarize_study(args.output_dir, args.seeds, args.milestones))
            write_json(args.output_dir / "summary.json", {"updated_at": timestamp(), "inputs": provenance, **summary})
            if not active and not counts.get("pending", 0):
                break
            if not active:
                waiting_since = waiting_since or time.monotonic()
                if time.monotonic() - waiting_since > args.wait_seconds:
                    for task in status.values():
                        if task["state"] == "pending":
                            task.update(state="blocked", reason="no admissible GPU within wait limit")
                    write_json(args.output_dir / "status.json", {"updated_at": timestamp(), "counts": dict(Counter(t["state"] for t in status.values())), "tasks": status})
                    raise RuntimeError("GPU admission wait limit exceeded")
            else:
                waiting_since = None
            time.sleep(15)
        if counts.get("failed", 0):
            raise RuntimeError("training workers failed; inspect seed logs")
        print("STUDY COMPLETE", counts, flush=True)
    except Exception as error:
        write_json(args.output_dir / "queue_error.json", {"error": str(error), "updated_at": timestamp(),
                   "active_workers": {name: {"pid": value[0].pid, "gpu": value[1]} for name, value in active.items()}})
        raise


def main():
    from scripts.falcon_objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description="Validation-only multi-seed Falcon budget/objective study")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 123])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--objectives", choices=OBJECTIVES, nargs="+")
    parser.add_argument("--milestones", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("output must be new and outside the input snapshot")
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("choose one or two distinct nonnegative GPU indices")
    if not args.milestones or args.wait_seconds <= 0:
        parser.error("invalid milestones or wait limit")
    study_commands(ROOT, args.output_dir, args.data_root, args.seeds, args.milestones,
                   variants=args.variants, objectives=args.objectives)
    run(args)


if __name__ == "__main__":
    main()
