import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compact(metrics):
    return {k: v for k, v in metrics.items() if k not in {"predictions", "daily"}}


def monthly_metrics(daily):
    result = {}
    for month in sorted({d["date"][:7] for d in daily}):
        group = [d for d in daily if d["date"].startswith(month)]
        valid = [d["rank_ic"] for d in group if d["rank_ic"] is not None]
        excess = [d["top3_excess_5d"] for d in group if d["top3_excess_5d"] is not None]
        result[month] = {"n_dates": len(group), "mean_rank_ic": float(np.mean(valid)) if valid else None,
                         "mean_top3_excess_5d": float(np.mean(excess)) if excess else None}
    return result


def scores_for_batch(model, batch, device, amp, horizon):
    import torch

    context = torch.from_numpy(batch["context"]).to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        output = model(context, horizon)
    median = model.config.quantiles.index(0.5)
    scores = torch.expm1(output["prediction"][:, :, 0, -1, median] - context[:, :, 0, -1]).cpu().numpy()[0]
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite context-probe score")
    return scores


def verify_original_scores(actual, original):
    left = {(r["date"], r["code"]): r["score"] for r in actual["predictions"]}
    right = {(r["date"], r["code"]): r["score"] for r in original["predictions"]}
    if left.keys() != right.keys():
        raise ValueError("original sampled evaluation keys changed")
    keys = sorted(left)
    np.testing.assert_allclose([left[k] for k in keys], [right[k] for k in keys], rtol=1e-4, atol=1e-5)


def context_sensitivity(model, panel, sampled_batches, sampled, full, device, amp, horizon, context, seed):
    from scripts.falcon_data import fixed_entity_batch
    from scripts.falcon_trial import context_shift_metrics

    small_lookup = {(r["date"], r["code"]): r["score"] for r in sampled["predictions"]}
    full_lookup = {(r["date"], r["code"]): r["score"] for r in full["predictions"]}
    results = []
    for batch in sampled_batches:
        position = int(batch["positions"][0])
        day = str(panel.dates[position].date())
        targets = batch["entities"][0]
        targets = targets[targets >= 0]
        eligible = panel.eligible(position, context)
        others = np.setdiff1d(eligible, targets)
        rng = np.random.default_rng(np.random.SeedSequence([seed, position, 30000]))
        extended = np.concatenate([targets, rng.permutation(others)[:max(0, 64 - len(targets))]])
        medium_batch = fixed_entity_batch(panel, position, context, horizon, extended)
        medium = scores_for_batch(model, medium_batch, device, amp, horizon)[:len(targets)]
        reference = [small_lookup[(day, panel.codes[s])] for s in targets]
        large = [full_lookup[(day, panel.codes[s])] for s in targets]
        for name, n_context, values in [("64", len(extended), medium), ("full", len(eligible), large)]:
            results.append({"date": day, "target_count": len(targets), "context": name,
                            "context_entities": n_context, "target_codes": [panel.codes[s] for s in targets],
                            **context_shift_metrics(reference, values)})
    return results


def run(args):
    import torch

    from scripts.falcon_data import (
        file_hash,
        fixed_entity_batch,
        load_panel,
        split_positions,
        verify_snapshot,
    )
    from scripts.falcon_model import FalconConfig, FalconForecaster
    from scripts.falcon_trial import evaluate, fixed_batches, timestamp, write_json

    original_config = json.loads((args.checkpoints_dir / "run_config.json").read_text())
    training_args = SimpleNamespace(**original_config["args"])
    versions = {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    if versions != original_config["versions"]:
        raise RuntimeError("runtime versions differ from the checkpoint experiment")
    if file_hash(ROOT / "scripts/falcon_model.py") != original_config["source_sha256"]["falcon_model.py"]:
        raise RuntimeError("model implementation differs from the trained checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = device.type == "cuda" and training_args.precision == "bf16"
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(mode=0o700)
    try:
        write_json(args.output_dir / "status.json", {"state": "verifying_inputs", "pid": os.getpid(), "updated_at": timestamp()})
        provenance = verify_snapshot(args.data_root)
        if provenance["manifest_sha256"] != original_config["inputs"]["manifest_sha256"]:
            raise RuntimeError("input snapshot differs from training")
        panel = load_panel(args.data_root)
        splits = split_positions(panel.dates, training_args.context, training_args.horizon,
                                 training_args.validation_start, training_args.test_start)
        variants = list(training_args.variants)
        checkpoints = {name: args.checkpoints_dir / (name + ".pt") for name in variants}
        hashes = {name: file_hash(path) for name, path in checkpoints.items()}
        write_json(args.output_dir / "evaluation_config.json", {"started_at": timestamp(), "inputs": provenance,
                   "checkpoint_sha256": hashes, "versions": versions, "training_args": vars(training_args),
                   "precision": training_args.precision, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "source_sha256": {p.name: file_hash(p) for p in (ROOT / "scripts").glob("falcon_*.py")},
                   "protocol": {"weights": "fixed original checkpoints; no optimizer or model selection",
                                "full_pool": "all asof PIT main-board members with original context availability; one complete same-date group, never silently chunked",
                                "diagnostics": "all labeled-endpoint validation/test dates; missing outcomes excluded only from metrics, not candidate selection",
                                "context_probe": "same original 16 targets under original, 64 and full-pool conditioning",
                                "bootstrap": "moving blocks of 10 trading dates, 2000 samples, seed 42; descriptive within-window uncertainty",
                                "test_reuse": "previously inspected historical period; not a fresh untouched test",
                                "limitations": "Single seed, small models, legacy T-to-T+5 target; top3 forward-return spread is not executable net P&L or an annualized strategy return; no deployment"}})
        summary = []
        for variant in variants:
            saved = torch.load(checkpoints[variant], map_location="cpu", weights_only=True)
            if saved["data_manifest_sha256"] != provenance["manifest_sha256"] or saved["config"]["variant"] != variant:
                raise RuntimeError("checkpoint metadata mismatch")
            model = FalconForecaster(FalconConfig(**saved["config"])).to(device).eval()
            model.load_state_dict(saved["model_state"])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = {"variant": variant, "checkpoint_sha256": hashes[variant], "splits": {}, "context_sensitivity": {}}
            original = json.loads((args.checkpoints_dir / f"result_{variant}.json").read_text())
            test_positions = np.array([int(t) for t in splits["test"] if len(panel.eligible(int(t), training_args.context)) >= 3])
            golden_batches = fixed_batches(panel, test_positions, training_args, training_args.seed + 20000)
            golden = evaluate(model, golden_batches, panel, device, amp)
            verify_original_scores(golden, original["heldout_diagnostics"])
            result["original_sampled_predictions_verified"] = True
            print("GOLDEN VERIFIED", variant, flush=True)
            for split in ["validation", "test"]:
                positions = np.array([int(t) for t in splits[split] if len(panel.eligible(int(t), training_args.context)) >= 3])
                if not len(positions):
                    raise RuntimeError("empty evaluation split")
                sampling_seed = training_args.seed + (10000 if split == "validation" else 20000)
                probes = golden_batches if split == "test" else fixed_batches(panel, positions, training_args, sampling_seed)
                sampled = golden if split == "test" else evaluate(model, probes, panel, device, amp)

                total_dates = len(positions)

                def progress(done, row, total=total_dates, name=variant, phase=split):
                    if done == 1 or done % 10 == 0 or done == total:
                        print(name, phase, f"{done}/{total}", row["date"], flush=True)
                        write_json(args.output_dir / "status.json", {"state": "evaluating", "variant": name,
                                   "split": phase, "completed_dates": done, "total_dates": total, "updated_at": timestamp()})

                batches = (fixed_entity_batch(panel, int(t), training_args.context, training_args.horizon) for t in positions)
                full = evaluate(model, batches, panel, device, amp, progress=progress)
                write_json(args.output_dir / f"{variant}_{split}_full.json", full)
                sensitivity = context_sensitivity(model, panel, probes, sampled, full, device, amp,
                                                  training_args.horizon, training_args.context, training_args.seed)
                write_json(args.output_dir / f"{variant}_{split}_context.json", sensitivity)
                result["splits"][split] = {"full": compact(full), "sampled_original": compact(sampled),
                                           "monthly": monthly_metrics(full["daily"])}
                result["context_sensitivity"][split] = sensitivity
            result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
            if file_hash(checkpoints[variant]) != hashes[variant]:
                raise RuntimeError("checkpoint changed during evaluation")
            summary.append(result)
            write_json(args.output_dir / "summary.json", {"updated_at": timestamp(), "research_only": True,
                       "fresh_out_of_sample": False, "results": summary})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        verify_snapshot(args.data_root)
        write_json(args.output_dir / "status.json", {"state": "completed", "variants": variants, "updated_at": timestamp()})
        print("FIXED CHECKPOINT EVALUATION COMPLETE", flush=True)
    except Exception as error:
        write_json(args.output_dir / "status.json", {"state": "failed", "error": str(error), "updated_at": timestamp()})
        raise


def main():
    parser = argparse.ArgumentParser(description="Read-only full-PIT evaluation of frozen Falcon trial checkpoints")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; refusing overwrite")
    if any(args.output_dir.resolve().is_relative_to(p.resolve()) for p in [args.data_root, args.checkpoints_dir]):
        parser.error("output must be outside inputs and checkpoint directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args)


if __name__ == "__main__":
    main()
