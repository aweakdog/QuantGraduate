import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return float(value) if np.isfinite(value) else None


def tensors(batch, device):
    import torch

    context = torch.from_numpy(batch["context"]).to(device)
    targets = torch.from_numpy(batch["targets"]).to(device)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, :, 0] = True
    return context, targets, mask


def rank_correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3:
        return None
    a, b = pd.Series(left[keep]).rank().to_numpy(), pd.Series(right[keep]).rank().to_numpy()
    return finite(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else None


def summarize_return_group(scores, actual):
    scores, actual = np.asarray(scores, dtype=float), np.asarray(actual, dtype=float)
    if scores.shape != actual.shape or scores.ndim != 1:
        raise ValueError("scores and outcomes must be aligned vectors")
    present = np.flatnonzero(np.isfinite(scores))
    known = present[np.isfinite(actual[present])]
    top = present[np.argsort(-scores[present], kind="stable")[:3]]
    missing = int((~np.isfinite(actual[top])).sum())
    universe = float(actual[known].mean()) if len(known) else None
    excess = float(actual[top].mean() - universe) if len(top) and not missing and universe is not None else None
    return {"n_predictions": len(present), "n_labeled": len(known), "rank_ic": rank_correlation(scores, actual),
            "universe_mean_5d_return": universe, "top3_excess_5d": excess, "top3_missing_labels": missing,
            "endpoint_return_mae": float(np.abs(scores[known] - actual[known]).mean()) if len(known) else None,
            "zero_return_mae": float(np.abs(actual[known]).mean()) if len(known) else None}


def block_bootstrap_interval(values, block=10, samples=2000, seed=42):
    values = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    values[~np.isfinite(values)] = np.nan
    valid = np.isfinite(values)
    result = {"mean": float(values[valid].mean()) if valid.any() else None, "low": None, "high": None,
              "n_dates": len(values), "n_valid": int(valid.sum()), "block_days": block, "samples": samples}
    if min(block, samples) < 1:
        raise ValueError("bootstrap sizes must be positive")
    if len(values) < 2 * block or valid.sum() < 2 * block:
        return result
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - block + 1, size=(samples, int(np.ceil(len(values) / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(samples, -1)[:, :len(values)]
    draws = values[indices]
    counts = np.isfinite(draws).sum(axis=1)
    means = np.nansum(draws, axis=1)[counts > 0] / counts[counts > 0]
    result["low"], result["high"] = [float(v) for v in np.quantile(means, [0.025, 0.975])]
    return result


def context_shift_metrics(reference, other):
    reference, other = np.asarray(reference, dtype=float), np.asarray(other, dtype=float)
    if reference.ndim != 1 or reference.shape != other.shape or not len(reference):
        raise ValueError("probe scores must align")
    if not (np.isfinite(reference).all() and np.isfinite(other).all()):
        raise ValueError("nonfinite probe scores")
    delta = np.abs(reference - other) * 10000
    top = min(3, len(reference))
    a, b = set(np.argsort(-reference, kind="stable")[:top]), set(np.argsort(-other, kind="stable")[:top])
    return {"median_abs_change_bp": float(np.median(delta)), "max_abs_change_bp": float(delta.max()),
            "rank_agreement": rank_correlation(reference, other), "top3_overlap": len(a & b) / top}


def evaluate(model, batches, panel, device, amp, progress=None):
    import torch

    from scripts.falcon_model import quantile_loss
    from scripts.falcon_objectives import endpoint_returns, return_pinball

    was_training = model.training
    model.eval()
    rows, correlations, daily = [], [], []
    loss_sum = naive_sum = error_sum = weight_sum = 0.0
    crossing_sum = crossing_count = 0
    endpoint_coverage = np.zeros(len(model.quantiles), dtype=float)
    endpoint_count = endpoint_crossings = return_count = 0
    return_loss_sum = 0.0
    median = model.config.quantiles.index(0.5)
    with torch.no_grad():
        for batch in batches:
            context, target, target_mask = tensors(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(context, target.shape[-1])
                loss = quantile_loss(output, target, model.quantiles, target_mask)
            if not torch.isfinite(output["prediction"]).all() or not torch.isfinite(loss):
                raise RuntimeError("nonfinite evaluation forecast/loss")
            valid = torch.isfinite(target) & output["active"].unsqueeze(-1) & target_mask
            count = int(valid.sum())
            last = context[..., -1:]
            last = torch.where(torch.isfinite(last), last, output["location"])
            normalized_last = torch.asinh((last - output["location"]) / output["scale"])
            naive = dict(output, normalized_prediction=normalized_last.unsqueeze(-1).expand_as(output["normalized_prediction"]))
            naive_loss = quantile_loss(naive, target, model.quantiles, target_mask)
            errors = torch.where(valid, (output["prediction"][..., median] - target).abs(), 0)
            loss_sum += float(loss) * count
            naive_sum += float(naive_loss) * count
            error_sum += float(errors.sum())
            weight_sum += count
            crossings = output["normalized_prediction"][..., 1:] < output["normalized_prediction"][..., :-1]
            crossing_sum += int((crossings & valid.unsqueeze(-1)).sum())
            crossing_count += int(valid.sum()) * (len(model.quantiles) - 1)
            endpoint_valid = valid[:, :, 0, -1]
            covered = target[:, :, 0, -1, None] <= output["prediction"][:, :, 0, -1]
            endpoint_coverage += (covered & endpoint_valid[..., None]).sum(dim=(0, 1)).cpu().numpy()
            endpoint_count += int(endpoint_valid.sum())
            endpoint_crossings += int((crossings[:, :, 0, -1] & endpoint_valid[..., None]).sum())
            predicted_returns, realized_returns, return_valid = endpoint_returns(output, target, context, target_mask)
            count_returns = int(return_valid.sum())
            if count_returns:
                return_loss_sum += float(return_pinball(predicted_returns, realized_returns, return_valid, model.quantiles)) * count_returns
                return_count += count_returns
            forecasts = output["prediction"][:, :, 0, -1, median]
            scores = torch.expm1(forecasts - context[:, :, 0, -1]).cpu().numpy()
            actual = torch.expm1(target[:, :, 0, -1] - context[:, :, 0, -1]).cpu().numpy()
            if not np.isfinite(scores[batch["entities"] >= 0]).all():
                raise RuntimeError("nonfinite return score")
            for b, position in enumerate(batch["positions"]):
                present = batch["entities"][b] >= 0
                group = summarize_return_group(scores[b, present], actual[b, present])
                ic = group["rank_ic"]
                if ic is not None:
                    correlations.append(ic)
                daily.append({"date": str(panel.dates[position].date()), **group})
                if progress is not None:
                    progress(len(daily), daily[-1])
                for e, stock in enumerate(batch["entities"][b]):
                    if stock >= 0:
                        rows.append({"date": str(panel.dates[position].date()), "code": panel.codes[stock],
                                     "score": finite(scores[b, e]), "realized_5d_return": finite(actual[b, e]),
                                     "group_rank_ic": ic})
    model.train(was_training)
    if not weight_sum:
        raise RuntimeError("no evaluation observations")
    endpoint_labels = sum(d["n_labeled"] for d in daily)
    endpoint_mae = (sum(d["endpoint_return_mae"] * d["n_labeled"] for d in daily if d["n_labeled"]) / endpoint_labels
                    if endpoint_labels else None)
    zero_mae = (sum(d["zero_return_mae"] * d["n_labeled"] for d in daily if d["n_labeled"]) / endpoint_labels
                if endpoint_labels else None)
    return {"endpoint_return_mae": endpoint_mae, "zero_return_mae": zero_mae,
            "endpoint_return_pinball": return_loss_sum / return_count if return_count else None,
            "endpoint_return_pinball_scale": 0.05,
            "endpoint_quantile_crossing_fraction": endpoint_crossings / (endpoint_count * (len(model.quantiles) - 1)) if endpoint_count and len(model.quantiles) > 1 else None,
            "normalized_pinball": loss_sum / weight_sum, "naive_normalized_pinball": naive_sum / weight_sum,
            "log_price_mae": error_sum / weight_sum, "quantile_crossing_fraction": crossing_sum / max(1, crossing_count),
            "group_rank_ic_mean": float(np.mean(correlations)) if correlations else None,
            "n_rank_ic_groups": len(correlations), "n_target_observations": int(weight_sum), "predictions": rows,
            "daily": daily, "rank_ic_block_interval": block_bootstrap_interval([d["rank_ic"] for d in daily]),
            "endpoint_quantile_levels": list(model.config.quantiles),
            "endpoint_quantile_coverage": (endpoint_coverage / endpoint_count).tolist() if endpoint_count else None,
            "n_endpoint_labels": endpoint_count}


def fixed_batches(panel, positions, args, seed):
    from scripts.falcon_data import make_batch

    take = np.linspace(0, len(positions) - 1, min(args.eval_days, len(positions)), dtype=int)
    rng = np.random.default_rng(seed)
    return [make_batch(panel, [int(positions[i])], args.context, args.entities, args.horizon, rng) for i in take]


def milestone_steps(steps, milestones):
    result = tuple(milestones)
    if not result or tuple(sorted(set(result))) != result or result[0] < 1 or result[-1] != steps:
        raise ValueError("milestones must be positive, strictly increasing and end at total steps")
    return result


def train_variant(variant, panel, splits, args, directory, provenance):
    import torch

    from scripts.falcon_data import fixed_entity_batch, make_batch
    from scripts.falcon_model import FalconConfig, FalconForecaster
    from scripts.falcon_objectives import training_objective

    objective = getattr(args, "objective", "price_path")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    config = FalconConfig(variant=variant, width=args.width, heads=args.heads,
                          temporal_layers=args.temporal_layers, spatial_layers=args.spatial_layers,
                          prototypes=args.prototypes, patch_size=args.patch_size)
    model = FalconForecaster(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp = device.type == "cuda" and args.precision == "bf16"
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 unavailable; choose float32 explicitly")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    budget = getattr(args, "mode", "trial") == "budget"
    milestones = milestone_steps(args.steps, args.milestones) if budget else ()
    validation = ([fixed_entity_batch(panel, int(t), args.context, args.horizon) for t in splits["validation"]]
                  if budget else fixed_batches(panel, splits["validation"], args, args.seed + 10000))
    initial = evaluate(model, validation, panel, device, amp)
    best_loss, best_step = initial["normalized_pinball"], 0
    best_state = None if budget else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    started = time.monotonic()
    history, budget_records = [], []
    for step in range(1, args.steps + 1):
        positions = rng.choice(splits["train"], size=args.batch_dates, replace=True)
        batch = make_batch(panel, positions, args.context, args.entities, args.horizon, rng)
        context, targets, target_mask = tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(context, args.horizon)
            forecast_loss = training_objective(output, targets, model.quantiles, target_mask, context, objective,
                                               return_scale=getattr(args, "return_scale", 0.05),
                                               rank_weight=getattr(args, "rank_weight", 0.1),
                                               rank_temperature=getattr(args, "rank_temperature", 0.05))
            orthogonal = model.orthogonality_loss()
            loss = forecast_loss + args.orthogonality_weight * orthogonal
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite gradient norm")
        optimizer.step()
        if step % 128 == 0:
            write_json(directory / "status.json", {"state": "training", "variant": variant,
                       "step": step, "total_steps": args.steps, "updated_at": timestamp()})
        check = step in milestones if budget else step % args.eval_every == 0 or step == args.steps
        if check:
            write_json(directory / "status.json", {"state": "validating", "variant": variant,
                       "step": step, "total_steps": args.steps, "updated_at": timestamp()})
            metrics = evaluate(model, validation, panel, device, amp)
            if not budget and metrics["normalized_pinball"] < best_loss:
                best_loss, best_step = metrics["normalized_pinball"], step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            entry = {"step": step, "objective": objective, "train_objective_loss": float(forecast_loss.detach()),
                     "train_pinball": float(forecast_loss.detach()) if objective != "return_rank" else None,
                     "orthogonality": float(orthogonal.detach()), "validation_pinball": metrics["normalized_pinball"],
                     "validation_rank_ic": metrics["group_rank_ic_mean"], "validation_return_mae": metrics["endpoint_return_mae"],
                     "zero_return_mae": metrics["zero_return_mae"], "crossing_fraction": metrics["quantile_crossing_fraction"],
                     "endpoint_return_pinball": metrics["endpoint_return_pinball"],
                     "endpoint_crossing_fraction": metrics["endpoint_quantile_crossing_fraction"]}
            history.append(entry)
            print(variant, json.dumps(entry), flush=True)
            if budget:
                checkpoint = directory / f"{variant}_step{step}.pt"
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model_state": state, "config": asdict(config), "trained_steps": step,
                            "seed": args.seed, "objective": objective, "data_manifest_sha256": provenance["manifest_sha256"]}, checkpoint)
                budget_records.append({"step": step, "checkpoint": checkpoint.name,
                                       "validation": {k: v for k, v in metrics.items() if k != "predictions"}})
                write_json(directory / f"budget_{variant}.json", {"variant": variant, "seed": args.seed, "objective": objective,
                           "parameters": sum(p.numel() for p in model.parameters()), "config": asdict(config),
                           "milestones": budget_records, "complete": step == args.steps, "test_not_evaluated": True,
                           "history": history, "updated_at": timestamp()})
    if budget:
        result = {"variant": variant, "seed": args.seed, "objective": objective, "config": asdict(config),
                  "parameters": sum(p.numel() for p in model.parameters()), "milestones": budget_records,
                  "initial_validation": {k: v for k, v in initial.items() if k != "predictions"},
                  "elapsed_seconds": time.monotonic() - started, "test_not_evaluated": True,
                  "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                  "fixed_steps_not_selected_checkpoints": True, "research_only": True, "is_walk_forward_backtest": False}
        write_json(directory / f"result_{variant}.json", result)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = directory / f"{variant}.pt"
    torch.save({"model_state": best_state, "config": asdict(config), "best_step": best_step,
                "seed": args.seed, "data_manifest_sha256": provenance["manifest_sha256"]}, checkpoint)
    reloaded = FalconForecaster(FalconConfig(**asdict(config))).to(device).eval()
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["model_state"])
    probe, _, _ = tensors(validation[0], device)
    with torch.no_grad():
        torch.testing.assert_close(model(probe, args.horizon)["prediction"],
                                   reloaded(probe, args.horizon)["prediction"])
    final_validation = evaluate(model, validation, panel, device, amp)
    test = fixed_batches(panel, splits["test"], args, args.seed + 20000)
    heldout = evaluate(model, test, panel, device, amp)
    result = {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()), "config": asdict(config),
              "initial_validation_pinball": initial["normalized_pinball"], "best_step": best_step,
              "best_validation": {k: v for k, v in final_validation.items() if k != "predictions"},
              "heldout_diagnostics": heldout, "history": history, "checkpoint_reload_verified": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
              "research_only": True, "is_walk_forward_backtest": False}
    write_json(directory / f"result_{variant}.json", result)
    del model, reloaded, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {k: v for k, v in result.items() if k not in {"history", "heldout_diagnostics"}} | {
        "heldout_diagnostics": {k: v for k, v in heldout.items() if k != "predictions"}}


def main():
    from scripts.falcon_objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description="Local Falcon-inspired architecture and chronological training trial; no trading")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=["temporal", "dense", "prototype"], default=["temporal", "dense", "prototype"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["trial", "budget"], default="trial")
    parser.add_argument("--objective", choices=OBJECTIVES, default="price_path")
    parser.add_argument("--return-scale", type=float, default=0.05)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--milestones", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--precision", choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--eval-days", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--entities", type=int, default=16)
    parser.add_argument("--batch-dates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--orthogonality-weight", type=float, default=0.01)
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--test-start", default="2026-07-01")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.eval_days, args.entities, args.batch_dates) < 1:
        parser.error("training sizes must be positive")
    if len(args.variants) != len(set(args.variants)):
        parser.error("duplicate variants")
    if args.objective != "price_path" and args.mode != "budget":
        parser.error("objective comparisons require validation-only budget mode")
    if not np.isfinite([args.return_scale, args.rank_weight, args.rank_temperature]).all() or min(args.return_scale, args.rank_temperature) <= 0 or args.rank_weight < 0:
        parser.error("invalid return scale or ranking parameters")
    if args.mode == "budget":
        try:
            milestone_steps(args.steps, args.milestones)
        except ValueError as error:
            parser.error(str(error))
    if args.horizon != 5:
        parser.error("this financial trial fixes the legacy 5-day target; use the model API for other horizons")
    if args.output_dir.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("output must be outside the frozen input snapshot")
    if args.output_dir.exists():
        parser.error("output directory exists; refusing overwrite")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from scripts.falcon_data import file_hash, load_panel, split_positions, verify_snapshot

    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(mode=0o700)
    try:
        write_json(args.output_dir / "status.json", {"state": "preparing", "pid": os.getpid(), "updated_at": timestamp()})
        provenance = verify_snapshot(args.data_root)
        if args.expected_manifest_sha256 and provenance["manifest_sha256"] != args.expected_manifest_sha256:
            raise ValueError("snapshot differs from study's pinned manifest")
        panel = load_panel(args.data_root)
        splits = split_positions(panel.dates, args.context, args.horizon, args.validation_start, args.test_start)
        if args.mode == "budget":
            splits.pop("test")
        for name, positions in splits.items():
            splits[name] = np.array([int(t) for t in positions if len(panel.eligible(int(t), args.context)) >= 3], dtype=np.int64)
            if len(splits[name]) < 2:
                raise ValueError("insufficient split: " + name)
        split_info = {name: {"signal_start": str(panel.dates[p[0]].date()), "signal_end": str(panel.dates[p[-1]].date()),
                           "last_target_date": str(panel.dates[p[-1] + args.horizon].date()), "n_dates": len(p)} for name, p in splits.items()}
        write_json(args.output_dir / "run_config.json", {"started_at": timestamp(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "inputs": provenance, "channels": list(panel.channels), "splits": split_info,
                   "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "source_sha256": {p.name: file_hash(p) for p in (ROOT / "scripts").glob("falcon_*.py")},
                   "implementation_choices": {"normalization": "observed-context population moments + asinh/sinh, not the paper's arcsin/sin notation",
                                              "orthogonality": "squared cross-Gram of unit positive/negative prototype keys",
                                              "negative_weight": "positive softplus lambda initialized to 1",
                                              "financial_loss": args.objective,
                                              "objective_controls": "price_path: original normalized/asinh price loss over all five horizons; price_endpoint: same loss at horizon five only; return_endpoint: pinball on simple five-day returns divided by return_scale; return_rank: return_endpoint plus rank_weight times same-date pairwise logistic loss with rank_temperature. Actual constants are recorded in args. Other channels remain historical covariates.",
                                              "comparison_metrics": "Endpoint return MAE, rank IC, return pinball (fixed scale 0.05), endpoint crossings/coverage. Early-horizon diagnostics are not comparable when only the endpoint is supervised.",
                                              "cross_date_attention": "forbidden by independent batch axis",
                                              "quantiles": "linear unconstrained head; crossing rate reported, no claim of calibrated joint portfolio distribution",
                                              "scale": "small unpretrained model, shared width/depth; parameter counts differ across ablations"},
                   "limitations": ("Validation-only training-budget study. Fixed update milestones, full eligible PIT validation pool, no best-checkpoint selection, no test inference. Single-seed results require paired multi-seed review; no walk-forward P&L or automatic adoption."
                                   if args.mode == "budget" else
                                   "Engineering trial only. Fixed checkpoint chosen on validation; heldout evaluated after selection. No pretraining, no walk-forward P&L, no claim of beating production. Test groups are a fixed sampled subset, not the full stock universe.")})
        print("DATA READY", json.dumps(split_info), "shape", panel.values.shape, flush=True)
        results = []
        for variant in args.variants:
            print("START", variant, flush=True)
            results.append(train_variant(variant, panel, splits, args, args.output_dir, provenance))
            write_json(args.output_dir / "summary.json", {"updated_at": timestamp(), "research_only": True, "results": results})
        if verify_snapshot(args.data_root) != provenance:
            raise ValueError("snapshot changed during training")
        write_json(args.output_dir / "status.json", {"state": "completed", "variants": args.variants, "updated_at": timestamp()})
        print("TRIAL COMPLETE", flush=True)
    except Exception as error:
        write_json(args.output_dir / "status.json", {"state": "failed", "error": str(error), "updated_at": timestamp()})
        raise


if __name__ == "__main__":
    main()
