"""合并多个 walk-forward 预测缓存，并重新计算集成 IC。"""
import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import settings  # noqa: E402


def combine_day(rows, labels, method="plain"):
    codes = [str(c) for c in rows[0]["ranked"]]
    code_set = set(codes)
    arrays = []
    for row in rows:
        other = [str(c) for c in row["ranked"]]
        if set(other) != code_set:
            raise ValueError("同一预测日的候选股票集合不一致")
        values = dict(zip(other, row["pred_vals"], strict=True))
        array = np.array([values[c] for c in codes], dtype=float)
        if method == "zscore":
            std = array.std()
            array = (array - array.mean()) / std if std > 0 else array - array.mean()
        arrays.append(array)
    mean_pred = np.mean(arrays, axis=0)
    order = np.argsort(-mean_pred, kind="stable")
    ranked = [codes[i] for i in order]
    pred_vals = [round(float(mean_pred[i]), 8) for i in order]
    label_values = np.array([labels.get(c, np.nan) for c in ranked], dtype=float)
    valid = np.isfinite(label_values) & np.isfinite(pred_vals)
    ic = spearmanr(np.array(pred_vals)[valid], label_values[valid])[0] if valid.sum() > 5 else np.nan
    return {"ranked": ranked, "pred_vals": pred_vals, "ic": ic, "blocked": set()}


def validate_meta(caches):
    keys = (
        "train_file", "pit_universe", "label", "objective", "test_start",
        "test_end", "neutralize_style", "n_features", "feat_cutoff",
    )
    base = caches[0]["meta"]
    for cache in caches[1:]:
        for key in keys:
            if cache["meta"].get(key) != base.get(key):
                raise ValueError(
                    f"缓存元数据不一致: {key}={cache['meta'].get(key)!r}, "
                    f"期望 {base.get(key)!r}"
                )
    return base


def top_overlap(caches, n=3):
    values = []
    for left, right in combinations(caches, 2):
        overlaps = []
        for a, b in zip(left["preds"], right["preds"], strict=True):
            overlaps.append(len(set(a["ranked"][:n]) & set(b["ranked"][:n])) / n)
        values.append(float(np.mean(overlaps)))
    return float(np.mean(values)) if values else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("plain", "zscore"), default="plain")
    args = parser.parse_args()

    processed = settings.PROCESSED_DIR
    caches = []
    for name in args.inputs:
        with open(processed / name, "rb") as file:
            caches.append(pickle.load(file))
    base_meta = validate_meta(caches)

    dates = [[pd.Timestamp(row["date"]) for row in cache["preds"]] for cache in caches]
    if any(value != dates[0] for value in dates[1:]):
        raise ValueError("预测缓存日期序列不一致")

    matrix = pd.read_parquet(processed / args.matrix, columns=["date", "code", "y_target"])
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix["code"] = matrix["code"].astype(str)
    labels = {
        date: dict(zip(group["code"], group["y_target"], strict=True))
        for date, group in matrix.groupby("date")
    }

    predictions = []
    for index, date in enumerate(dates[0]):
        rows = [cache["preds"][index] for cache in caches]
        combined = combine_day(rows, labels.get(date, {}), args.method)
        predictions.append({"date": date, **combined})

    meta = dict(base_meta)
    meta.update({
        "model": "lightgbm_seed_ensemble",
        "ensemble_method": args.method,
        "ensemble_inputs": args.inputs,
        "ensemble_size": len(caches),
    })
    output = processed / args.output
    with open(output, "wb") as file:
        pickle.dump({"meta": meta, "preds": predictions}, file)

    individual_ics = [
        np.nanmean([row["ic"] for row in cache["preds"]]) for cache in caches
    ]
    ensemble_ics = np.array([row["ic"] for row in predictions], dtype=float)
    print(f"输入缓存: {len(caches)} | 日期: {len(predictions)} | 方法: {args.method}")
    print(f"单种子 IC: {[round(float(x), 5) for x in individual_ics]}")
    print(f"集成 IC: {np.nanmean(ensemble_ics):+.5f}")
    print(f"单种子两两 top3 重合率: {top_overlap(caches):.1%}")
    print(f"已保存: {output}")


if __name__ == "__main__":
    main()
