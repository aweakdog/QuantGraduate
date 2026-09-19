import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_NAME = "label_alignment_panel.parquet"
HELPER_COLUMNS = {"lab1_t1close_5d", "lab1_common", "lab1_calendar_5d"}
EXTERNAL_PREFIXES = (
    "a50_futures_", "dj_futures_", "nq_futures_", "sp_futures_", "sox_",
    "cn2y", "cn5y", "cn_pmi", "cn_commodity_idx_", "us2y", "us5y", "usdind", "usdjpy",
)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


# 五种标签口径。N2 的 CONTROL(common) 同时改了两件事: 多截断一天 + 只用共同有效行;
# purge6 / common5 各取其一, 用来把 CONTROL 相对现行的差异拆开归因 (N3, 2026-09-17)。
#   legacy  旧标签 C[T+5]/C[T]-1, 5 日截断, 全部行 demean
#   purge6  旧标签, 6 日截断, 全部行 demean            (只多截断一天)
#   common5 旧标签, 5 日截断, 仅共同有效行参与训练/demean (只筛行)
#   common  旧标签, 6 日截断, 仅共同有效行             (N2 CONTROL = purge6 + common5)
#   t1close 对齐标签 C[T+6]/C[T+1]-1, 6 日截断, 仅共同有效行
MODES = ("legacy", "purge6", "common5", "common", "t1close")
PANEL_MODES = {"common5", "common", "t1close"}      # 需要标签侧表(共同有效行)的口径
SIX_DAY_MODES = {"purge6", "common", "t1close"}     # 训练截断 6 日的口径


def label_horizon(label, mode):
    if mode not in MODES:
        raise ValueError("unknown label alignment")
    if mode != "legacy" and label != "5d":
        raise ValueError("label alignment research supports only 5d")
    return {"1d": 1, "2d": 2, "5d": 5}[label] + int(mode in SIX_DAY_MODES)


def uses_panel(mode):
    if mode not in MODES:
        raise ValueError("unknown label alignment")
    return mode in PANEL_MODES


def build_alignment_panel(base, klines, tolerance=1e-6):
    keys = base[["date", "code", "fwd_5d_ret"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    if keys.duplicated(["date", "code"]).any():
        raise ValueError("duplicate matrix keys")
    calendar = pd.DatetimeIndex(sorted(keys.date.unique()))
    parts = []
    for code, group in keys.groupby("code", sort=False):
        frame = klines.get(str(code)[:6])
        if frame is None:
            close = pd.Series(np.nan, index=calendar)
        else:
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"])
            close = pd.to_numeric(frame.drop_duplicates("date").set_index("date")["close"], errors="coerce")
            close = close.reindex(calendar).where(lambda s: s > 0)
        current = close.shift(-5) / close - 1
        aligned = close.shift(-6) / close.shift(-1) - 1
        out = group.copy()
        out["lab1_calendar_5d"] = current.reindex(group.date).to_numpy()
        out["lab1_t1close_5d"] = aligned.reindex(group.date).to_numpy()
        out["lab1_common"] = (out.fwd_5d_ret.notna() & out.lab1_calendar_5d.notna()
                              & np.isfinite(out.lab1_t1close_5d)
                              & ((out.fwd_5d_ret - out.lab1_calendar_5d).abs() <= tolerance))
        parts.append(out.drop(columns="fwd_5d_ret"))
    result = pd.concat(parts, ignore_index=True).set_index(["date", "code"])
    return result.reindex(pd.MultiIndex.from_frame(keys[["date", "code"]])).reset_index()


def apply_alignment(frame, panel, mode):
    if mode not in PANEL_MODES:
        raise ValueError("panel is used only for common-sample alignments")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("duplicate label panel keys")
    columns = ["date", "code", *sorted(HELPER_COLUMNS)]
    joined = frame.merge(panel[columns], on=["date", "code"], how="left", sort=False, validate="one_to_one")
    joined["lab1_common"] = joined.lab1_common.eq(True)
    return joined


def common_test_start(base, panel, requested="2023-09-19", minimum=250):
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(base.date).unique()))
    old_dates = pd.DatetimeIndex(sorted(base.loc[base.fwd_5d_ret.notna(), "date"].unique()))
    common_dates = pd.DatetimeIndex(sorted(panel.loc[panel.lab1_common, "date"].unique()))
    for pos in range(calendar.searchsorted(pd.Timestamp(requested)), len(calendar)):
        if pos >= 6 and (old_dates < calendar[pos - 5]).sum() >= minimum and (common_dates < calendar[pos - 6]).sum() >= minimum:
            return str(calendar[pos].date())
    raise ValueError("insufficient common warmup history")


def prune_features(features, arm):
    if len(features) != len(set(features)):
        raise ValueError("duplicate selected features")
    if arm == "FULL":
        return list(features)
    if arm not in {"NOEXO", "NOMACRO"}:
        raise ValueError("unknown pruning arm")
    return [f for f in features if not f.startswith(EXTERNAL_PREFIXES)
            and not (arm == "NOMACRO" and f.startswith("mkt_"))]


def validate_cache_contract(meta, mode, panel_sha256, features):
    expected = {"label_alignment": mode,
                "label_horizon": label_horizon(meta["label"], mode),
                "label_panel_sha256": panel_sha256}
    defaults = {"label_alignment": "legacy", "label_horizon": label_horizon(meta["label"], "legacy"),
                "label_panel_sha256": None}
    for key, value in expected.items():
        if meta.get(key, defaults[key]) != value:
            raise ValueError("prediction cache mismatch: " + key)
    if "selected_features" in meta and list(meta["selected_features"]) != list(features):
        raise ValueError("prediction cache mismatch: selected_features")


def load_panel(panel_path, train_path, frame, mode):
    panel_path = Path(panel_path)
    meta = json.loads(panel_path.with_suffix(".meta.json").read_text())
    if not meta["ok"] or meta["source_sha256"] != file_hash(train_path):
        raise ValueError("label panel QC or source fingerprint mismatch")
    checksum = file_hash(panel_path)
    if checksum != meta["panel_sha256"]:
        raise ValueError("label panel fingerprint mismatch")
    return apply_alignment(frame, pd.read_parquet(panel_path), mode), checksum


def prepare(root):
    proc = root / "data/processed"
    source = proc / "training_data_pit_v24_tick1.parquet"
    out = proc / PANEL_NAME
    if out.exists() or (root / "n2_inputs.json").exists():
        raise RuntimeError("N2 inputs already exist; refusing overwrite")
    base = pd.read_parquet(source, columns=["date", "code", "fwd_5d_ret"])
    klines = {}
    for code in sorted(base.code.astype(str).str[:6].unique()):
        path = root / "data/raw/kline" / (code + ".parquet")
        if path.exists():
            frame = pd.read_parquet(path).rename(columns={"时间": "date", "收盘价": "close"})
            klines[code] = frame[["date", "close"]]
    panel = build_alignment_panel(base, klines)
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(base.date).unique()))
    historical = (base.date <= calendar[-7]) & base.fwd_5d_ret.notna()
    fraction = float(panel.loc[historical, "lab1_common"].mean())
    mismatch = (panel.lab1_calendar_5d - base.fwd_5d_ret).abs() > 1e-6
    qc = {"ok": fraction >= 0.95, "test_end": str(calendar[-1].date()),
          "common_fraction_observable": fraction, "historical_eligible": int(historical.sum()),
          "common_rows": int(panel.lab1_common.sum()), "price_calendar_mismatch_rows": int(mismatch.sum()),
          "price_tolerance": 1e-6, "source_sha256": file_hash(source), "closure_horizon": 6,
          "formula": "C[T+6]/C[T+1]-1", "common_policy": "finite endpoints and legacy/calendar return agreement; training only"}
    panel.to_parquet(out, index=False)
    qc["panel_sha256"] = file_hash(out)
    out.with_suffix(".meta.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    pruning = {}
    for model in ["A", "B"]:
        feature_path = proc / f"features_V24PUT_T1{model}.json"
        features = json.loads(feature_path.read_text())["selected_features"]
        pruning[model] = {}
        for arm in ["NOEXO", "NOMACRO"]:
            selected = prune_features(features, arm)
            removed = [f for f in features if f not in selected]
            if not selected or not removed:
                raise ValueError("empty/no-op feature ablation")
            data = {"selected_features": selected, "removed_features": removed,
                    "source_features_sha256": file_hash(feature_path), "arm": arm,
                    "regime_filter": "unchanged", "label_alignment": "legacy"}
            path = proc / f"features_N2_{model}_{arm}.json"
            with path.open("x", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            pruning[model][arm] = {"selected": len(selected), "removed": removed, "sha256": file_hash(path)}
    result = {"label_ready": qc["ok"], "prune_ready": True, "test_end": qc["test_end"],
              "label_qc": qc, "pruning": pruning}
    (root / "n2_inputs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not qc["ok"]:
        raise RuntimeError("label common-sample fraction below 95%; investigate before LAB1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    parser.parse_args()
    prepare(ROOT)
