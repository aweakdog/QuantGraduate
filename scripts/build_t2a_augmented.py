import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_COLS = ["t2a_rskew", "t2a_rkurt", "t2a_down_share", "t2a_jump_share"]
COLS = [c + "_ma5_lag1" for c in RAW_COLS]


def minute_features(frame, min_returns=180):
    d = frame.copy()
    minute = pd.to_numeric(d["minute"], errors="coerce")
    minute = (minute // 100) * 60 + minute % 100
    d["m"] = minute
    d = d[((d.m >= 570) & (d.m <= 689)) | ((d.m >= 780) & (d.m <= 896))]
    d = d.sort_values(["code", "m"]).drop_duplicates(["code", "m"])
    d["session"] = (d.m >= 780).astype(int)
    groups = [d["code"], d["session"]]
    px = pd.to_numeric(d["close"], errors="coerce").where(lambda x: x > 0)
    r = np.log(px).groupby(groups).diff()
    contiguous = d["m"].groupby(groups).diff().eq(1)
    d["r"] = r.where(contiguous).replace([np.inf, -np.inf], np.nan)
    d["r2"] = d.r ** 2
    d["r3"] = d.r ** 3
    d["r4"] = d.r ** 4
    d["down"] = d.r2.where(d.r < 0, 0).where(d.r.notna())
    d["bp"] = (d.r.abs() * d.r.abs().groupby(groups).shift(1)).where(contiguous)
    g = d.groupby("code", sort=True)
    a = g.agg(n=("r", "count"), rv=("r2", "sum"), r3=("r3", "sum"), r4=("r4", "sum"),
              down=("down", "sum"), bp=("bp", "sum"), nbp=("bp", "count"), qc=("qc_vol_ratio", "median"))
    rv = a.rv.where(a.rv > 1e-20)
    out = pd.DataFrame(index=a.index)
    out[RAW_COLS[0]] = np.sqrt(a.n) * a.r3 / rv ** 1.5
    out[RAW_COLS[1]] = a.n * a.r4 / rv ** 2
    out[RAW_COLS[2]] = a.down / rv
    bpv = np.pi / 2 * a.bp * a.n / a.nbp.where(a.nbp > 0)
    out[RAW_COLS[3]] = ((rv - bpv).clip(lower=0) / rv).clip(0, 1)
    valid = (a.n >= min_returns) & a.qc.between(0.95, 1.05) & (a.nbp > 0)
    out.loc[~valid, RAW_COLS] = np.nan
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def lag_features(base, daily):
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(base["date"]).unique()))
    keys = base.copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["_c6"] = keys["code"].astype(str).str[:6]
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str).str[:6]
    if d.duplicated(["date", "code"]).any():
        raise ValueError("duplicate daily feature keys")
    for raw, col in zip(RAW_COLS, COLS, strict=True):
        wide = d.pivot(index="date", columns="code", values=raw).reindex(calendar)
        wide = wide.rolling(5, min_periods=5).mean().shift(1)
        lookup = wide.stack()
        keys[col] = lookup.reindex(pd.MultiIndex.from_arrays([keys.date, keys._c6])).to_numpy(dtype=np.float32)
    return keys.drop(columns="_c6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24_tick1.parquet")
    ap.add_argument("--output", default="training_data_pit_v24_tick1_t2a.parquet")
    args = ap.parse_args()
    proc = ROOT / "data/processed"
    source, output = proc / args.source, proc / args.output
    if source.resolve() == output.resolve() or output.exists():
        raise RuntimeError("output must be a new research matrix")
    base = pd.read_parquet(source)
    if set(COLS) & set(base.columns):
        raise RuntimeError("source already contains T2a features")
    end = pd.Timestamp(base.date.max())
    files = sorted(p for p in (proc / "min1").glob("*.parquet") if p.stem <= end.strftime("%Y%m%d"))
    if not files:
        raise RuntimeError("minute panel empty")
    parts, hashes = [], {}
    started = time.monotonic()
    for i, path in enumerate(files, 1):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path, columns=["code", "minute", "close", "qc_vol_ratio"])
        row = minute_features(frame)
        row["date"] = pd.Timestamp(path.stem)
        parts.append(row)
        if i % 100 == 0 or i == len(files):
            print(f"T2a {i}/{len(files)} {path.stem} elapsed={time.monotonic()-started:.0f}s", flush=True)
    daily = pd.concat(parts, ignore_index=True)
    augmented = lag_features(base, daily)
    pd.testing.assert_frame_equal(augmented[base.columns], base)
    latest = augmented[augmented.date == end]
    coverage = latest[COLS].notna().mean().to_dict()
    historical = augmented.loc[augmented.date >= pd.Timestamp("2023-09-19"), COLS].notna().mean().to_dict()
    ok = min(coverage.values()) >= 0.85 and min(historical.values()) >= 0.85
    qc = {"ok": ok, "test_end": str(end.date()), "last_minute_day": files[-1].stem,
          "n_days": len(files), "latest_coverage": coverage, "history_coverage": historical,
          "features": COLS, "lag": 1, "smoothing_days": 5, "min_contiguous_returns": 180,
          "sessions": "09:30-11:29 and 13:00-14:56; exclude auctions, lunch jumps and missing-minute bridges",
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "minute_sha256": hashes}
    (ROOT / "t2a_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if not ok:
        raise RuntimeError(f"T2a coverage failed: latest={coverage}, historical={historical}")
    temp = output.with_suffix(".parquet.part")
    augmented.to_parquet(temp, index=False)
    temp.replace(output)
    daily.to_parquet(proc / "t2a_daily_research.parquet", index=False)
    for model in ["A", "B"]:
        fp = proc / f"features_V24PUT_T1{model}.json"
        spec = json.loads(fp.read_text())
        spec["selected_features"] = list(spec["selected_features"]) + COLS
        target = proc / f"features_V24PUT_T1{model}_T2A.json"
        with target.open("x", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    print("T2A READY", coverage, historical, flush=True)


if __name__ == "__main__":
    main()
