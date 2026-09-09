# -*- coding: utf-8 -*-
"""T1A 二期(研究臂): 在生产矩阵上追加 10 列订单结构扩列, 并生成两份特征表

源矩阵 = 生产矩阵 training_data_pit_v24_tick1 (已含 t1a_*_ma5 lag1 与 t1b_* 8 列)。
追加列(全部 lag1, 与生产 T1A 同一 PIT 口径; 逐笔包 D+1 早晨才到):

  新信息 4 列 (t1a2_daily 面板, scripts/t1a2_order_features.py 抽的):
    t1a2_dur_buy_ma5 / t1a2_dur_sell_ma5     漫长订单占比 ma5
    t1a2_pbig_buy_ma5 / t1a2_pbig_sell_ma5   被动大单占比 ma5
  长窗 4 列 (t1a_daily 面板原始 4 列, 生产只用了 ma5):
    t1a_big_buy_ma20 / t1a_long_buy_ma20 / t1a_big_sell_ma20 / t1a_long_sell_ma20
  净差 2 列 (直接由矩阵里的生产列相减, lag 自然一致):
    t1a_big_net_ma5 = t1a_big_buy_ma5 - t1a_big_sell_ma5
    t1a_long_net_ma5 = t1a_long_buy_ma5 - t1a_long_sell_ma5

特征表(基础 = 生产 features_V24PUT_T1A.json 84 列, 只追加不改序):
    features_V24PUT_T1A2n.json  84 + 新信息 4 = 88   ("新信息"臂: 纯增量信息值多少)
    features_V24PUT_T1A2f.json  84 + 全部 10 = 94    ("全扩"臂)
基线臂直接用 features_V24PUT_T1A.json 在**同一份**输出矩阵上训(09-03 不变性检验:
未用列不改种子路径), 三臂同夜同矛阵同 te。

用法(041):
    python scripts/build_t1a2_augmented.py
    python scripts/build_t1a2_augmented.py --source training_data_pit_v24_tick1.parquet \
        --output training_data_pit_v24_tick1_t1a2.parquet
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

T1A_RAW = ["t1a_big_buy", "t1a_long_buy", "t1a_big_sell", "t1a_long_sell"]
T1A2_RAW = ["t1a2_dur_buy", "t1a2_dur_sell", "t1a2_pbig_buy", "t1a2_pbig_sell"]
NEW_MA5 = [c + "_ma5" for c in T1A2_RAW]
LONG_MA20 = [c + "_ma20" for c in T1A_RAW]
NET_MA5 = ["t1a_big_net_ma5", "t1a_long_net_ma5"]
ALL_NEW = NEW_MA5 + LONG_MA20 + NET_MA5
LAG = 1


def panel_frame(panel_dir: Path, raw_cols, window: int, min_periods: int, lag: int):
    """日度面板 -> 按 code 滚动均值 -> 滞后 lag 天 的 (date, _c6) 表"""
    files = sorted(panel_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"面板为空: {panel_dir}")
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    panel["_c6"] = panel["code"].astype(str).str.zfill(6)
    panel = panel.sort_values(["_c6", "date"])
    miss = [c for c in raw_cols if c not in panel.columns]
    if miss:
        raise RuntimeError(f"{panel_dir.name} 缺列 {miss}")
    out_cols = []
    for c in raw_cols:
        oc = f"{c}_ma{window}"
        panel[oc] = panel.groupby("_c6")[c].transform(
            lambda s: s.rolling(window, min_periods=min_periods).mean())
        out_cols.append(oc)
    z = panel[["_c6", "date"] + out_cols].copy()
    if lag > 0:
        z[out_cols] = z.groupby("_c6", sort=False)[out_cols].shift(lag)
    print(f"  面板 {panel_dir.name}: {len(files)} 天 {panel['_c6'].nunique()} 只 "
          f"(末日 {panel['date'].max():%Y-%m-%d}) -> {out_cols} lag={lag}")
    return z.reset_index(drop=True)


def write_feature_tables(base_from: str):
    base = json.loads((PROC / base_from).read_text(encoding="utf-8"))
    feats = list(base["selected_features"])
    for tag, cols in (("T1A2n", NEW_MA5), ("T1A2f", ALL_NEW)):
        dup = [c for c in cols if c in feats]
        if dup:
            raise RuntimeError(f"基础特征表里已有 {dup}")
        p = PROC / f"features_V24PUT_{tag}.json"
        p.write_text(json.dumps({"selected_features": feats + cols}, ensure_ascii=False),
                     encoding="utf-8")
        print(f"  特征表 {p.name}: {len(feats)} + {len(cols)} = {len(feats) + len(cols)} 列")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24_tick1.parquet")
    ap.add_argument("--output", default="training_data_pit_v24_tick1_t1a2.parquet")
    ap.add_argument("--features-base", default="features_V24PUT_T1A.json")
    a = ap.parse_args()
    t0 = time.time()

    src = PROC / a.source
    mat = pd.read_parquet(src)
    mat["date"] = pd.to_datetime(mat["date"])
    need = [c + "_ma5" for c in T1A_RAW]
    miss = [c for c in need if c not in mat.columns]
    if miss:
        print(f"ERROR: 源矩阵缺生产 T1A 列 {miss}, 先跑 build_t1_augmented")
        return 2
    mat = mat.drop(columns=[c for c in ALL_NEW if c in mat.columns])
    mat["_c6"] = mat["code"].astype(str).str.extract(r"(\d{6})")[0]
    n0 = len(mat)
    print(f"矩阵 {n0:,} 行 x {mat.shape[1]} 列 (源 {src.name})")

    frames = [
        panel_frame(PROC / "t1a2_daily", T1A2_RAW, 5, 3, LAG),
        panel_frame(PROC / "t1a_daily", T1A_RAW, 20, 10, LAG),
    ]
    out = mat
    for f in frames:
        if f.duplicated(["date", "_c6"]).any():
            print("ERROR: 面板 (date, code) 有重复")
            return 2
        out = out.merge(f, on=["date", "_c6"], how="left")
        if len(out) != n0:
            print(f"ERROR: 合并后行数变了 {n0:,} -> {len(out):,}")
            return 2
    out["t1a_big_net_ma5"] = out["t1a_big_buy_ma5"] - out["t1a_big_sell_ma5"]
    out["t1a_long_net_ma5"] = out["t1a_long_buy_ma5"] - out["t1a_long_sell_ma5"]
    out = out.drop(columns=["_c6"])

    out_path = PROC / a.output
    tmp = out_path.with_suffix(".parquet.t1a2tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    write_feature_tables(a.features_base)

    b_win = out["date"] >= "2022-09-01"
    cov = {c: f"{out.loc[b_win, c].notna().mean():.1%}" for c in ALL_NEW}
    last = out[out["date"] == out["date"].max()]
    cov_last = last[ALL_NEW].notna().all(axis=1).mean()
    print(f"T1A2 增广完成 -> {out_path.name} ({out.shape[0]:,} x {out.shape[1]}) "
          f"耗时 {time.time() - t0:.0f}s")
    print(f"2022-09 起非空率: {cov}")
    print(f"最新日 {out['date'].max():%Y-%m-%d} 全 10 列覆盖 {cov_last:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
