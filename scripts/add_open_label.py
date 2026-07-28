"""
把开盘价执行标签 fwd_1d_open_ret 烤入 training_data_v23.parquet

逻辑:
- 逐股读 raw/kline/{code6}.parquet 的 open, 算 open.shift(-1)/open - 1
- 按 (date, code) merge 进现有 v23, 不重跑整条特征工程 (与 build_v23.py 同思路)
- 原子写回 (temp + os.replace), 失败不影响原文件

feature_engine.calc_labels 已同步产出该列, 本脚本用于给已有 v23 补列,
使 trainer 无需在加载期读 raw kline 也能用开盘价标签。
"""
import os
import sys
import pandas as pd

sys.path.insert(0, r"D:\myAI\WorkBuddy-workspace\quant-strategy")
from pipeline.feature_engine import read_kline  # noqa: E402

BASE = r"D:\myAI\WorkBuddy-workspace\quant-strategy\data"
V23 = os.path.join(BASE, "processed", "training_data_v23.parquet")


def main():
    df = pd.read_parquet(V23)
    print(f"v23: {len(df):,} rows, {df['code'].nunique()} stocks, {len(df.columns)} cols")

    if "fwd_1d_open_ret" in df.columns:
        print("fwd_1d_open_ret 已存在 -> 直接校验缺失率")
        miss = df["fwd_1d_open_ret"].isna().sum()
        print(f"  缺失: {miss} ({miss / len(df) * 100:.2f}%)")
        return

    rows = []
    for code in df["code"].drop_duplicates():
        code6 = str(code)[:6]
        dk = read_kline(code6)
        if dk is None or "open" not in dk.columns:
            continue
        dk = dk.copy()
        dk["date"] = pd.to_datetime(dk["date"])
        o = dk["open"].astype(float)
        tmp = pd.DataFrame({
            "date": dk["date"].values,
            "code": code,
            "fwd_1d_open_ret": (o.shift(-1) / o - 1).values,
        })
        rows.append(tmp)

    op = pd.concat(rows, ignore_index=True)
    print(f"open label 覆盖: {op['code'].nunique()} stocks, {len(op):,} rows")

    v23x = df.merge(op, on=["date", "code"], how="left")
    miss = v23x["fwd_1d_open_ret"].isna().sum()
    print(f"fwd_1d_open_ret 缺失: {miss} ({miss / len(v23x) * 100:.2f}%)")

    tmp_path = V23 + ".tmp"
    v23x.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, V23)
    print(f"已写回: {V23}  (cols={len(v23x.columns)}, rows={len(v23x):,})")
    print("新增列:", [c for c in v23x.columns if c not in df.columns])


if __name__ == "__main__":
    main()
