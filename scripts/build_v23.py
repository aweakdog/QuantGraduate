"""
生成 training_data_v23.parquet (v22 + 补齐 vol_ma20)

v23 相对 v22 的唯一差异: 新增特征 vol_ma20 = volume.rolling(20).mean()
- 复用 feature_engine.read_kline (与 calc_technical_features 内部完全相同的代码路径/数据源)
- 按 (date, code) 对齐 merge 进 v22, 不改动 v22 原有 280 列、不覆盖 v22 文件
- 输出 training_data_v23.parquet (281 特征)

等价性: feature_engine 重跑 build_all 时, vol_ma20 也是用同一 read_kline 的 volume
        做 rolling(20).mean() 得到, 故本脚本生成的 v23 与重跑 pipeline 完全一致,
        但省去整条特征工程(资金流/事件/宏观/链主)的重算开销。
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from pipeline.feature_engine import read_kline  # noqa: E402

BASE = _ROOT / "data"
V22 = BASE / "processed" / "training_data_v22.parquet"
OUT = BASE / "processed" / "training_data_v23.parquet"


def main():
    df = pd.read_parquet(V22)
    print(f"v22: {len(df):,} rows, {df['code'].nunique()} stocks, {len(df.columns)} cols")

    if "vol_ma20" in df.columns:
        # feature_engine.build_all 已直接产出 vol_ma20 (新版本), 无需二次计算
        print("v22 已含 vol_ma20 -> v23 直接复用 v22 (复制, 不再 merge 以免列名冲突)")
        v23 = df.copy()
    else:
        rows = []
        for code in df["code"].drop_duplicates():
            code6 = str(code)[:6]
            dk = read_kline(code6)
            if dk is None or "volume" not in dk.columns:
                continue
            v = dk["volume"].astype(float)
            # 与 feature_engine L253 的 vol_ma5 对称: 默认 min_periods=window(=20)
            ma20 = v.rolling(20).mean()
            tmp = pd.DataFrame(
                {"date": dk["date"].values, "code": code, "vol_ma20": ma20.values}
            )
            rows.append(tmp)

        vol_df = pd.concat(rows, ignore_index=True)
        print(f"vol_ma20 计算覆盖: {vol_df['code'].nunique()} stocks, {len(vol_df):,} rows")

        v23 = df.merge(vol_df, on=["date", "code"], how="left")
        miss = v23["vol_ma20"].isna().sum()
        print(f"vol_ma20 缺失: {miss} ({miss / len(v23) * 100:.2f}%)")

    v23.to_parquet(OUT, index=False)
    print(f"已保存: {OUT}  (cols={len(v23.columns)}, rows={len(v23):,})")
    print(f"vol_ma20 缺失: {v23['vol_ma20'].isna().sum()}")
    # 健全性: v23 相对 v22 新增列 (正常应为空, 因 vol_ma20 已在 v22)
    only_vol = [c for c in v23.columns if c not in df.columns]
    print(f"v3 相对 v22 新增列: {only_vol}")


if __name__ == "__main__":
    main()
