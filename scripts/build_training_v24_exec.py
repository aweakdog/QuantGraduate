import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description="为训练集补上 T+1 开盘买入的执行标签")
    ap.add_argument("--source", default="training_data_v23.parquet")
    ap.add_argument("--output", default="training_data_v24.parquet")
    a = ap.parse_args()
    SOURCE, OUTPUT = PROC / a.source, PROC / a.output
    TEMP = OUTPUT.with_suffix(".tmp.parquet")

    df = pd.read_parquet(SOURCE)
    df["date"] = pd.to_datetime(df["date"])
    labels = []
    for code in df["code"].drop_duplicates():
        code6 = str(code)[:6]
        kline = pd.read_parquet(ROOT / "data" / "raw" / "kline" / f"{code6}.parquet")
        kline["date"] = pd.to_datetime(kline["date"])
        kline = kline.sort_values("date").drop_duplicates("date")
        labels.append(pd.DataFrame({
            "date": kline["date"],
            "code": code,
            "fwd_1d_exec_ret": kline["close"].shift(-1) / kline["open"].shift(-1) - 1,
        }))

    label_df = pd.concat(labels, ignore_index=True)
    df = df.drop(columns=["fwd_1d_exec_ret"], errors="ignore")
    df = df.merge(label_df, on=["date", "code"], how="left", validate="one_to_one")
    df.to_parquet(TEMP, index=False)
    TEMP.replace(OUTPUT)

    values = df["fwd_1d_exec_ret"].dropna()
    print(f"created: {OUTPUT}")
    print(f"rows={len(df):,} codes={df['code'].nunique()} cols={len(df.columns)}")
    print(f"exec_label_nonnull={len(values):,} coverage={len(values) / len(df):.4%}")
    print(values.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).to_dict())
    print(f"abs_over_20pct={int((values.abs() > 0.2).sum())}")


if __name__ == "__main__":
    main()
