from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "training_data_v23.parquet"
OUTPUT = ROOT / "data" / "processed" / "training_data_v25.parquet"
TEMP = ROOT / "data" / "processed" / "training_data_v25.tmp.parquet"


def main() -> None:
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
            "fwd_1d_t1_open_ret": kline["open"].shift(-2) / kline["open"].shift(-1) - 1,
        }))

    label_df = pd.concat(labels, ignore_index=True)
    df = df.drop(columns=["fwd_1d_t1_open_ret"], errors="ignore")
    df = df.merge(label_df, on=["date", "code"], how="left", validate="one_to_one")
    df.to_parquet(TEMP, index=False)
    TEMP.replace(OUTPUT)

    values = df["fwd_1d_t1_open_ret"].dropna()
    print(f"created: {OUTPUT}")
    print(f"rows={len(df):,} codes={df['code'].nunique()} cols={len(df.columns)}")
    print(f"label_nonnull={len(values):,} coverage={len(values) / len(df):.4%}")
    print(values.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).to_dict())


if __name__ == "__main__":
    main()
