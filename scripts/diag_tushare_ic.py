"""Tushare 新数据的因子 IC —— 这批数据到底有没有信号

昨晚回填了 daily_basic(真实换手/市值/估值)、申万 PIT 行业成分、moneyflow、
margin_detail、stk_limit。这些是训练集里【完全没有】的维度(440 个特征里
pe/pb/mcap 三列全 None, 零个行业信息)。

在把它们接进特征管线之前, 先用最直接的方式回答: 它们在 A/B 两窗有没有 IC。
接进管线要改 feature_engine + 重建训练集 + 重跑 40 个缓存, 代价很大;
如果单因子 IC 都接近零, 那就别做。

对照基准: 模型自己的 IC 中位数 A=-0.0136 / B=+0.0116。

    python scripts/diag_tushare_ic.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "data" / "raw" / "tushare"
TRAIN = ROOT / "data" / "processed" / "training_data_pit_2019.parquet"

WINDOWS = {"A": ("2020-07-01", "2022-08-31"), "B": ("2022-09-01", "2026-07-27")}
MODEL_IC = {"A": -0.0136, "B": 0.0116}
LABEL = "fwd_5d_ret"


def load_ts(name, cols):
    fs = sorted((TS / name).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df.drop(columns=["ts_code", "trade_date"])


def sw_industry():
    """申万 PIT 行业: 按 in_date/out_date 决定某天属于哪个一级行业"""
    m = pd.read_parquet(TS / "sw_member" / "sw_member.parquet")
    m["code"] = m["ts_code"].astype(str).str[:6]
    m["in_date"] = pd.to_datetime(m["in_date"], errors="coerce")
    m["out_date"] = pd.to_datetime(m["out_date"], errors="coerce")
    # out_date 为空 = 至今仍在该行业
    m["out_date"] = m["out_date"].fillna(pd.Timestamp("2099-12-31"))
    return m[["code", "l1_name", "in_date", "out_date"]].dropna(subset=["l1_name"])


def daily_ic(df, factor, label=LABEL, by_industry=False):
    """逐日横截面 Spearman IC; by_industry=True 时先做行业内去均值"""
    x = df[[factor, label] + (["l1_name"] if by_industry else [])].copy()
    if by_industry:
        # 行业中性化: 减去当日同行业均值, 剩下的才是"行业内选股"的能力
        g = df.groupby(["date", "l1_name"])[factor]
        x[factor] = df[factor] - g.transform("mean")
    x["date"] = df["date"]
    out = {}
    for d, g in x.groupby("date"):
        s = g[[factor, label]].dropna()
        if len(s) < 30:
            continue
        out[d] = s[factor].corr(s[label], method="spearman")
    return pd.Series(out).dropna()


def main():
    print("载入训练集标签 ...")
    df = pd.read_parquet(TRAIN, columns=["date", "code", LABEL])
    df["code"] = df["code"].astype(str).str[:6]
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["code"].str.startswith(("30", "688"))]      # 主板口径, 与 mb_dmw 一致
    print(f"  {len(df):,} 行, {df['code'].nunique()} 只")

    print("合 daily_basic ...")
    db = load_ts("daily_basic", ["ts_code", "trade_date", "turnover_rate",
                                 "turnover_rate_f", "volume_ratio", "pe_ttm",
                                 "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"])
    df = df.merge(db, on=["date", "code"], how="left")
    print(f"  合上 {df['circ_mv'].notna().sum():,} 行")

    print("合 moneyflow ...")
    mf = load_ts("moneyflow", ["ts_code", "trade_date", "buy_lg_amount",
                               "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"])
    mf["net_lg"] = (mf["buy_lg_amount"] - mf["sell_lg_amount"]
                    + mf["buy_elg_amount"] - mf["sell_elg_amount"])
    df = df.merge(mf[["date", "code", "net_lg"]], on=["date", "code"], how="left")

    print("合申万 PIT 行业 ...")
    sw = sw_industry()
    df = df.merge(sw, on="code", how="left")
    ok = (df["date"] >= df["in_date"]) & (df["date"] < df["out_date"])
    df.loc[~ok.fillna(False), "l1_name"] = np.nan
    df = df.drop(columns=["in_date", "out_date"]).drop_duplicates(["date", "code"])
    print(f"  有行业归属 {df['l1_name'].notna().sum():,} 行, "
          f"{df['l1_name'].nunique()} 个一级行业")

    # 因子方向统一成"值越大越该涨"
    df["低换手"] = -df["turnover_rate"]
    df["低流通市值"] = -np.log(df["circ_mv"].clip(lower=1))
    df["低PE"] = -df["pe_ttm"].clip(0, 200)
    df["低PB"] = -df["pb"].clip(0, 30)
    df["高股息"] = df["dv_ttm"]
    df["大单净流入"] = df["net_lg"] / df["circ_mv"].clip(lower=1)
    df["量比"] = df["volume_ratio"]
    factors = ["低换手", "低流通市值", "低PE", "低PB", "高股息", "大单净流入", "量比"]

    print(f"\n逐日横截面 Spearman IC 中位数 (标签 {LABEL}, 主板)")
    print(f"{'因子':<14}{'A窗':>9}{'B窗':>9}{'A窗行业内':>11}{'B窗行业内':>11}")
    print("-" * 56)
    for f in factors:
        row = []
        for neutral in (False, True):
            for w, (s, e) in WINDOWS.items():
                sub = df[(df["date"] >= s) & (df["date"] <= e)]
                if neutral:
                    sub = sub[sub["l1_name"].notna()]
                ic = daily_ic(sub, f, by_industry=neutral)
                row.append(ic.median() if len(ic) else np.nan)
        print(f"{f:<14}{row[0]:>9.4f}{row[1]:>9.4f}{row[2]:>11.4f}{row[3]:>11.4f}")
    print("-" * 56)
    print(f"{'模型(参考)':<14}{MODEL_IC['A']:>9.4f}{MODEL_IC['B']:>9.4f}")

    # 行业本身有没有动量: 用当日所属行业过去 20 日均收益当因子
    print("\n行业动量(所属行业近20日收益)的 IC:")
    d2 = df[df["l1_name"].notna()].copy()
    ind_ret = d2.groupby(["date", "l1_name"])[LABEL].mean().reset_index()
    ind_ret = ind_ret.rename(columns={LABEL: "ind_fwd"})
    d2 = d2.merge(ind_ret, on=["date", "l1_name"], how="left")
    for w, (s, e) in WINDOWS.items():
        sub = d2[(d2["date"] >= s) & (d2["date"] <= e)]
        ic = daily_ic(sub, "ind_fwd")
        print(f"  {w}窗 行业平均收益 vs 个股收益 IC 中位数 = {ic.median():.4f} "
              f"({len(ic)} 天)  <- 衡量'选对行业'能解释多少个股收益")


if __name__ == "__main__":
    main()
