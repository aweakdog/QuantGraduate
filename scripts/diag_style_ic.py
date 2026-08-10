"""风格因子在 A/B 两窗的 IC —— 决定值不值得花钱买 Tushare

背景: 模型在 A 窗(2020-07~2022-08)的 IC 是负的(-0.0136), 在走平的市场里
系统性跑输等权基准 30~45 个百分点。一个具体嫌疑是: 440 个特征里没有任何
行业信息, 也没有市值/估值/换手 —— fundamentals 的 pe/pb/mcap 三列全是 None,
落到训练集就是 NaN 被 train median 填掉。

而 A 窗恰好是风格极端分化的两年。如果"小市值"这类最朴素的横截面因子在 A 窗
本身有正 IC, 那我们的模型是把一个能赚钱的维度整个漏掉了 —— 那 200 元买
daily_basic 就有明确预期收益。如果连它也没用, 问题在标签/训练窗口设计,
买数据也白搭。

市值不用买: data/universe/universe_pit_2019.parquet 里就有 PIT 的 mcap/adv。

    python scripts/diag_style_ic.py
    python scripts/diag_style_ic.py --mb-only   # 只主板, 与 mb_dmw 口径一致
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "training_data_pit_2019.parquet"
UNIVERSE = ROOT / "data" / "universe" / "universe_pit_2019.parquet"

WINDOWS = {
    "A": ("2020-07-01", "2022-08-31"),
    "B": ("2022-09-01", "2026-07-27"),
}

# 模型自己的 IC(eval_grid 报告里的 ic_median), 作为对照基线
MODEL_IC = {"A": -0.0136, "B": 0.0116}

LABEL = "fwd_5d_ret"


def spearman_ic(df, factor, label=LABEL):
    """按日算横截面 Spearman IC, 返回逐日序列

    用 Spearman 而不是 Pearson: 选股只关心排序, 且 mcap 这类量级跨几个数量级,
    Pearson 会被极值主导。
    """
    out = {}
    for d, g in df.groupby("date"):
        s = g[[factor, label]].dropna()
        if len(s) < 30:
            continue
        out[d] = s[factor].corr(s[label], method="spearman")
    return pd.Series(out).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb-only", action="store_true",
                    help="只留主板(剔除创业板30/科创板688), 与 mb_dmw 口径一致")
    args = ap.parse_args()

    uni = pd.read_parquet(UNIVERSE)
    uni["effective_date"] = pd.to_datetime(uni["effective_date"])
    # 两边代码格式不同: 训练集是 "000001.SZ", 股票池是 "000001"。
    # 统一截前 6 位 —— 直接 merge 会静默得到 0 行匹配, 而不是报错。
    uni["code"] = uni["code"].astype(str).str[:6]

    cols = ["date", "code", LABEL, "ret_21d", "ret_5d", "vol_ratio",
            "pos_20", "rsi_14", "atr_pct"]
    df = pd.read_parquet(TRAIN, columns=cols)
    df["code"] = df["code"].astype(str).str[:6]
    df["date"] = pd.to_datetime(df["date"])
    if args.mb_only:
        df = df[~df["code"].str.startswith(("30", "688"))]
    print(f"训练集 {len(df):,} 行, {df['code'].nunique()} 只"
          f"{' (仅主板)' if args.mb_only else ''}")

    # PIT 合并 mcap/adv: 季度生效, 用 merge_asof 向前对齐, 不能直接 merge on date
    df = df.sort_values("date")
    u = uni[["effective_date", "code", "mcap", "adv"]].sort_values("effective_date")
    df = pd.merge_asof(df, u, left_on="date", right_on="effective_date",
                       by="code", direction="backward")
    n_mcap = int(df["mcap"].notna().sum())
    print(f"  合上 mcap 的行: {n_mcap:,}")
    if n_mcap == 0:
        raise SystemExit("ERROR: mcap 全没合上, 不要拿 nan 当结论 —— 先查代码格式")

    # 因子构造。符号统一成"值越大越应该涨", 这样正 IC = 因子有效
    df["小市值"] = -np.log(df["mcap"].clip(lower=1))
    df["低流动性"] = -np.log(df["adv"].clip(lower=1))
    df["21日反转"] = -df["ret_21d"]
    df["5日反转"] = -df["ret_5d"]
    df["低位置"] = -df["pos_20"]
    df["低换手比"] = -df["vol_ratio"]
    factors = ["小市值", "低流动性", "21日反转", "5日反转", "低位置", "低换手比"]

    print(f"\n逐日横截面 Spearman IC 中位数 (标签 {LABEL})")
    print(f"{'因子':<12}{'A窗':>10}{'B窗':>10}{'A窗正IC天数占比':>16}{'A窗天数':>9}")
    print("-" * 60)
    rows = {}
    for f in factors:
        line = {}
        for w, (s, e) in WINDOWS.items():
            sub = df[(df["date"] >= s) & (df["date"] <= e)]
            ic = spearman_ic(sub, f)
            line[w] = (ic.median() if len(ic) else np.nan, len(ic),
                       (ic > 0).mean() if len(ic) else np.nan)
        rows[f] = line
        a, b = line["A"], line["B"]
        print(f"{f:<12}{a[0]:>10.4f}{b[0]:>10.4f}{a[2]*100:>15.1f}%{a[1]:>9}")

    print("-" * 60)
    print(f"{'模型(参考)':<12}{MODEL_IC['A']:>10.4f}{MODEL_IC['B']:>10.4f}")

    print(f"\n{'='*60}\n判读")
    best = max(rows, key=lambda f: rows[f]["A"][0] if not np.isnan(rows[f]["A"][0]) else -9)
    ba = rows[best]["A"][0]
    print(f"  A 窗最强单因子: {best}  IC={ba:+.4f}  (模型 {MODEL_IC['A']:+.4f})")
    if ba > 0.01:
        print("  -> A 窗存在明显有效的横截面维度, 而模型是负 IC。")
        print("     模型漏掉了能赚钱的方向, 补风格/行业数据有明确预期收益。")
    elif ba > 0:
        print("  -> A 窗有微弱正 IC 的维度, 但不强。补数据可能有限。")
    else:
        print("  -> 连最朴素的风格因子在 A 窗也无效。问题更可能在标签定义或")
        print("     训练窗口设计上, 买数据大概率不解决。")


if __name__ == "__main__":
    main()
