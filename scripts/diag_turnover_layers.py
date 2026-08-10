"""低换手因子的可交易性检验 —— IC 0.074 里到底能落袋多少

昨晚的单因子诊断里 低换手 是唯一两窗全正、且行业中性化后仍全正的因子
(A +0.0183 / B +0.0740)。但 IC 高不等于能赚钱, 低换手常常等价于低流动性,
真实的坑在三处:
  1) 冲击成本: 换手最低那一层可能日成交额只有几千万, 我们的资金买得进去吗
  2) 涨跌停/停牌: 买不进或卖不掉的样本在 IC 里照样算, 在实盘里是拿不到的
  3) 交易成本: 5 日调仓 = 一年 48 次往返, 每次 0.52% 就是 25%/年

所以这里不看 IC, 直接做分层组合回测, 扣真实成本、剔掉不可交易的样本,
再看还剩多少。对照 mb_dmw 基线: A 窗超额 -28.6%, B 窗 +23.4%。

    python scripts/diag_turnover_layers.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "data" / "raw" / "tushare"
TRAIN = ROOT / "data" / "processed" / "training_data_pit_2019.parquet"

WINDOWS = {"A": ("2020-07-01", "2022-08-31"), "B": ("2022-09-01", "2026-07-27")}
LABEL = "fwd_5d_ret"
HOLD = 5                      # 与线上一致: 5 日持有
N_LAYER = 5
# 与 eval_grid 的 EXEC_BASE 对齐: 佣金 0.06%/边 + 滑点 0.2%/边
ROUND_TRIP = (0.0006 + 0.002) * 2
# 基线对照 (mb_dmw base3/g3_regime 的超额中位数)
BASELINE = {"A": -28.6, "B": 23.4}


def load_ts(name, cols):
    fs = sorted((TS / name).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df.drop(columns=["ts_code", "trade_date"])


def build():
    df = pd.read_parquet(TRAIN, columns=["date", "code", LABEL])
    df["code"] = df["code"].astype(str).str[:6]
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["code"].str.startswith(("30", "688"))]     # 主板, 与 mb_dmw 同口径
    print(f"训练集(主板) {len(df):,} 行, {df['code'].nunique()} 只")

    db = load_ts("daily_basic", ["ts_code", "trade_date", "turnover_rate", "circ_mv"])
    df = df.merge(db, on=["date", "code"], how="left")

    dl = load_ts("daily", ["ts_code", "trade_date", "close", "amount"])
    df = df.merge(dl, on=["date", "code"], how="left")

    lim = load_ts("stk_limit", ["ts_code", "trade_date", "up_limit", "down_limit"])
    df = df.merge(lim, on=["date", "code"], how="left")

    sus = load_ts("suspend_d", ["ts_code", "trade_date", "suspend_type"])
    sus["halted"] = True
    df = df.merge(sus[["date", "code", "halted"]], on=["date", "code"], how="left")
    df["halted"] = df["halted"].fillna(False)

    # 可买: 未停牌 且 收盘没封涨停(封死了买不进)
    # 用未复权 close 与当日涨跌停价比, 留 0.1% 容差避免浮点毛刺
    df["limit_up"] = df["close"] >= df["up_limit"] * 0.999
    df["limit_dn"] = df["close"] <= df["down_limit"] * 1.001
    df["tradable"] = ~df["halted"] & ~df["limit_up"] & df["close"].notna()

    # T+1 执行版标签: close_{t+1} -> close_{t+6}, 与线上 t1close 一致
    df = df.sort_values(["code", "date"])
    df["lab_t1"] = df.groupby("code")[LABEL].shift(-1)
    return df


def layer_backtest(sub, factor, label, ascending=True):
    """每 HOLD 天调一次仓, 按 factor 分 N_LAYER 层等权持有, 扣往返成本

    返回每层的周期收益序列。假设每次调仓 100% 换手 —— 对分层组合这是接近
    真实的(分层边界天天变), 也是保守的。
    """
    dates = sorted(sub["date"].unique())
    rebal = dates[::HOLD]
    rows = []
    for d in rebal:
        g = sub[(sub["date"] == d) & sub["tradable"]
                & sub[factor].notna() & sub[label].notna()]
        if len(g) < N_LAYER * 6:
            continue
        q = pd.qcut(g[factor].rank(method="first", ascending=ascending),
                    N_LAYER, labels=False)
        rec = {"date": d, "n": len(g)}
        for k in range(N_LAYER):
            m = q == k
            # 毛收益与净收益分开算: 毛收益看因子本身强不强,
            # 净收益看扣完一年 48 次往返后还剩什么。
            rec[f"L{k+1}_gross"] = g.loc[m, label].mean()
            rec[f"L{k+1}"] = rec[f"L{k+1}_gross"] - ROUND_TRIP
            rec[f"amt{k+1}"] = g.loc[m, "amount"].median()
        rec["bench"] = g[label].mean()          # 全池等权, 买入持有, 不扣成本
        top3 = g.nsmallest(3, factor) if ascending else g.nlargest(3, factor)
        rec["top3_gross"] = top3[label].mean()
        rec["top3"] = rec["top3_gross"] - ROUND_TRIP
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(series, n_per_year):
    """周期收益序列 -> 累计/年化/夏普"""
    s = series.dropna()
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    cum = (1 + s).prod() - 1
    yrs = len(s) / n_per_year
    ann = (1 + cum) ** (1 / yrs) - 1 if cum > -1 else -1.0
    sharpe = s.mean() / s.std() * np.sqrt(n_per_year) if s.std() > 0 else np.nan
    return cum * 100, ann * 100, sharpe


def main():
    df = build()
    n_per_year = 242 / HOLD          # 每年约 48 个调仓周期

    for label, tag in ((LABEL, "T日收盘执行(与标签对齐)"),
                       ("lab_t1", "T+1尾盘执行(与线上一致)")):
        print(f"\n{'='*78}")
        print(f"执行方式: {tag}   | 每 {HOLD} 日调仓, 往返成本 {ROUND_TRIP*100:.2f}%")
        print(f"{'='*78}")
        for w, (s, e) in WINDOWS.items():
            sub = df[(df["date"] >= s) & (df["date"] <= e)]
            r = layer_backtest(sub, "turnover_rate", label, ascending=True)
            if r.empty:
                print(f"  {w} 窗: 样本不足")
                continue
            print(f"\n  【{w} 窗】 {len(r)} 个调仓周期, 每期均 {r['n'].mean():.0f} 只可交易")
            print(f"  {'分层':<12}{'毛年化%':>9}{'净年化%':>9}{'毛夏普':>8}"
                  f"{'日均成交额(亿)':>15}")
            for k in range(N_LAYER):
                _, gann, gsh = stats(r[f"L{k+1}_gross"], n_per_year)
                _, nann, _ = stats(r[f"L{k+1}"], n_per_year)
                amt = r[f"amt{k+1}"].median() / 1e5      # 千元 -> 亿元
                name = f"L{k+1}" + ("(最低换手)" if k == 0 else
                                    "(最高换手)" if k == N_LAYER - 1 else "")
                print(f"  {name:<12}{gann:>9.1f}{nann:>9.1f}{gsh:>8.2f}{amt:>15.2f}")
            _, bann, bsh = stats(r["bench"], n_per_year)
            print(f"  {'基准等权':<12}{bann:>9.1f}{'-':>9}{bsh:>8.2f}"
                  f"        <- 买入持有, 无换手成本")
            _, tann, tsh = stats(r["top3_gross"], n_per_year)
            _, tnann, _ = stats(r["top3"], n_per_year)
            print(f"  {'最低3只':<12}{tann:>9.1f}{tnann:>9.1f}{tsh:>8.2f}"
                  f"        <- 与线上3只持仓可比")
            # 毛收益口径下的超额才能与基准比 (基准也是毛的)
            _, g1ann, _ = stats(r["L1_gross"], n_per_year)
            print(f"  L1 毛超额(对基准) = {g1ann - bann:+.1f}pp/年")
            # 多空组合: 两腿都付成本, 但成本在差里抵消, 所以用毛收益算。
            # 这是看因子单调性最干净的口径 —— 排序能力真存在就应该为正。
            ls = (r["L1_gross"] - r[f"L{N_LAYER}_gross"]).dropna()
            lann, lsh = stats(ls, n_per_year)[1], stats(ls, n_per_year)[2]
            mono = all(stats(r[f"L{k+1}_gross"], n_per_year)[1]
                       >= stats(r[f"L{k+2}_gross"], n_per_year)[1]
                       for k in range(N_LAYER - 1))
            print(f"  多空(L1-L{N_LAYER}, 毛) 年化 {lann:+.1f}% 夏普 {lsh:+.2f} | "
                  f"分层单调递减: {'是' if mono else '否'}")

    # 可交易性: 被剔除的样本占多少
    print(f"\n{'='*78}")
    print("可交易性剔除统计 (占全部样本)")
    tot = len(df)
    print(f"  停牌      {df['halted'].sum():>9,}  {df['halted'].mean()*100:.2f}%")
    print(f"  收盘涨停  {df['limit_up'].sum():>9,}  {df['limit_up'].mean()*100:.2f}%")
    print(f"  收盘跌停  {df['limit_dn'].sum():>9,}  {df['limit_dn'].mean()*100:.2f}%")
    print(f"  可交易    {df['tradable'].sum():>9,}  {df['tradable'].mean()*100:.2f}%"
          f"  (共 {tot:,} 行)")


if __name__ == "__main__":
    main()
