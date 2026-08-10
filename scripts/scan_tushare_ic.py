"""扫描 Tushare 落地数据里每个字段对我们标签的截面预测力 (rank IC)。

为什么要先做这个: 把新特征接进 feature_engine + 重建训练矩阵 + 重跑特征筛选
+ 重建 40 份缓存, 是几小时的机时。而"这批数据里到底有没有 alpha"可以用几分钟
的 IC 扫描先回答。已有教训: 低换手因子 IC 0.074 看着很高, 但分层不单调、
多空为负, 白跑了一轮。

关键设计:
  分 A/B 两窗口分别报 —— 我们的核心问题就是 A/B 不一致(A窗零滑点也亏 20%),
    只在单个窗口有 IC 的特征没有价值, 要找【两窗同号】的。
  报覆盖率 —— 之前发现 con_*/tev_* 常年只覆盖 10~17%, 模型学的是"这只股票
    在不在数据源里"而不是因子本身。覆盖率低于 60% 的字段要打问号。
  同时报【原始值】和【5日变化】—— 换手率/估值这类字段, 通常水平项弱、
    变化项强(水平项主要是行业和市值的代理)。
  用 rank IC (Spearman) 而不是 Pearson —— 估值类字段有极端值和负值(PE为负),
    Pearson 会被少数异常股主导。

    python scripts/scan_tushare_ic.py --tables daily_basic
    python scripts/scan_tushare_ic.py --tables daily_basic,moneyflow --main-board
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TS = ROOT / "data" / "raw" / "tushare"
KLINE = ROOT / "data" / "raw" / "kline"

# 与 eval_grid.py 的两个评测窗口严格一致
WINDOWS = {"A": ("2020-07-01", "2022-08-31"), "B": ("2022-09-01", "2026-07-27")}
LABEL_H = 5          # 与线上 label=5d 一致

# 这些列是标识/价格本身, 不是候选因子
SKIP_COLS = {"ts_code", "trade_date", "ann_date", "end_date", "code", "date",
             "close", "pre_close", "open", "high", "low", "symbol", "name"}

ap = argparse.ArgumentParser()
ap.add_argument("--tables", default="daily_basic",
                help="逗号分隔, 如 daily_basic,moneyflow,margin_detail")
ap.add_argument("--main-board", action="store_true",
                help="只算主板(剔除 30/688/8/4 开头), 与线上4条线一致")
ap.add_argument("--min-names", type=int, default=200,
                help="截面上至少这么多只有值才算这天的 IC")
ap.add_argument("--universe", default="",
                help="限制到 PIT 股票池 parquet(如 universe_pit_2019.parquet)。"
                     "【重要】不加这个参数是在 3700 只全主板上算 IC, 但我们实际"
                     "只在池子筛出的约 300 只大市值高流动股里选 —— 换手/市值/"
                     "估值的离散度在池内小得多, IC 会大幅衰减。不限制的数字会高估")
args = ap.parse_args()


def load_close():
    frames = []
    for f in KLINE.glob("*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["date", "close"])
        except Exception:
            continue
        if df.empty:
            continue
        df["code"] = f.stem[-6:]
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df.pivot_table(index="date", columns="code", values="close")


def build_label(px):
    """fwd_5d_ret 按日期 demean —— 与 wf_v35 的标签处理一致"""
    fwd = px.shift(-LABEL_H) / px - 1
    return fwd.sub(fwd.mean(axis=1), axis=0)


def universe_mask(path, index, columns):
    """把 PIT 池展开成 date x code 的布尔掩码。

    池子是按 effective_date 的快照(每期约300只), 每个交易日生效的是【不晚于
    它的最近一期】—— 这与 wf_v35 取池的口径一致, 不能用未来那期的成分。
    """
    u = pd.read_parquet(path)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    u["code"] = u["code"].astype(str).str[-6:]
    mask = pd.DataFrame(False, index=index, columns=columns)
    effs = sorted(u["effective_date"].unique())
    for i, eff in enumerate(effs):
        end = effs[i + 1] if i + 1 < len(effs) else index.max() + pd.Timedelta(days=1)
        codes = [c for c in u.loc[u["effective_date"] == eff, "code"].unique()
                 if c in mask.columns]
        sel = (index >= eff) & (index < end)
        if sel.any() and codes:
            mask.loc[sel, codes] = True
    return mask


def rank_ic(fm, lm, min_names):
    """逐日截面 Spearman: 先按行转秩再算行间相关, 比循环 scipy 快几十倍"""
    common = fm.index.intersection(lm.index)
    fm, lm = fm.loc[common], lm.loc[common]
    cols = fm.columns.intersection(lm.columns)
    fm, lm = fm[cols], lm[cols]
    valid = fm.notna() & lm.notna()
    fr = fm.where(valid).rank(axis=1)
    lr = lm.where(valid).rank(axis=1)
    n = valid.sum(axis=1)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    lr = lr.sub(lr.mean(axis=1), axis=0)
    num = (fr * lr).sum(axis=1)
    den = np.sqrt((fr ** 2).sum(axis=1) * (lr ** 2).sum(axis=1))
    ic = (num / den).where(n >= min_names)
    return ic.dropna()


def main():
    print("加载K线并构造标签 ...", flush=True)
    px = load_close()
    if args.main_board:
        keep = [c for c in px.columns if not c.startswith(("30", "688", "8", "4"))]
        px = px[keep]
    lab = build_label(px)
    print(f"  {px.shape[1]} 只 x {px.shape[0]} 天"
          f"{' (仅主板)' if args.main_board else ''}")
    if args.universe:
        p = ROOT / "data" / "universe" / args.universe
        mask = universe_mask(p, lab.index, lab.columns)
        # 标签在池外置 NaN -> rank_ic 的 valid 掩码会自动只在池内算截面
        lab = lab.where(mask)
        # 标签必须【在池内重新 demean】: 原来是对全主板取的均值, 池内截面的
        # 基准不同, 不重算会把"池整体 vs 全市场"的偏移混进 IC
        lab = lab.sub(lab.mean(axis=1), axis=0)
        n_per_day = mask.sum(axis=1)
        print(f"  已限制到 PIT 池 {args.universe}: "
              f"每日池内 {n_per_day[n_per_day > 0].median():.0f} 只(中位)")

    rows = []
    for table in [t.strip() for t in args.tables.split(",") if t.strip()]:
        d = TS / table
        files = sorted(p for p in d.glob("*.parquet") if p.stem != "None")
        if not files:
            print(f"  跳过 {table}: 无数据")
            continue
        print(f"\n加载 {table} ({len(files)} 个年份文件) ...", flush=True)
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        dcol = "trade_date" if "trade_date" in df.columns else "ann_date"
        df["_d"] = pd.to_datetime(df[dcol].astype(str), format="%Y%m%d",
                                  errors="coerce")
        df["_c"] = df["ts_code"].astype(str).str.split(".").str[0].str[-6:]
        df = df.dropna(subset=["_d"])
        cands = [c for c in df.columns
                 if c not in SKIP_COLS and not c.startswith("_")
                 and pd.api.types.is_numeric_dtype(df[c])]
        print(f"  {len(df):,} 行, {len(cands)} 个数值候选字段")

        for col in cands:
            try:
                fm = df.pivot_table(index="_d", columns="_c", values=col)
            except Exception as e:
                print(f"    {col}: pivot 失败 {str(e)[:50]}")
                continue
            # 覆盖率: 在标签有值的格子里, 该字段有多少非空
            for variant, mat in (("原始", fm), ("5日变化", fm - fm.shift(5))):
                rec = {"table": table, "field": col, "variant": variant}
                for wname, (s, e) in WINDOWS.items():
                    m = mat.loc[(mat.index >= s) & (mat.index <= e)]
                    l = lab.loc[(lab.index >= s) & (lab.index <= e)]
                    ic = rank_ic(m, l, args.min_names)
                    if len(ic) < 30:
                        rec[f"ic_{wname}"] = np.nan
                        rec[f"t_{wname}"] = np.nan
                        continue
                    rec[f"ic_{wname}"] = ic.mean()
                    rec[f"t_{wname}"] = (ic.mean() / ic.std()
                                         * np.sqrt(len(ic)) if ic.std() else 0)
                    rec[f"n_{wname}"] = len(ic)
                cm = fm.loc[(fm.index >= WINDOWS["B"][0])]
                rec["cover"] = float(cm.notna().mean().mean()) if len(cm) else 0.0
                rows.append(rec)
            print(f"    {col:<18} 完成", flush=True)

    if not rows:
        raise SystemExit("没有算出任何结果")
    res = pd.DataFrame(rows)
    res["同号"] = np.sign(res["ic_A"]) == np.sign(res["ic_B"])
    # 排序键: 两窗口 IC 绝对值的较小者 —— 挑的是下限, 单窗口强不算
    res["下限"] = res[["ic_A", "ic_B"]].abs().min(axis=1)
    res.loc[~res["同号"], "下限"] = 0.0
    res = res.sort_values("下限", ascending=False)

    out = ROOT / "data" / "processed" / "tushare_ic_scan.csv"
    res.to_csv(out, index=False)

    print(f"\n{'='*94}")
    print("按【两窗口同号且较弱那侧的 IC】排序 —— 只在单窗口有效的一律排到最后")
    print(f"{'='*94}")
    print(f"{'字段':<20}{'变体':<9}{'IC_A':>9}{'t_A':>7}{'IC_B':>9}{'t_B':>7}"
          f"{'同号':>6}{'覆盖':>8}")
    for r in res.head(40).itertuples():
        print(f"{r.field:<20}{r.variant:<9}{r.ic_A:>9.4f}{r.t_A:>7.1f}"
              f"{r.ic_B:>9.4f}{r.t_B:>7.1f}"
              f"{'是' if getattr(r, '同号') else '否':>5}"
              f"{r.cover*100:>7.0f}%")
    print(f"\n完整结果 -> {out}")
    print("参考: 现有 80 特征模型整体 IC 是 A -0.0136 / B +0.0116, "
          "单特征 IC 超过 0.02 且两窗同号就值得接进去")


if __name__ == "__main__":
    main()
