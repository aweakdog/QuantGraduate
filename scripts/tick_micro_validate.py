# -*- coding: utf-8 -*-
"""校验逐笔微观特征的正确性

沪深两所编码不同, 解析错了不会报错, 只会静默产出垃圾特征 —— 必须对账。
三道硬检查:
  1. 成交额对账: 逐笔累加的 day_amt 必须等于日线成交额 (tushare daily.amount)。
     这一条能同时抓出 分交易所解析错误 / 撤单被误计成成交 / 双边重复计数。
  2. 分交易所体检: 每个特征在 SH / SZ 上的分布必须可比。若某列只在一所有值,
     或两所中位数差一个量级, 说明编码分支写错了。
  3. 覆盖率: 逐笔覆盖全市场, 每列非空率应接近 100%。低覆盖列就是新的掩码隐患
     (docs 方法论 §8.4), 要么修要么剔。

用法
────
    python scripts/tick_micro_validate.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MICRO = ROOT / "data/processed/tick_micro"


def load():
    fs = sorted(MICRO.glob("*.parquet"))
    if not fs:
        raise SystemExit(f"{MICRO} 是空的")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_daily(dates):
    """读日线成交额做对账基准。优先 tushare/daily (千元), 退化到 kline 单股文件"""
    tsd = ROOT / "data/raw/tushare/daily"
    if tsd.exists():
        fs = sorted(tsd.rglob("*.parquet"))
        if fs:
            keep = [f for f in fs if any(d in f.name for d in dates)] or fs[-40:]
            d = pd.concat([pd.read_parquet(f) for f in keep], ignore_index=True)
            return d, "tushare/daily"
    kd = ROOT / "data/raw/kline"
    if kd.exists():
        fs = sorted(kd.glob("*.parquet"))[:800]
        d = pd.concat([pd.read_parquet(f).assign(
            code=f.stem) for f in fs], ignore_index=True)
        return d, "kline"
    return None, None


def main():
    df = load()
    df["exch"] = np.where(df["code"].str.startswith("6"), "SH", "SZ")
    print(f"样本 {len(df)} 股票日  {df['date'].min()}~{df['date'].max()}  "
          f"股票 {df['code'].nunique()} 只  (SH {df.exch.eq('SH').sum()} / "
          f"SZ {df.exch.eq('SZ').sum()})")

    # ---------- 1. 成交额对账 ----------
    d, src = load_daily(sorted(df["date"].unique()))
    print(f"\n===== 1. 成交额对账 (vs {src or '找不到日线表'}) =====")
    if d is not None:
        cols = {c.lower(): c for c in d.columns}
        ccol = cols.get("code") or cols.get("ts_code")
        dcol = cols.get("date") or cols.get("trade_date")
        acol = cols.get("amount") or cols.get("amt") or cols.get("成交额")
        if not all([ccol, dcol, acol]):
            print(f"  日线表列名不认识: {list(d.columns)[:15]}")
        else:
            d = d[[ccol, dcol, acol]].copy()
            d.columns = ["code", "date", "amt_daily"]
            d["code"] = d["code"].astype(str).str.extract(r"(\d{6})")[0]
            d["date"] = pd.to_datetime(d["date"].astype(str)).dt.strftime("%Y%m%d")
            m = df.merge(d, on=["code", "date"], how="inner")
            if m.empty:
                print("  无重叠样本, 跳过")
            else:
                # tushare daily.amount 单位是千元
                for unit, nm in [(1e3, "千元"), (1e4, "万元"), (1.0, "元")]:
                    ratio = m["day_amt"] / (m["amt_daily"] * unit)
                    med = ratio.median()
                    if 0.5 < med < 2:
                        print(f"  日线单位判定为 {nm}")
                        print(f"  比值 中位 {med:.4f}  均值 {ratio.mean():.4f}  "
                              f"5%~95% [{ratio.quantile(.05):.4f}, "
                              f"{ratio.quantile(.95):.4f}]")
                        bad = (abs(ratio - 1) > 0.02)
                        print(f"  偏离>2% 的样本 {bad.sum()}/{len(m)} "
                              f"({bad.mean() * 100:.2f}%)")
                        for e in ["SH", "SZ"]:
                            s = ratio[m.exch.eq(e)]
                            print(f"    {e}: 中位 {s.median():.4f}  "
                                  f"偏离>2% {(abs(s - 1) > 0.02).mean() * 100:.2f}%")
                        break
                else:
                    print(f"  ⚠ 任何单位都对不上, 比值中位 "
                          f"{(m['day_amt'] / m['amt_daily']).median():.3g} —— 解析可能有错")

    # ---------- 2. 分交易所体检 ----------
    print("\n===== 2. 分交易所分布 (两所差一个量级 = 编码分支写错) =====")
    feats = [c for c in df.columns
             if c not in ("code", "date", "exch", "prev_close", "close", "vwap")]
    print(f"{'特征':<22}{'非空%':>7}{'SH中位':>12}{'SZ中位':>12}{'比值':>8}  警告")
    for c in feats:
        s = pd.to_numeric(df[c], errors="coerce")
        cov = s.notna().mean() * 100
        a = s[df.exch.eq("SH")].median()
        b = s[df.exch.eq("SZ")].median()
        warn = ""
        if cov < 95:
            warn += "覆盖低 "
        if pd.isna(a) or pd.isna(b):
            warn += "单所缺失 "
            rr = np.nan
        else:
            rr = a / b if b != 0 else np.nan
            if rr == rr and (abs(rr) > 5 or abs(rr) < 0.2):
                warn += "两所量级不符 "
        print(f"{c:<22}{cov:>7.1f}{a:>12.5g}{b:>12.5g}{rr:>8.2f}  {warn}")

    # ---------- 3. 分布合理性 ----------
    print("\n===== 3. 关键量的合理区间 =====")
    checks = [
        ("act_buy_ratio", 0.35, 0.65, "主动买占比应在 0.5 附近"),
        ("cxl_rate_b", 0.05, 1.5, "买方撤单率"),
        ("cxl_rate_s", 0.05, 1.5, "卖方撤单率"),
        ("spread_bp", 1, 100, "相对价差 bp"),
        ("ord_amt_ratio", 0.5, 20, "委托额/成交额, 应 >1"),
        ("amt_close30_share", 0.05, 0.35, "尾盘30min成交占比"),
    ]
    for c, lo, hi, why in checks:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            print(f"  {c:<20} 全空 ⚠")
            continue
        inr = ((s >= lo) & (s <= hi)).mean() * 100
        print(f"  {c:<20} 中位 {s.median():>9.4f}  "
              f"在[{lo},{hi}]内 {inr:>5.1f}%   {why}"
              + ("  ⚠" if inr < 80 else ""))


if __name__ == "__main__":
    main()
