"""诊断: 窗口A 的 alpha 为什么消失 —— 按月对齐【模型表现】与【市场结构】

背景 (已排除的四类原因, 别再重复验证):
  不是成本      零滑点下窗口A 仍 -20.5%
  不是持有天数  相位平均后 4~15 天无一致效应
  不是相位运气  12 个换仓相位全为负
  不是缺估值特征 daily_basic 全部字段在窗口A 的 top5 超额为零或为负
所以窗口A (2020-07~2022-08) 是该时段真实缺 alpha。本脚本要回答的是:
【这个时段有什么可事前识别的特征】—— 如果有, 才谈得上做状态判别器;
如果没有, 那这套策略就只能承认是"看运气的市场依赖型"。

主要假设: 我们 440 个特征里动量类占绝大多数(ret_*/mom_*/rsi 等), 所以模型
本质上是个动量选股器。若窗口A 是反转占优/风格切换剧烈的时段, 动量系统性失效。
这个假设可检验: 直接量一个纯动量因子的分层收益, 看它是否与模型表现同步。

    python scripts/diag_window_regime.py --variant mb_dmw --seeds 20
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"
KLINE = ROOT / "data" / "raw" / "kline"
TS = ROOT / "data" / "raw" / "tushare"

WINDOWS = {"A": ("2020-07-01", "2022-08-31"), "B": ("2022-09-01", "2026-07-27")}
LABEL_H = 5
TOP_N = 5
SEEDS = [42, 7, 2024, 1, 2, 3, 5, 11, 17, 23, 29, 37, 43, 53, 61, 71, 83, 97,
         888, 1234]

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="mb_dmw")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--cache-prefix", default="preds_eval_mbdmw")
args = ap.parse_args()


def load_close():
    frames = []
    for f in KLINE.glob("*.parquet"):
        try:
            d = pd.read_parquet(f, columns=["date", "close"])
        except Exception:
            continue
        if d.empty:
            continue
        d["code"] = f.stem[-6:]
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a["date"] = pd.to_datetime(a["date"])
    return a.pivot_table(index="date", columns="code", values="close")


def main():
    seeds = SEEDS[:args.seeds]
    px = load_close()
    px = px[[c for c in px.columns if not c.startswith(("30", "688", "8", "4"))]]
    fwd = px.shift(-LABEL_H) / px - 1

    # ── 1. 从预测缓存取: 每日 IC + top5 实际超额 ──────────────
    # 缓存里 ranked 已按预测值降序, 直接取前 TOP_N 就是我们会买的那几只。
    # 超额基准用【当日池内等权】—— 池就是 ranked 全体, 这样 top5 超额直接回答
    # "选股相对不选股赚了多少", 与市场涨跌无关。
    ic_rows, ex_rows = [], []
    for win in ("A", "B"):
        for sd in seeds:
            p = PROC / f"{args.cache_prefix}_{win}_s{sd}.pkl"
            if not p.exists():
                continue
            preds = pickle.load(open(p, "rb"))["preds"]
            for dp in preds:
                d = pd.Timestamp(dp["date"])
                if d not in fwd.index:
                    continue
                row = fwd.loc[d]
                # 缓存里是 600536.SH 这种带交易所后缀的写法, K线列名是 6 位数字
                pool = [c6 for c6 in (str(c).split(".")[0][-6:]
                                      for c in dp["ranked"])
                        if c6 in row.index and not np.isnan(row[c6])]
                if len(pool) < 50:
                    continue
                base = row[pool].mean()
                top = row[pool[:TOP_N]].mean()
                ex_rows.append({"date": d, "win": win, "seed": sd,
                                "top_ex": top - base, "pool_n": len(pool)})
                if dp.get("ic") is not None and not pd.isna(dp["ic"]):
                    ic_rows.append({"date": d, "win": win, "seed": sd,
                                    "ic": dp["ic"]})
    if not ex_rows:
        raise SystemExit(f"没找到预测缓存 {args.cache_prefix}_*.pkl")
    ex = pd.DataFrame(ex_rows)
    ic = pd.DataFrame(ic_rows)
    print(f"读到 {ex['seed'].nunique()} 个种子 x {ex['date'].nunique()} 个交易日")

    # ── 2. 市场结构变量 (逐日, 只用当日及以前的信息) ───────────
    ret20 = px / px.shift(20) - 1
    ret5 = px / px.shift(5) - 1
    ma20 = px.rolling(20).mean()

    def factor_spread(sig):
        """纯因子分层价差: 信号前 20% 与后 20% 的次5日超额之差

        这是"这个风格今天赚不赚钱"的直接度量, 不含任何模型。
        """
        r = fwd.sub(fwd.mean(axis=1), axis=0)
        v = sig.where(r.notna())
        rk = v.rank(axis=1, pct=True)
        hi = r.where(rk >= 0.8).mean(axis=1)
        lo = r.where(rk <= 0.2).mean(axis=1)
        return hi - lo

    mkt = pd.DataFrame({
        "动量价差": factor_spread(ret20),      # >0 动量占优, <0 反转占优
        "短反价差": factor_spread(-ret5),      # >0 短期反转占优
        "截面离散": fwd.std(axis=1),           # 没有离散度就没有可赚的价差
        "breadth": (px > ma20).mean(axis=1),
    })
    idx_p = TS / "index_daily"
    if idx_p.exists():
        idf = pd.concat([pd.read_parquet(f) for f in idx_p.glob("*.parquet")],
                        ignore_index=True)
        idf["date"] = pd.to_datetime(idf["trade_date"].astype(str),
                                     format="%Y%m%d", errors="coerce")
        piv = idf.pivot_table(index="date", columns="ts_code", values="close")
        # 大盘/小盘相对强弱: 沪深300 vs 中证1000。风格切换的直接刻画
        if "000300.SH" in piv and "000852.SH" in piv:
            r300 = piv["000300.SH"].pct_change(20)
            r852 = piv["000852.SH"].pct_change(20)
            mkt["大盘-小盘"] = (r300 - r852).reindex(mkt.index)

    # ── 3. 按月汇总 ────────────────────────────────────────────
    ex["m"] = ex["date"].dt.to_period("M")
    ic["m"] = ic["date"].dt.to_period("M")
    # 先对种子取中位(消除种子噪声), 再按月平均
    daily = ex.groupby(["date", "win"])["top_ex"].median().reset_index()
    daily["m"] = daily["date"].dt.to_period("M")
    mon = daily.groupby(["m", "win"])["top_ex"].mean().reset_index()
    micd = ic.groupby("date")["ic"].median()
    mon["ic"] = mon["m"].map(micd.groupby(micd.index.to_period("M")).mean())
    for c in mkt.columns:
        s = mkt[c]
        mon[c] = mon["m"].map(s.groupby(s.index.to_period("M")).mean())

    print(f"\n{'='*100}")
    print("按月: 模型 top5 超额(每5日, 相对池内等权) 与市场结构")
    print(f"{'='*100}")
    print(f"{'月份':<9}{'窗':<3}{'top5超额':>10}{'模型IC':>9}{'动量价差':>10}"
          f"{'短反价差':>10}{'截面离散':>10}{'breadth':>9}{'大盘-小盘':>10}")
    ctx = [c for c in ("动量价差", "短反价差", "截面离散", "breadth", "大盘-小盘")
           if c in mon.columns]
    for _, r in mon.iterrows():
        cells = [f"{str(r['m']):<9}", f"{r['win']:<3}",
                 f"{r['top_ex']*100:>8.2f}%"]
        cells.append("      n/a" if pd.isna(r["ic"]) else f"{r['ic']:>9.4f}")
        for c in ctx:
            v = r[c]
            cells.append("       n/a" if pd.isna(v) else f"{v*100:+9.2f}%")
        print("".join(cells))

    # ── 4. 窗口级对比 + 相关性 ─────────────────────────────────
    print(f"\n{'='*100}")
    print("窗口级均值")
    print(f"{'='*100}")
    cols = ["top_ex", "ic"] + [c for c in mkt.columns if c in mon.columns]
    print(f"{'窗口':<6}" + "".join(f"{c:>12}" for c in cols))
    for win in ("A", "B"):
        sub = mon[mon["win"] == win]
        print(f"{win:<6}" + "".join(
            f"{sub[c].mean():>12.4f}" if c == "ic"
            else f"{sub[c].mean()*100:>11.3f}%" for c in cols))

    print(f"\n{'='*100}")
    print("模型 top5 超额 与 各市场变量的月度相关 (全样本 / 只看窗口A)")
    print(f"{'='*100}")
    for c in mkt.columns:
        if c not in mon.columns:
            continue
        full = mon[["top_ex", c]].dropna()
        a = mon[mon["win"] == "A"][["top_ex", c]].dropna()
        rf = full["top_ex"].corr(full[c]) if len(full) > 10 else np.nan
        ra = a["top_ex"].corr(a[c]) if len(a) > 10 else np.nan
        print(f"  {c:<12} 全样本 {rf:+.3f}   窗口A {ra:+.3f}   (n={len(full)}/{len(a)})")

    out = PROC / "diag_window_regime.csv"
    mon.to_csv(out, index=False)
    print(f"\n逐月明细 -> {out}")
    print("\n判读: 若【动量价差】在窗口A 显著为负且与 top5 超额高度正相关, "
          "则模型失效可归因于动量风格失效 —— 这是可事前观测的, 能做状态判别。"
          "\n      若各变量都解释不了, 则窗口A 的失败没有可识别前兆, "
          "状态判别器必然是事后拟合, 不该做。")


if __name__ == "__main__":
    main()
