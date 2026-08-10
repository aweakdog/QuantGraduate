"""对 Tushare 候选字段做分层 + top5 检验 —— IC 之后的第二道关

为什么需要这一关: 上一次"低换手因子 IC 0.074"就是只过了 IC 就被当成发现,
后来分层不单调、多空为负, 白跑一轮。IC 是【全截面】的秩相关, 由中部主导;
而我们只买排名最前的 5 只, top5 完全可以和 IC 反向。
diag_entry_path.py 已经暴露过同一个陷阱: 各腿【均值】全为正但【中位数】全为负,
均值是右尾驱动的, 集中持仓吃不到那条尾巴。

三个必须同时看的东西:
  分层单调性  —— 5 层的收益要大致单调, 否则 IC 只是噪声的巧合
  top5 收益   —— 我们实际交易的位置, 这才是能不能赚钱的直接证据
  不重叠采样  —— 5日前瞻在逐日采样下重叠, t 值被放大约 sqrt(5) 倍。
                 每5天取一次才是独立样本, 两个都报, 看结论是否一致

    python scripts/scan_tushare_layers.py --main-board \
        --universe universe_pit_2019.parquet
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

WINDOWS = {"A": ("2020-07-01", "2022-08-31"), "B": ("2022-09-01", "2026-07-27")}
LABEL_H = 5
TOP_N = 5          # 与线上口径一致 (改用 5 只)
N_LAYERS = 5

# 每项的 sign 是 IC 扫描给出的符号 —— 决定"哪一端是好的": -1 表示值越小越好。
# 只测【水平项】和明确有 IC 的变化项: 池内重算后大部分"5日变化"变体的 IC 都
# 塌到 0 或翻号, 是噪声。
CANDIDATES = {
    "daily_basic": [
        ("turnover_rate",   -1, "换手率"),
        ("turnover_rate_f", -1, "自由流通换手率"),
        ("dv_ttm",          +1, "股息率TTM"),
        ("dv_ratio",        +1, "股息率"),
        ("pb",              -1, "市净率"),
        ("pe_ttm",          -1, "市盈率TTM"),
        ("pe",              -1, "市盈率"),
        ("ps_ttm",          -1, "市销率"),
        ("total_mv",        +1, "总市值"),
        ("float_share",     +1, "流通股本"),
    ],
    # 融资融券。IC 扫描里 rzmre 的两窗 t 值 -4.5/-4.6, 是至今 A 窗口最强的单因子,
    # 方向合理(融资买入多 = 杠杆投机拥挤 -> 后续跑输)。但它是【金额水平量】,
    # 极可能只是市值/换手的代理 —— 所以除了原值, 还要测【对流通市值归一】后的
    # 版本, 剥掉规模成分才知道是不是真的新信息。
    "margin_detail": [
        ("rzmre",  -1, "融资买入额"),
        ("rzche",  -1, "融资偿还额"),
        ("rzye",   -1, "融资余额"),
        ("rqye",   -1, "融券余额"),
        ("rqmcl",  -1, "融券卖出量"),
        ("rzrqye", -1, "融资融券余额"),
    ],
}

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="daily_basic", choices=list(CANDIDATES))
ap.add_argument("--main-board", action="store_true")
ap.add_argument("--universe", default="universe_pit_2019.parquet")
ap.add_argument("--top-n", type=int, default=TOP_N)
ap.add_argument("--norm-by-mv", action="store_true",
                help="把字段除以当日流通市值(circ_mv)再排序。用于判断一个金额类"
                     "字段到底是自身有信息, 还是只是规模的代理")
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
    a = pd.concat(frames, ignore_index=True)
    a["date"] = pd.to_datetime(a["date"])
    return a.pivot_table(index="date", columns="code", values="close")


def universe_mask(path, index, columns):
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


def layer_stats(fm, lab, sign, n_layers, top_n):
    """返回 (各层收益, top_n收益序列, bottom_n收益序列)

    fm 按 sign 调向后, 排名 1 = 最该买的。收益用按日期 demean 过的标签,
    所以数字是【相对池内平均】的超额, 0 就是没本事。
    """
    valid = fm.notna() & lab.notna()
    f = (fm * sign).where(valid)
    r = lab.where(valid)
    # 降序排名: 值越大(调向后)越靠前
    rk = f.rank(axis=1, ascending=False)
    n = valid.sum(axis=1)
    layers = []
    for q in range(n_layers):
        lo = n * q / n_layers
        hi = n * (q + 1) / n_layers
        sel = rk.gt(lo, axis=0) & rk.le(hi, axis=0)
        layers.append(r.where(sel).mean(axis=1))
    top = r.where(rk <= top_n).mean(axis=1)
    bot = r.where(rk.gt(n - top_n, axis=0)).mean(axis=1)
    return layers, top, bot


def fmt(series, sample):
    s = series.dropna()
    if sample > 1:
        s = s.iloc[::sample]
    if len(s) < 20:
        return "n/a", 0.0
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() else 0.0
    return f"{s.mean()*100:+.3f}%", t


def main():
    px = load_close()
    if args.main_board:
        px = px[[c for c in px.columns
                 if not c.startswith(("30", "688", "8", "4"))]]
    fwd = px.shift(-LABEL_H) / px - 1
    lab = fwd.sub(fwd.mean(axis=1), axis=0)
    mask = universe_mask(ROOT / "data" / "universe" / args.universe,
                         lab.index, lab.columns)
    lab = lab.where(mask)
    lab = lab.sub(lab.mean(axis=1), axis=0)      # 池内重新 demean
    print(f"池内每日 {mask.sum(axis=1).replace(0, np.nan).median():.0f} 只 | "
          f"top{args.top_n} | 收益均为【相对池内均值】的5日超额")

    def read_table(name):
        files = sorted(p for p in (TS / name).glob("*.parquet")
                       if p.stem != "None")
        d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        d["_d"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d",
                                 errors="coerce")
        d["_c"] = d["ts_code"].astype(str).str.split(".").str[0].str[-6:]
        return d

    db = read_table(args.table)
    mv = None
    if args.norm_by_mv:
        # 用 circ_mv(流通市值, 单位万元) 归一。归一后若 top5 超额消失, 说明原字段
        # 的信息其实来自规模而非字段本身。
        mv = read_table("daily_basic").pivot_table(
            index="_d", columns="_c", values="circ_mv")
        print("已按流通市值归一 (剥离规模成分)")
    print(f"表: {args.table}  ({len(db):,} 行)")

    for field, sign, cn in CANDIDATES[args.table]:
        if field not in db.columns:
            continue
        fm = db.pivot_table(index="_d", columns="_c", values=field)
        if mv is not None:
            fm = fm / mv.reindex(index=fm.index, columns=fm.columns)
        print(f"\n{'='*88}")
        print(f"{field} ({cn})  方向: {'值越小越好' if sign < 0 else '值越大越好'}")
        print(f"{'='*88}")
        for wname, (s, e) in WINDOWS.items():
            m = fm.loc[(fm.index >= s) & (fm.index <= e)]
            l = lab.loc[(lab.index >= s) & (lab.index <= e)]
            cols = m.columns.intersection(l.columns)
            idx = m.index.intersection(l.index)
            layers, top, bot = layer_stats(m.loc[idx, cols], l.loc[idx, cols],
                                           sign, N_LAYERS, args.top_n)
            for sample, tagn in ((1, "逐日(重叠)"), (LABEL_H, "每5日(独立)")):
                cells = []
                means = []
                for i, lay in enumerate(layers, 1):
                    v, _ = fmt(lay, sample)
                    cells.append(f"L{i} {v}")
                    sl = lay.dropna()
                    means.append(sl.iloc[::sample].mean() if len(sl) else np.nan)
                tv, tt = fmt(top, sample)
                bv, _ = fmt(bot, sample)
                ls = pd.Series(top.dropna() - bot.dropna()).iloc[::sample]
                lsv = f"{ls.mean()*100:+.3f}%" if len(ls) > 20 else "n/a"
                lst = (ls.mean() / ls.std() * np.sqrt(len(ls))
                       if len(ls) > 20 and ls.std() else 0.0)
                # 单调性: L1>L2>...>L5 的相邻对里有多少是对的
                ok = sum(1 for i in range(len(means) - 1)
                         if means[i] > means[i + 1])
                print(f"  {wname} {tagn:<12} {' | '.join(cells)}")
                print(f"  {'':<15} top{args.top_n} {tv}(t{tt:+.1f})  "
                      f"末{args.top_n} {bv}  多空 {lsv}(t{lst:+.1f})  "
                      f"单调 {ok}/{len(means)-1}")
    print("\n判读标准: top5 超额为正且 t>2、多空为正、单调至少 3/4, "
          "且【A/B 两窗口都成立】才值得接进特征集")


if __name__ == "__main__":
    main()
