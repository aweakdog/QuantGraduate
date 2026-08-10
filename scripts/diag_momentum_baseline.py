"""纯动量基线 + "推荐都在涨"这句话的信息含量。

两个要回答的问题
────────────────
1. "过去一周推荐的几乎都在涨" 能说明什么?
   在 A 股, 涨跌是高度同步的。如果某周 80% 的股票都在涨, 那么随便选 5 只、
   有 4 只以上在涨的概率就有 74% —— 这时候"推荐都在涨"是行情的属性, 不是
   模型的属性。所以必须先算出【同期全市场的上涨比例】做基准率, 再看选出来
   的票有没有超出它。这里给出历史上每周上涨比例的分布, 以及由它推出的随机
   选股"命中率"。

2. 退回到纯动量策略, 有没有效?
   现有模型的重要性里波动率占 28%、趋势动量占 22%, 本来就偏动量。这里剥掉
   机器学习, 直接用最朴素的几种动量/反转定义跑一遍, 看在 2020-2026 的 A 股
   上到底是动量还是反转占优 —— 如果连纯动量都没有正超额, 那"暂时相信动量"
   就没有依据。

必须先有随机选股对照组 (2026-08-07 补)
──────────────────────────────────────
首版跑出"近5日涨幅 -63.0%"和"近5日反转 -63.4%"—— 两个【方向完全相反】的信号
亏得几乎一样多。这种对称性通常意味着差距不来自信号, 而来自所有配置共有的东西:
仓位/成本/相位逻辑的 bug, 或者 topN=5 相对全池等权的固有拖累。单看这两个数字
无法分辨, 因此结论当时被判定为不可用。

对照组的作用是把"机器的拖累"量化出来: 随机打分走完全相同的 run_signal, 若随机
组也亏到 -60% 附近, 那 -63% 里就几乎没有信号的成分。所有信号因此改为报
【vs 随机组】的差额, 而不是 vs 池内等权基准。

对照组当场抓出的第二个错误
──────────────────────────
加上对照组后, 20日动量与20日反转(方向完全相反)同时"跑赢随机"约 55pp —— 又是
同一种对称性。追下去发现是 run_signal 自己的 bug: 相位聚合用了中位数, 而随机
信号的相位近乎独立、持续性信号的相位高度相关, 中位数压低前者远多于后者。已改
为均值(见 run_signal 内注释)。

教训: 每次出现"方向相反的两个配置表现雷同"就说明差异来自共同的机制而非信号,
必须先查机器。这个脚本因此连续抓出两个问题。

口径
────
- 股票池: PIT 成分 + 剔除创业板/科创板(与线上账户权限一致)。
- 成交: 信号日 T 收盘算分数, T+1 收盘买入, 持有 HOLD 个交易日后 T+1 收盘卖出。
  与回测引擎的 t1close 一致。
- 成本: 每边 trade_cost + slippage, 与 eval_grid 的默认值一致(0.06% + 0.05%)。
- 相位: 换仓日有 HOLD 种相位, 全部跑一遍取【均值】(不能用中位数, 见
  run_signal 内注释)。均值即 1/HOLD 资金跑每个相位的分档阶梓换仓。
  ➔ 因此组合实际同时持有最多 HOLD x TOPN 只(默认 5x5=25), 而不是 5 只。
    这比线上的单相位 5 只分散得多, 数字不能直接当成线上预期。
- 基准: 池内等权买入持有, 与引擎同口径。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KLINE = ROOT / "data" / "raw" / "kline"
UNIV = ROOT / "data" / "universe"

SIGNALS = {
    "mom_5d":       ("近5日涨幅", lambda r: r["c"] / r["c"].shift(5) - 1),
    "mom_20d":      ("近20日涨幅", lambda r: r["c"] / r["c"].shift(20) - 1),
    "mom_60d":      ("近60日涨幅", lambda r: r["c"] / r["c"].shift(60) - 1),
    "mom_20d_sk5":  ("近20日涨幅(跳过最近5日)",
                     lambda r: r["c"].shift(5) / r["c"].shift(25) - 1),
    "rev_5d":       ("近5日跌幅(反转)", lambda r: -(r["c"] / r["c"].shift(5) - 1)),
    "rev_20d":      ("近20日跌幅(反转)", lambda r: -(r["c"] / r["c"].shift(20) - 1)),
}


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
    if not frames:
        raise SystemExit(f"{KLINE} 下没有可读K线")
    a = pd.concat(frames, ignore_index=True)
    a["date"] = pd.to_datetime(a["date"])
    return a.pivot_table(index="date", columns="code", values="close")


def pit_mask(px, universe, skip_boards):
    """date x code 的可交易布尔矩阵"""
    u = pd.read_parquet(UNIV / universe)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    eff = pd.DatetimeIndex(sorted(u["effective_date"].unique()))
    members = {d: set(g["code"].astype(str).str.zfill(6))
               for d, g in u.groupby("effective_date")}
    cols = list(px.columns)
    m = pd.DataFrame(False, index=px.index, columns=cols)
    per = np.searchsorted(eff, px.index.values, side="right") - 1
    for i, d in enumerate(eff):
        rows = per == i
        if not rows.any():
            continue
        ok = np.array([c in members[pd.Timestamp(d)] for c in cols])
        m.loc[rows, :] = np.broadcast_to(ok, (rows.sum(), len(cols)))
    if skip_boards:
        bad = np.array([c.startswith(tuple(skip_boards)) for c in cols])
        m.loc[:, bad] = False
    return m & px.notna()


def random_score(px, seed):
    """随机打分 —— 对照组的关键: 它走的是与真信号【完全相同】的 run_signal。

    如果随机组的收益也远低于池内等权基准, 说明差距来自机器本身(仓位/成本/
    相位逻辑有 bug, 或 topN 集中度与成本的固有拖累), 而不是信号没用。
    反之若随机组贴近基准, 机器就是可信的, 那么真信号的亏损才是真结论。
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random(px.shape), index=px.index, columns=px.columns)


def stats(r, n):
    cum = (1 + r).cumprod()
    return {
        "tot": cum.iloc[-1] - 1,
        "ann": (1 + r).prod() ** (252 / n) - 1,
        "sh": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
        "dd": (cum / cum.expanding().max() - 1).min(),
    }


def run_signal(score, px, tradable, hold, topn, cost):
    """返回 (含成本日收益, 不含成本日收益)。对 hold 种相位各跑一遍逐日取中位。

    为什么要同时给毛收益: 一个 topN=5 的组合相对 238 只等权基准, 天然有两层
    拖累 —— 交易成本, 以及集中度带来的波动率拖累(几何收益低于算术收益)。
    只看净收益无法分辨"信号差"和"这两层拖累"。毛收益剥掉第一层。
    """
    dates = px.index
    # fill_method=None: 不做前向填充。默认的 pad 会把停牌日缺失价格补成前一日,
    # 使停牌期间的收益变成 0 而非缺失, 等于凭空给组合加了无风险持有期。
    ret1 = px.pct_change(fill_method=None)
    curves, gross = [], []
    for off in range(hold):
        pos = pd.DataFrame(0.0, index=dates, columns=px.columns)
        turn = pd.Series(0.0, index=dates)
        held = []
        for i in range(max(60, hold), len(dates) - 1):
            if (i - off) % hold != 0:
                continue
            s = score.iloc[i].where(tradable.iloc[i])
            pick = list(s.nlargest(topn).index) if s.notna().any() else []
            if not pick:
                continue
            # T+1 收盘换仓 -> 从 i+1 起持有到下一个换仓日的 i+1
            j0 = i + 1
            j1 = min(i + 1 + hold, len(dates) - 1)
            if j0 >= j1:
                continue
            pos.iloc[j0 + 1:j1 + 1] = 0.0
            for c in pick:
                pos.iloc[j0 + 1:j1 + 1, pos.columns.get_loc(c)] = 1.0 / len(pick)
            # 换手: 与上一期的重叠决定成本
            new = set(pick)
            old = set(held)
            frac = len(new - old) / len(new) if new else 0.0
            turn.iloc[j0] = frac * 2      # 卖旧 + 买新, 各一边
            held = pick
        g = (pos * ret1).sum(axis=1)
        curves.append(g - turn * cost)
        gross.append(g)
    # 相位聚合必须用【均值】而不是中位数。
    # 均值 = 1/hold 资金分别跑每个相位 = 分档阶梯换仓, 是真实可交易的组合。
    # 中位数则不对应任何组合, 而且"逐日取中位再复利"存在系统性偏差: 随机信号的
    # hold 个相位近乎独立, 持续性信号的相位高度相关, 中位数对前者的压低远强于
    # 后者 —— 会凭空造出"任何有持续性的信号都跑赢随机"的假象。首版就踩了这个坑:
    # 20日动量与20日反转(方向相反)同时"跑赢随机"约 55pp, 全部是这个偏差。
    return (pd.concat(curves, axis=1).mean(axis=1),
            pd.concat(gross, axis=1).mean(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--universe", default="universe_pit_2019.parquet")
    ap.add_argument("--skip-boards", default="30,688")
    ap.add_argument("--trade-cost", type=float, default=0.0006)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--start", default="2020-07-01")
    ap.add_argument("--rand-seeds", default="1,2,3,4,5,6,7,8,9,10",
                    help="随机对照组的种子。单个种子毫无意义, 必须看分布")
    a = ap.parse_args()

    skip = tuple(x for x in a.skip_boards.split(",") if x)
    cost = a.trade_cost + a.slippage
    print("加载K线 ...", flush=True)
    px = load_close()
    px = px[px.index >= pd.Timestamp("2020-01-01")]
    print(f"  {px.shape[1]} 只 x {px.shape[0]} 天  ({px.index.min().date()} ~ "
          f"{px.index.max().date()})")
    trad = pit_mask(px, a.universe, skip)
    # 从未进池的股票永远不会被选中, 基准也按 trad 掩码取均值 -> 裁掉不改变任何
    # 数字, 只是把列数从近 6000 降到几百, 否则多种子对照组跑不完。
    keep = trad.any(axis=0)
    px, trad = px.loc[:, keep], trad.loc[:, keep]
    print(f"  曾进池 {int(keep.sum())} 只 | 可交易股票数中位: "
          f"{int(trad.sum(axis=1).median())}")

    ret1 = px.pct_change(fill_method=None).where(trad)
    bench = ret1.mean(axis=1)
    sub = px.index >= pd.Timestamp(a.start)

    # ── 1. 普涨基准率 ──
    print(f"\n{'=' * 76}\n1. \"推荐的都在涨\" 到底有多少信息量\n{'=' * 76}")
    fwd5 = (px.shift(-5) / px - 1).where(trad)
    up = (fwd5 > 0).sum(axis=1) / trad.sum(axis=1)
    up = up[sub].dropna()
    print(f"样本: {len(up)} 个交易日, 每日统计\"未来5日上涨的股票占比\"")
    print(f"{'分位':<10}{'上涨占比':>10}{'随机选5只至少4只涨的概率':>26}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        p = up.quantile(q)
        p4 = 5 * p ** 4 * (1 - p) + p ** 5
        print(f"{f'{int(q * 100)}%分位':<10}{p * 100:>9.1f}%{p4 * 100:>25.1f}%")
    p = up.mean()
    print(f"\n均值 {p * 100:.1f}% -> 随机选5只至少4只涨的概率 "
          f"{(5 * p ** 4 * (1 - p) + p ** 5) * 100:.1f}%, "
          f"5只全涨 {p ** 5 * 100:.1f}%")

    b = bench[sub]
    bt = (1 + b.fillna(0)).prod() - 1
    n = len(b)

    # ── 2. 随机选股对照组 (先验证机器, 再看信号) ──
    print(f"\n{'=' * 76}\n2. 随机选股对照组 —— 校验回测机器\n{'=' * 76}")
    print(f"随机打分走完全相同的 run_signal。判读标准:")
    print(f"  随机组【毛收益】≈ 池内等权基准  -> 机器可信, 信号的亏损是真结论")
    print(f"  随机组【毛收益】远低于基准      -> 机器或口径有问题, 信号结论作废")
    seeds = [int(x) for x in a.rand_seeds.split(",") if x.strip()]
    rnet, rgro = [], []
    for i, s in enumerate(seeds, 1):
        net, gro = run_signal(random_score(px, s), px, trad, a.hold, a.topn, cost)
        rnet.append(net[sub].fillna(0))
        rgro.append(gro[sub].fillna(0))
        print(f"  种子 {s:<4} ({i}/{len(seeds)}) 净收益 "
              f"{stats(rnet[-1], n)['tot'] * 100:>8.1f}%  毛收益 "
              f"{stats(rgro[-1], n)['tot'] * 100:>8.1f}%", flush=True)
    rnet_m = pd.concat(rnet, axis=1).median(axis=1)
    rgro_m = pd.concat(rgro, axis=1).median(axis=1)
    sn, sg = stats(rnet_m, n), stats(rgro_m, n)
    # 保留每个种子各自的总收益: 判断一个信号是否真的超出随机, 要看它有没有跳出
    # 【整个随机包络】, 而不是跟随机中位数比 —— 种子间跨度本身就很大。
    net_seed = [stats(x, n)["tot"] * 100 for x in rnet]
    gro_seed = [stats(x, n)["tot"] * 100 for x in rgro]
    nets = sorted(net_seed)
    print(f"\n{len(seeds)} 个随机种子净收益: 最差 {nets[0]:.1f}%  "
          f"中位 {np.median(nets):.1f}%  最好 {nets[-1]:.1f}%  "
          f"(跨度 {nets[-1] - nets[0]:.1f}pp —— 任何小于此的差距都无意义)")
    print(f"{'':<26}{'总收益%':>10}{'年化%':>9}{'夏普':>7}{'最大回撤%':>11}")
    print(f"{'随机选股(含成本)':<26}{sn['tot'] * 100:>10.1f}{sn['ann'] * 100:>9.1f}"
          f"{sn['sh']:>7.2f}{sn['dd'] * 100:>11.1f}")
    print(f"{'随机选股(不含成本)':<26}{sg['tot'] * 100:>10.1f}{sg['ann'] * 100:>9.1f}"
          f"{sg['sh']:>7.2f}{sg['dd'] * 100:>11.1f}")
    print(f"{'池内等权基准':<26}{bt * 100:>10.1f}")
    print(f"\n差距归因 (随机组 vs 基准, 共 {(sn['tot'] - bt) * 100:.1f}pp):")
    print(f"  交易成本      {(sn['tot'] - sg['tot']) * 100:>8.1f}pp")
    print(f"  集中度/波动拖累 {(sg['tot'] - bt) * 100:>8.1f}pp"
          f"   (只持{a.topn}只 vs 全池等权, 几何收益的固有损失)")

    # ── 3. 纯动量/反转基线 ──
    print(f"\n{'=' * 76}\n3. 纯信号基线 (每相位 top{a.topn}, 持有{a.hold}日, "
          f"{a.hold}相位取均值 -> 实际同时持仓最多 {a.hold * a.topn} 只, "
          f"单边成本{cost * 100:.2f}%)\n{'=' * 76}")
    print(f"不要看相对池内等权基准的差距 —— 那里面混着交易成本与集中度拖累,")
    print(f"随机选股同样会亏掉。只有相对随机组的差额才是信号本身的贡献。")
    ns = len(seeds)
    print(f"{'信号':<24}{'净收益%':>9}{'毛收益%':>9}{'vs随机毛pp':>11}"
          f"{'超随机种子(净)':>15}{'超随机种子(毛)':>15}{'t':>7}")
    for key, (desc, fn) in SIGNALS.items():
        sc = fn({"c": px})
        r, g = run_signal(sc, px, trad, a.hold, a.topn, cost)
        r, g = r[sub].fillna(0), g[sub].fillna(0)
        st, stg = stats(r, n), stats(g, n)
        vr = r - rnet_m
        t = vr.mean() / vr.std() * np.sqrt(len(vr)) if vr.std() > 0 else 0
        wn = sum(st["tot"] * 100 > x for x in net_seed)
        wg = sum(stg["tot"] * 100 > x for x in gro_seed)
        print(f"{desc:<24}{st['tot'] * 100:>9.1f}{stg['tot'] * 100:>9.1f}"
              f"{(stg['tot'] - sg['tot']) * 100:>11.1f}"
              f"{f'{wn}/{ns}':>15}{f'{wg}/{ns}':>15}{t:>7.2f}")
    print(f"{'随机选股(中位)':<24}{sn['tot'] * 100:>9.1f}{sg['tot'] * 100:>9.1f}")
    print(f"{'池内等权基准':<24}{bt * 100:>9.1f}")
    print(f"\n判读顺序:")
    print(f"  1. 【超随机种子(毛)】= {ns}/{ns} 才说明信号跳出了随机包络; "
          f"{ns // 2}/{ns} 附近等于没有能力。")
    print(f"  2. 毛收益剥掉了换手率差异带来的成本节省 —— 只看净收益会把"
          f"'换手低'误当成'选股准'。")
    print(f"  3. t 值仍需 >2 才谈得上显著; 累计 pp 再大, t 小就是噪声。")


if __name__ == "__main__":
    main()
