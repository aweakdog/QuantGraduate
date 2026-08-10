"""量 top-N 选票在信号日之后的【逐日收益路径】。

要判定的事: base(T+1尾盘买) 比 t0close(T尾盘买) 好 25pp, 是不是因为
"信号日后第一天" 这段收益是负的 —— 也就是模型选出来的票在 T->T+1 先跌一下。

若 d1 显著为负而 d2..d6 为正, 则"等一天躲开追高"的说法成立;
若 d1 为正, 那我之前那套解释就是错的, 25pp 的差异得从别处找(如卖出端/成本)。

同时对比两种持有窗口的累计收益:
    label 窗口   close_t   -> close_{t+5}   (模型真正在优化的)
    base 实际    close_t+1 -> close_{t+6}   (线上真正在持有的)

    python scripts/diag_entry_path.py --seeds 42,7,123 --topn 3
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

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="42,7,123,1,2")
ap.add_argument("--topn", type=int, default=3)
ap.add_argument("--win", default="B", choices=["A", "B"])
ap.add_argument("--horizon", type=int, default=7, help="往后看几天")
ap.add_argument("--limit-thresh", type=float, default=0.098,
                help="信号日涨幅超过此值即视为涨停/接近涨停(买不到)")
ap.add_argument("--trade-cost", type=float, default=0.0003, help="佣金/边")
ap.add_argument("--slippage", type=float, default=0.002, help="滑点/边")
args = ap.parse_args()


def load_close():
    """加载全市场收盘价矩阵 date x code6。"""
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
        raise SystemExit(f"ERROR: {KLINE} 下没有可读的 K线")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df.pivot_table(index="date", columns="code", values="close")


def main():
    print(f"加载K线 ...", flush=True)
    px = load_close()
    # 键必须保持 pd.Timestamp: to_numpy() 会变成 datetime64, 与 preds 里的
    # Timestamp 哈希不等, 会导致所有信号日静默落空(样本数 0)
    dates = list(px.index)
    dpos = {d: i for i, d in enumerate(dates)}
    print(f"  {px.shape[1]} 只 x {px.shape[0]} 天")

    H = args.horizon
    # legs[k] 收集所有 (信号日, 票) 在第 k 天的单日收益 close_{t+k}/close_{t+k-1}-1
    # 再按"信号日当天是否已涨停"分组: 涨停的票 T 日收盘买不进去(t0close 拿不到),
    # 但 T+1 收盘板打开后 base 买得到。这是解释两者差异的关键分组。
    legs = {k: [] for k in range(1, H + 1)}
    legs_ok = {k: [] for k in range(1, H + 1)}    # 信号日未涨停 -> t0close 买得到
    legs_lmt = {k: [] for k in range(1, H + 1)}   # 信号日已涨停 -> t0close 买不到
    n_sig = n_pick = n_lmt = 0

    for sd in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        f = PROC / f"preds_eval_mbdmw_{args.win}_s{sd}.pkl"
        if not f.exists():
            print(f"  跳过 s{sd}: 无缓存")
            continue
        preds = pickle.load(open(f, "rb"))["preds"]
        for dp in preds:
            d = pd.Timestamp(dp["date"])
            if d not in dpos:
                continue
            gp = dpos[d]
            if gp + H >= len(dates):
                continue
            codes = [c[0] if isinstance(c, (list, tuple)) else c
                     for c in dp["ranked"][:args.topn]]
            n_sig += 1
            for code in codes:
                # preds 里是 '600221.SH', 不能用 [-6:] (会得到 '221.SH')
                c6 = str(code).split(".")[0][-6:]
                if c6 not in px.columns:
                    continue
                s = px[c6].to_numpy()  # 该票的收盘价序列, 与 dates 同序
                # 信号日当天涨幅: close_t / close_{t-1} - 1
                is_lmt = False
                if gp >= 1:
                    p0, p1 = s[gp - 1], s[gp]
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                        is_lmt = (p1 / p0 - 1) >= args.limit_thresh
                n_pick += 1
                n_lmt += int(is_lmt)
                for k in range(1, H + 1):
                    a, b = s[gp + k - 1], s[gp + k]
                    if np.isfinite(a) and np.isfinite(b) and a > 0:
                        r = b / a - 1
                        legs[k].append(r)
                        (legs_lmt if is_lmt else legs_ok)[k].append(r)

    print(f"\n窗口 {args.win} | top{args.topn} | 种子 {args.seeds} | 信号日样本 {n_sig}")
    print("\n信号日后逐日单日收益 (毛, 未扣成本):")
    print(f"{'第k天':<8}{'区间':<22}{'均值':>9}{'中位':>9}{'胜率':>8}{'样本':>8}")
    cum_from_t, cum_from_t1 = 1.0, 1.0
    for k in range(1, H + 1):
        v = np.array(legs[k])
        if v.size == 0:
            continue
        rng = f"close_t+{k-1} -> close_t+{k}"
        print(f"d{k:<7}{rng:<22}{v.mean()*100:>8.3f}%{np.median(v)*100:>8.3f}%"
              f"{(v > 0).mean()*100:>7.1f}%{v.size:>8}")
        if k <= 5:
            cum_from_t *= 1 + v.mean()
        if 2 <= k <= 6:
            cum_from_t1 *= 1 + v.mean()

    print(f"\n累计 (按各腿均值复利):")
    print(f"  label窗口  close_t   -> close_t+5 : {(cum_from_t-1)*100:+.3f}%"
          f"   <- 模型在优化这个")
    print(f"  base实际   close_t+1 -> close_t+6 : {(cum_from_t1-1)*100:+.3f}%"
          f"   <- 线上在持有这个")
    d1 = np.array(legs[1])
    if d1.size:
        print(f"\n  d1 (信号日->次日) 均值 {d1.mean()*100:+.3f}% ,"
              f" 这一天 base 是躲开的, t0close 是吃进去的")

    # ── 关键分组: 信号日已涨停的票, T日收盘买不进去 ──
    print(f"\n{'='*66}")
    print(f"按信号日是否已涨停(涨幅>={args.limit_thresh*100:.1f}%)分组")
    print(f"  选中样本 {n_pick} 只次, 其中信号日涨停 {n_lmt} 只次 "
          f"({n_lmt/max(1,n_pick)*100:.1f}%)")
    print(f"\n{'第k天':<7}{'未涨停(t0可买)':>22}{'已涨停(t0买不到)':>24}")
    print(f"{'':7}{'均值':>10}{'样本':>11}{'均值':>11}{'样本':>12}")
    cum_ok = cum_all = 1.0
    for k in range(1, H + 1):
        a, b = np.array(legs_ok[k]), np.array(legs_lmt[k])
        if a.size == 0 and b.size == 0:
            continue
        print(f"d{k:<6}{(a.mean()*100 if a.size else 0):>9.3f}%{a.size:>11}"
              f"{(b.mean()*100 if b.size else 0):>10.3f}%{b.size:>12}")
        if k <= 5:
            cum_ok *= 1 + (a.mean() if a.size else 0)
            cum_all *= 1 + np.array(legs[k]).mean()
    print(f"\n  close_t->close_t+5 累计:")
    print(f"    全部选票        {(cum_all-1)*100:+.3f}%")
    print(f"    剔除涨停后      {(cum_ok-1)*100:+.3f}%   <- t0close 实际能买到的")
    print(f"  差值 {((cum_ok-cum_all))*100:+.3f}pp —— 若为负, 说明 T日收盘买的"
          f"超额收益被涨停板锁在门外, 这就是 t0close 打不赢 base 的原因")

    # ── 最优持有天数: 买入固定在 close_t+1 (可实盘), 卖出扫 t+2..t+H ──
    # 每个周期只付一次往返成本, 所以拉长持有 = 摊薄成本, 但要看信号衰减得多快。
    # 换仓周期 = 持有天数 N, 一年约 242 个交易日 -> 242/N 个周期。
    rt_cost = (args.trade_cost + args.slippage) * 2
    print(f"\n{'='*66}")
    print(f"最优持有天数 (买入固定 close_t+1, 往返成本 {rt_cost*100:.2f}%)")
    print(f"{'持有N天':<9}{'区间':<20}{'毛/周期':>10}{'净/周期':>10}"
          f"{'年化净':>10}{'年换仓':>8}")
    best = None
    for N in range(1, H):
        # 持有 N 天 = 吃 d2..d(N+1) 这 N 条腿
        gross = 1.0
        okleg = True
        for k in range(2, N + 2):
            v = legs.get(k)
            if not v:
                okleg = False
                break
            gross *= 1 + float(np.mean(v))
        if not okleg:
            break
        net = gross - 1 - rt_cost
        cycles = 242.0 / N
        ann = (1 + net) ** cycles - 1 if net > -1 else -1.0
        rng = f"t+1 -> t+{N+1}"
        print(f"{N:<9}{rng:<20}{(gross-1)*100:>9.3f}%{net*100:>9.3f}%"
              f"{ann*100:>9.1f}%{cycles:>8.0f}")
        if best is None or ann > best[1]:
            best = (N, ann)
    if best:
        print(f"\n  最优持有 {best[0]} 天, 年化净 {best[1]*100:+.1f}%  "
              f"(现状是 5 天)")
        print(f"  注: 这是 top{args.topn} 等权、忽略涨停不可买/停牌/整手约束的"
              f"上界估计, 用于定方向而非预测收益")


if __name__ == "__main__":
    main()
