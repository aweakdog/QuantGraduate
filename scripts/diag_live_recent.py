"""最近的"每日推荐"到底跑赢了没有 —— 和同期全市场做对照。

为什么不能只看"推荐的都在涨"
────────────────────────────
A 股涨跌高度同步。行情好的那几天, 池子里七八成股票都在涨, 这时候随便挑 5 只
"几乎都涨"是大概率事件, 和模型好不好没关系。所以唯一有意义的问题是:
    推荐票的收益, 减去同期【同一股票池等权】的收益, 是正是负?

同时报三个数, 缺一不可:
    推荐涨的比例  vs  全池涨的比例      (方向对不对)
    推荐平均收益  vs  全池平均收益      (幅度够不够)
    样本天数                            (够不够下结论 —— 通常不够)

成交口径与线上一致: 信号日 T 出榜, T+1 尾盘买入(exec_mode=t1close), 所以收益
从 T+1 收盘算起, 算到数据最后一天。持有不足一天的当天不计入。

免责: 一周的样本量做不了任何统计推断, 这个脚本的用途是【防止把普涨误读成
alpha】, 不是用来证明策略有效。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KLINE = ROOT / "data" / "raw" / "kline"
LIVE = ROOT / "data" / "live"
UNIV = ROOT / "data" / "universe"


def load_close(codes=None):
    frames = []
    files = KLINE.glob("*.parquet")
    for f in files:
        if codes is not None and f.stem[-6:] not in codes:
            continue
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


def detect_skip(plan):
    """从候选名单反推该档案能不能买创业板/科创板。

    plan 的 config 里没有 skip_boards 字段(线上是在别处过滤的), 但 ranked 是
    过滤后的候选池 —— 里面出现过 30/688 就说明该档案有权限。必须这样按档案区分,
    否则拿主板池去比一个买创业板的档案, 会把板块暴露算成选股能力。
    """
    has = any(str(c).zfill(6).startswith(("30", "688"))
              for c in plan.get("ranked", []))
    return () if has else ("30", "688")


def pool_codes(px, universe, skip_boards):
    """最近一期 PIT 成分 + 可买板块"""
    u = pd.read_parquet(UNIV / universe)
    u["effective_date"] = pd.to_datetime(u["effective_date"])
    last = u[u["effective_date"] == u["effective_date"].max()]
    s = set(last["code"].astype(str).str.zfill(6))
    return {c for c in px.columns
            if c in s and not c.startswith(tuple(skip_boards))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aggr5w")
    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--days", type=int, default=7, help="回看几个信号日")
    ap.add_argument("--universe", default="", help="留空则用 plan 里记录的池")
    a = ap.parse_args()

    plans = sorted(glob.glob(str(LIVE / f"plan_{a.profile}_[0-9]*.json")))[-a.days:]
    if not plans:
        raise SystemExit(f"没找到 plan_{a.profile}_*.json")

    last_plan = json.load(open(plans[-1]))
    skip = detect_skip(last_plan)
    univ = a.universe or last_plan["config"]["pit_universe"]

    px = load_close()
    pool = pool_codes(px, univ, skip)
    last_date = px.index.max()
    board = "仅主板" if skip else "全板块(含创业板/科创板)"
    print(f"档案={a.profile}  行情最新={last_date.date()}  池={univ}  "
          f"可买范围={board}  对照池={len(pool)} 只  取每日推荐前{a.topn}\n")

    rows = []
    for p in plans:
        d = json.load(open(p))
        sig = pd.Timestamp(d["signal_date"])
        recs = [x for x in d["recommend"] if not x.get("blocked")][:a.topn]
        # T+1 尾盘买入
        after = px.index[px.index > sig]
        if len(after) < 2:
            continue                      # 买入日就是最后一天, 没有持有期
        buy_d = after[0]
        r_pick = []
        for x in recs:
            c = str(x["code"]).zfill(6)
            if c not in px.columns:
                continue
            p0, p1 = px.at[buy_d, c], px.at[last_date, c]
            if pd.isna(p0) or pd.isna(p1):
                continue
            r_pick.append(p1 / p0 - 1)
        if not r_pick:
            continue
        cols = [c for c in pool if not pd.isna(px.at[buy_d, c])
                and not pd.isna(px.at[last_date, c])]
        r_pool = (px.loc[last_date, cols] / px.loc[buy_d, cols] - 1).values
        rows.append({
            "signal": sig.date(), "buy": buy_d.date(), "n": len(r_pick),
            "pick_up": np.mean(np.array(r_pick) > 0) * 100,
            "pool_up": np.mean(r_pool > 0) * 100,
            "pick_ret": np.mean(r_pick) * 100,
            "pool_ret": np.mean(r_pool) * 100,
        })

    if not rows:
        raise SystemExit("没有可评估的信号日 (推荐票都缺行情, 或持有期不足)")
    df = pd.DataFrame(rows)
    print(f"{'信号日':<12}{'买入日':<12}{'票数':>5}{'推荐涨%':>9}{'全池涨%':>9}"
          f"{'推荐收益%':>11}{'全池收益%':>11}{'超额%':>9}")
    for _, r in df.iterrows():
        print(f"{r['signal']!s:<12}{r['buy']!s:<12}{r['n']:>5.0f}"
              f"{r['pick_up']:>9.0f}{r['pool_up']:>9.0f}"
              f"{r['pick_ret']:>11.2f}{r['pool_ret']:>11.2f}"
              f"{r['pick_ret'] - r['pool_ret']:>9.2f}")
    ex = df["pick_ret"] - df["pool_ret"]
    print(f"\n合计 {len(df)} 个信号日")
    print(f"  推荐平均上涨比例 {df['pick_up'].mean():.0f}%  "
          f"vs  全池 {df['pool_up'].mean():.0f}%")
    print(f"  推荐平均收益 {df['pick_ret'].mean():.2f}%  "
          f"vs  全池 {df['pool_ret'].mean():.2f}%  "
          f"-> 超额 {ex.mean():.2f}%")
    print(f"  超额为正的信号日: {(ex > 0).sum()}/{len(ex)}")
    print("\n注: 样本只有几天, 不能做统计推断。这里唯一能排除的是"
          "\"把普涨当成选股能力\"。")


if __name__ == "__main__":
    main()
