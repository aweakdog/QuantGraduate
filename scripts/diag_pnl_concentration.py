# -*- coding: utf-8 -*-
"""检验回测收益对少数几笔交易的依赖度

总收益在持仓极少时统计上不可靠(t≈1), 本脚本给出三个更可信的判据:
  1. 日收益序列的 t 值
  2. FIFO 配对后每笔操作净盈亏的集中度 (top1/top5/top10 占比)
  3. 剔除最赚的 5/10 笔后是否还赚钱

用法:
  python scripts/diag_pnl_concentration.py <result.json> [<result2.json> ...]
"""
import json
import sys
from collections import defaultdict, deque

import numpy as np


def paired_pnl(trades):
    """FIFO 配对 买入->卖出, 返回每笔完整操作的净盈亏"""
    pend = defaultdict(deque)
    ops = []
    for tr in trades:
        code = tr["code"]
        if tr["action"] == "buy":
            pend[code].append(dict(tr))
            continue
        left = tr["shares"]
        while left > 0 and pend[code]:
            b = pend[code][0]
            sh = min(b["shares"], left)
            pnl = (sh * (tr["price"] - b["price"])
                   - b["fee"] * sh / b["shares"]
                   - tr["fee"] * sh / tr["shares"])
            ops.append(pnl)
            b["shares"] -= sh
            left -= sh
            if b["shares"] == 0:
                pend[code].popleft()
    return np.array(sorted(ops, reverse=True))


def analyze(path):
    d = json.load(open(path, encoding="utf-8"))
    s = d["summary"]
    r = np.array([x["daily_ret"] for x in d["daily"]])
    n = len(r)
    t_daily = r.mean() / r.std() * np.sqrt(n) if r.std() > 0 else float("nan")

    ops = paired_pnl(d["trades"])
    tot = ops.sum()

    print(f"{'=' * 66}")
    print(f"  {path.split('/')[-1]}")
    print(f"{'=' * 66}")
    print(f"  总收益 {s['total_return_pct']:+.1f}%  夏普 {s['sharpe']:.2f}  "
          f"IR {s['information_ratio']:.2f}  回撤 {s['max_dd_pct']:.1f}%")
    print(f"  IC {s['ic_mean']:+.4f} (t={s['ic_tstat']:.2f})   <- 唯一统计上可靠的指标")
    print(f"  交易日 {n} | 日收益 t = {t_daily:.2f}")
    print(f"  配对操作 {len(ops)} 笔 | 净盈亏合计 ¥{tot:,.0f} | 胜率 {(ops > 0).mean() * 100:.1f}%")
    print(f"  集中度: 最赚1笔 {ops[0] / tot * 100:.1f}% | "
          f"最赚5笔 {ops[:5].sum() / tot * 100:.1f}% | "
          f"最赚10笔 {ops[:10].sum() / tot * 100:.1f}%")
    ex5, ex10 = tot - ops[:5].sum(), tot - ops[:10].sum()
    cap = d.get("initial_capital", 0)
    print(f"  剔除最赚5笔:  ¥{ex5:,.0f}  ({ex5 / cap * 100:+.1f}% of 本金)")
    print(f"  剔除最赚10笔: ¥{ex10:,.0f}  ({ex10 / cap * 100:+.1f}% of 本金)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyze(p)
