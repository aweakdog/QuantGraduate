"""汇总 t0close(T日收盘成交) vs base(T+1尾盘成交) 的多种子对照。

要回答的问题: 把"信号->成交"的一天延迟消掉, 能拿回多少收益。
t0close 是上界: 它假设 14:50 时全天日频特征(含资金流/成交量)都已收口。

    python scripts/sum_t0close.py
"""
import json
import re
import statistics as st
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# 标签和执行的对齐关系, 是这次对照的关键, 打印出来提醒自己
NOTE = """
标签 fwd_5d_ret = close_{t+5}/close_t - 1  ->  模型学的是"T日收盘买入"
  base3/base5   (t1close): 实际 T+1 收盘买  -> 标签与执行错位 1 天
  t0close3/5    (close)  : 实际 T   收盘买  -> 标签与执行对齐
所以差值 = 延迟成本 + 修正错位的收益, 两者混在一起。
"""


def collect(tag, win):
    """收集某配置某窗口下所有种子的结果。"""
    pat = re.compile(rf"^wf_daily_EVMBDMW_{re.escape(tag)}_{win}_s(\d+)_ts.*\.json$")
    out = {}
    for f in PROC.glob(f"wf_daily_EVMBDMW_{tag}_{win}_s*.json"):
        m = pat.match(f.name)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        s = d.get("summary", d)
        out[int(m.group(1))] = {
            "ret": s.get("total_return_pct"), "sharpe": s.get("sharpe"),
            "mdd": s.get("max_dd_pct"), "excess": s.get("excess_annual_pct"),
            "ic": s.get("ic_mean"), "cost": s.get("total_cost_pct"),
        }
    return out


def agg(rows, key):
    v = [r[key] for r in rows.values() if isinstance(r.get(key), (int, float))]
    if not v:
        return None
    return st.median(v), min(v), max(v), sum(1 for x in v if x < 0), len(v)


def main():
    print(NOTE)
    hdr = f"{'配置':<12}{'窗口':<4}{'种子':>4}{'收益中位':>10}{'最差':>9}{'最好':>9}{'夏普中位':>9}{'回撤中位':>9}{'亏损种子':>9}"
    for win in ("A", "B"):
        print("=" * len(hdr))
        print(f"窗口 {win}  " + ("(2020-07~2022-08 弱势期)" if win == "A"
                                else "(2022-09~2026-07 训练同期)"))
        print(hdr)
        base = {}
        for tag in ("base3", "t0close3", "base5", "t0close5"):
            rows = collect(tag, win)
            if not rows:
                print(f"{tag:<12}{win:<4}{'—— 无结果 ——':>20}")
                continue
            r = agg(rows, "ret")
            sh = agg(rows, "sharpe")
            md = agg(rows, "mdd")
            if not r:
                continue
            med, lo, hi, neg, n = r
            ic = agg(rows, "ic")
            cs = agg(rows, "cost")
            print(f"{tag:<12}{win:<4}{n:>4}{med:>9.1f}%{lo:>8.1f}%"
                  f"{hi:>8.1f}%{(sh[0] if sh else 0):>9.2f}"
                  f"{(md[0] if md else 0):>8.1f}%{neg:>6}/{n}"
                  f"   IC{(ic[0] if ic else 0):+.4f} 费{(cs[0] if cs else 0):.1f}%")
            base[tag] = med
        for n3 in ("3", "5"):
            b, t = base.get(f"base{n3}"), base.get(f"t0close{n3}")
            if b is not None and t is not None:
                print(f"  -> {n3}只: 消除一天延迟 {b:.1f}% -> {t:.1f}% "
                      f"(差 {t-b:+.1f}pp)")


if __name__ == "__main__":
    main()
