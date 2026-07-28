"""汇总 PIT 无偏股票池上的回测结果, 并与旧的216人工池结果对照"""
import glob
import json

ORDER = {"off": 0, "ma": 1, "breadth": 2, "both": 3, "any": 4}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f, encoding="utf-8"))
        rows.append((d["regime_filter"], d["summary"], d.get("stability"), d))
    rows.sort(key=lambda x: ORDER.get(x[0], 9))
    return rows


def table(rows, title):
    print(f"\n{title}")
    hdr = (f"{'配置':<8}{'空仓%':>7}{'总收益%':>9}{'年化%':>8}{'夏普':>7}{'回撤%':>8}"
           f"{'超额年化%':>11}{'IR':>7}{'beta':>7}{'alpha%':>8}{'费用%':>7}{'交易':>6}"
           f"{'IC':>7}")
    print(hdr)
    print("-" * 100)
    for n, s, _, _ in rows:
        print(f"{n:<10}{s['cash_days_pct']:>6.1f}{s['total_return_pct']:>9.1f}"
              f"{s['annualized_return_pct']:>8.1f}{s['sharpe']:>7.2f}{s['max_dd_pct']:>8.1f}"
              f"{s['excess_annual_pct']:>10.1f}{s['information_ratio']:>8.2f}"
              f"{s['beta']:>7.3f}{s['alpha_annual_pct']:>8.1f}{s['total_cost_pct']:>7.1f}"
              f"{s['n_trades']:>6}{s['ic_mean']:>7.3f}")
    b = rows[0][1]
    print(f"基准(池内等权买入持有): 总收益 {b['benchmark_total_pct']:+.1f}% "
          f"年化 {b['benchmark_annual_pct']:+.1f}%")


pit = load("data/processed/wf_daily_pit_*_cap100000.json")
old = load("data/processed/wf_daily_regime_*_cap100000.json")

table(pit, "【PIT 无偏池 519只, 每日300只成分】")
table(old, "【旧: 216只人工池(含前视偏差)】")

print("\n分段稳健性 (PIT池):")
for n, _, h, _ in pit:
    f1 = "跑赢" if h[0]["strategy_pct"] > h[0]["benchmark_pct"] else "跑输"
    f2 = "跑赢" if h[1]["strategy_pct"] > h[1]["benchmark_pct"] else "跑输"
    print(f"  {n:<9} 前半 {h[0]['strategy_pct']:+7.1f}% vs 基准 {h[0]['benchmark_pct']:+6.1f}% ({f1}) | "
          f"后半 {h[1]['strategy_pct']:+7.1f}% vs {h[1]['benchmark_pct']:+6.1f}% ({f2})")

print("\n同配置对比 (PIT vs 人工池):")
om = {n: s for n, s, _, _ in old}
print(f"{'配置':<9}{'PIT总收益':>10}{'旧总收益':>10}{'PIT夏普':>9}{'旧夏普':>8}"
      f"{'PITalpha':>10}{'旧alpha':>9}")
for n, s, _, _ in pit:
    if n in om:
        o = om[n]
        print(f"{n:<11}{s['total_return_pct']:>9.1f}{o['total_return_pct']:>10.1f}"
              f"{s['sharpe']:>9.2f}{o['sharpe']:>8.2f}"
              f"{s['alpha_annual_pct']:>10.1f}{o['alpha_annual_pct']:>9.1f}")

d = pit[0][3]
print(f"\n入选特征数: {d['features']} | 训练集: {d['train_file']} | 成分约束: {d['pit_universe']}")
