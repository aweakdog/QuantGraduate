"""汇总 regime 空仓过滤器扫参结果对比"""
import glob
import json

FILES = sorted(glob.glob("data/processed/wf_daily_regime_*_cap100000.json"))
ORDER = {"off": 0, "ma": 1, "breadth": 2, "both": 3, "any": 4}

rows = []
for f in FILES:
    d = json.load(open(f, encoding="utf-8"))
    rows.append((d["regime_filter"], d["summary"], d["stability"]))
rows.sort(key=lambda x: ORDER.get(x[0], 9))

hdr = (f"{'配置':<8}{'空仓%':>7}{'总收益%':>9}{'年化%':>8}{'夏普':>7}{'回撤%':>8}"
       f"{'超额年化%':>11}{'IR':>7}{'beta':>7}{'alpha%':>8}{'费用%':>7}{'交易':>6}")
print(hdr)
print("-" * 95)
for n, s, _ in rows:
    print(f"{n:<10}{s['cash_days_pct']:>6.1f}{s['total_return_pct']:>9.1f}"
          f"{s['annualized_return_pct']:>8.1f}{s['sharpe']:>7.2f}{s['max_dd_pct']:>8.1f}"
          f"{s['excess_annual_pct']:>10.1f}{s['information_ratio']:>8.2f}{s['beta']:>7.3f}"
          f"{s['alpha_annual_pct']:>8.1f}{s['total_cost_pct']:>7.1f}{s['n_trades']:>6}")

b = rows[0][1]
print(f"\n基准(等权买入持有): 总收益 {b['benchmark_total_pct']:+.1f}% "
      f"年化 {b['benchmark_annual_pct']:+.1f}%")

print("\n分段稳健性:")
for n, _, h in rows:
    print(f"  {n:<9} 前半 {h[0]['strategy_pct']:+7.1f}% vs 基准 {h[0]['benchmark_pct']:+6.1f}% | "
          f"后半 {h[1]['strategy_pct']:+7.1f}% vs 基准 {h[1]['benchmark_pct']:+6.1f}%")
