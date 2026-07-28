"""诊断 v33: 信号有效(IC t=6.17)但组合亏损, 缺口在哪?"""
import pandas as pd, numpy as np, json
from pathlib import Path

p = Path('data/processed/wf_daily_v33_select_timing_dart_ts2022-01-01_te2026-07-16_cap100000.json')
res = json.loads(p.read_text())
d = pd.DataFrame(res['daily'])
d['date'] = pd.to_datetime(d['date'])

DF = pd.read_parquet('data/processed/training_data_v24.parquet')
DF['date'] = pd.to_datetime(DF['date'])
DF = DF.dropna(subset=['fwd_1d_ret'])
DF['code'] = DF['code'].astype(str)

# 每日全市场等权 raw 收益
mkt = DF.groupby('date')['fwd_1d_ret'].mean()

# ── 1. 资金部署率 ──
d['deployed'] = 1 - d['cash'] / d['portfolio_value']
print("=== 1. 资金部署率 ===")
print(f"  平均部署率      : {d['deployed'].mean()*100:.1f}%")
print(f"  中位数          : {d['deployed'].median()*100:.1f}%")
print(f"  部署率>50%的天数: {(d['deployed']>0.5).sum()} / {len(d)}")
print(f"  部署率<10%的天数: {(d['deployed']<0.1).sum()} / {len(d)}")

# ── 2. 每日收益分布 ──
print("\n=== 2. 策略日收益分布 ===")
r = d['daily_ret']
print(f"  mean={r.mean()*100:+.4f}%  std={r.std()*100:.4f}%")
print(f"  ==0 的天数      : {(r.abs()<1e-9).sum()} / {len(d)}")
print(f"  >0 : {(r>1e-9).sum()},  <0 : {(r<-1e-9).sum()}")

# ── 3. 选出的 top3 的【真实】次日收益 vs 全市场 ──
print("\n=== 3. 选股本身有没有 alpha (不含执行/成本) ===")
lut = DF.set_index(['date', 'code'])['fwd_1d_ret']
rows = []
for _, row in d.iterrows():
    dt = row['date']
    picks = row['holdings'] if isinstance(row['holdings'], list) else []
    if not picks:
        continue
    vals = [lut.get((dt, str(c)), np.nan) for c in picks]
    vals = [v for v in vals if v == v]
    if not vals:
        continue
    rows.append({'date': dt, 'pick_ret': np.mean(vals), 'n': len(vals),
                 'mkt_ret': mkt.get(dt, np.nan)})
pk = pd.DataFrame(rows).dropna()
print(f"  有持仓的天数    : {len(pk)}")
print(f"  持仓票次日均收益: {pk['pick_ret'].mean()*100:+.4f}%/day")
print(f"  同日全市场均收益: {pk['mkt_ret'].mean()*100:+.4f}%/day")
print(f"  选股超额        : {(pk['pick_ret']-pk['mkt_ret']).mean()*100:+.4f}%/day"
      f"  -> 年化 {((pk['pick_ret']-pk['mkt_ret']).mean())*252*100:+.1f}%")
print(f"  纯选股累计(无成本): {((1+pk['pick_ret']).prod()-1)*100:+.1f}%")
print(f"  同期全市场累计    : {((1+pk['mkt_ret']).prod()-1)*100:+.1f}%")

# ── 4. 成本拖累 ──
print("\n=== 4. 成本 ===")
tot_cost = d['sell_cost'].sum() + d['buy_cost'].sum()
print(f"  总费用          : {tot_cost:,.0f} 元 = 初始资金的 {tot_cost/100000*100:.1f}%")
print(f"  日均费用/组合值 : {((d['sell_cost']+d['buy_cost'])/d['portfolio_value']).mean()*100:.4f}%/day")

# ── 5. 拒单影响 ──
print("\n=== 5. 持仓数分布 ===")
print(d['n_holdings'].value_counts().sort_index().to_string())
