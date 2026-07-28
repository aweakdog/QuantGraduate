"""诊断: v33 亏损是来自 alpha 失效, 还是来自未对冲的市场 beta?"""
import pandas as pd, numpy as np, json
from pathlib import Path

DF = pd.read_parquet('data/processed/training_data_v24.parquet')
DF['date'] = pd.to_datetime(DF['date'])
DF = DF.dropna(subset=['fwd_1d_ret'])

START, END = pd.Timestamp('2022-09-01'), pd.Timestamp('2026-07-16')
m = (DF['date'] >= START) & (DF['date'] <= END)
sub = DF[m]

# 1. 等权全市场(216只)日收益
mkt = sub.groupby('date')['fwd_1d_ret'].mean()
print(f"Universe (216 stocks) equal-weight, {mkt.index.min().date()} ~ {mkt.index.max().date()}")
print(f"  days               : {len(mkt)}")
print(f"  cum return         : {((1+mkt).prod()-1)*100:+.1f}%")
print(f"  annualized         : {((1+mkt).prod()**(252/len(mkt))-1)*100:+.1f}%")
print(f"  daily mean         : {mkt.mean()*100:+.4f}%")
print(f"  daily std          : {mkt.std()*100:.4f}%")

# 2. 隔夜 vs 日内 分解
if 'fwd_1d_exec_ret' in sub.columns:
    intraday = sub.groupby('date')['fwd_1d_exec_ret'].mean()
    print(f"\nDecomposition (universe average):")
    print(f"  intraday (o->c)    : {((1+intraday).prod()-1)*100:+.1f}% cum, {intraday.mean()*100:+.4f}%/day")
    # overnight = (1+c2c)/(1+intraday) - 1 approximately, aligned by date
    both = pd.concat([mkt.rename('c2c'), intraday.rename('intra')], axis=1).dropna()
    ovn = (1 + both['c2c']) / (1 + both['intra']) - 1
    print(f"  overnight (c->o)   : {((1+ovn).prod()-1)*100:+.1f}% cum, {ovn.mean()*100:+.4f}%/day")

# 3. v33 回测的实际每日收益 vs 市场
p = Path('data/processed/wf_daily_v33_select_timing_dart_ts2022-01-01_te2026-07-16_cap100000.json')
if p.exists():
    res = json.loads(p.read_text())
    d = pd.DataFrame(res['daily'])
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    joined = pd.concat([d['daily_ret'].rename('strat'), mkt.rename('mkt')], axis=1).dropna()
    print(f"\nv33 strategy vs universe ({len(joined)} common days):")
    print(f"  strat cum          : {((1+joined['strat']).prod()-1)*100:+.1f}%")
    print(f"  mkt   cum          : {((1+joined['mkt']).prod()-1)*100:+.1f}%")
    print(f"  strat - mkt (daily): {(joined['strat']-joined['mkt']).mean()*100:+.4f}%/day")
    beta = np.cov(joined['strat'], joined['mkt'])[0,1] / np.var(joined['mkt'])
    alpha_d = joined['strat'].mean() - beta * joined['mkt'].mean()
    print(f"  beta to universe   : {beta:.3f}")
    print(f"  daily alpha        : {alpha_d*100:+.4f}%  -> annualized {alpha_d*252*100:+.1f}%")

    # 4. IC 显著性
    ics = d['ic'].dropna().astype(float)
    t = ics.mean() / ics.std() * np.sqrt(len(ics))
    print(f"\nIC diagnostics:")
    print(f"  mean={ics.mean():.4f} std={ics.std():.4f} n={len(ics)} t-stat={t:.2f}")
