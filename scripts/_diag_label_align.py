"""决定性诊断: 训练目标(行业中性化超额)是否与真实赚钱目标(raw收益)对齐?

方法: 完美先知测试 (perfect foresight)
  如果用"未来真实的 fwd_1d_excess"选 top3, 其 raw 收益是多少?
  若连完美先知都跑输市场 -> 标签定义本身就是错的, 不是模型的问题
"""
import pandas as pd, numpy as np, json
from pathlib import Path

DF = pd.read_parquet('data/processed/training_data_v24.parquet')
DF['date'] = pd.to_datetime(DF['date'])
DF = DF.dropna(subset=['fwd_1d_ret'])
DF['code'] = DF['code'].astype(str)

# 概念分组
cp = json.loads(Path('data/universe/concept_stock_map.json').read_text())
s2c = cp.get('stock_to_concepts', {})
c2g = {str(k)[:6]: v[0] for k, v in s2c.items() if v}
DF['group'] = DF['code'].map(lambda c: c2g.get(c[:6], 'unknown'))

m = (DF['date'] >= '2022-09-01') & (DF['date'] <= '2026-07-16')
sub = DF[m].copy()

# 三种标签
sub['excess_date'] = sub.groupby('date')['fwd_1d_ret'].transform(lambda x: x - x.mean())
sub['excess_group'] = sub.groupby(['date', 'group'])['fwd_1d_ret'].transform(lambda x: x - x.mean())

mkt = sub.groupby('date')['fwd_1d_ret'].mean()

print("=== 完美先知: 按不同标签选 top3, 看真实 raw 收益 ===\n")
for lab in ['fwd_1d_ret', 'excess_date', 'excess_group']:
    daily = []
    for dt, g in sub.groupby('date'):
        if len(g) < 10:
            continue
        top = g.nlargest(3, lab)
        daily.append({'date': dt, 'raw': top['fwd_1d_ret'].mean(), 'mkt': mkt[dt]})
    dd = pd.DataFrame(daily)
    exc = (dd['raw'] - dd['mkt']).mean()
    print(f"按 {lab:14s} 选top3:")
    print(f"    raw日均 {dd['raw'].mean()*100:+.4f}%   超额 {exc*100:+.4f}%/day"
          f"   年化超额 {exc*252*100:+.1f}%")
    print(f"    累计 {((1+dd['raw']).prod()-1)*100:+.1f}%  (同期市场 {((1+dd['mkt']).prod()-1)*100:+.1f}%)\n")

# ── 分组结构诊断 ──
print("=== 概念分组结构 ===")
gs = sub.groupby(['date', 'group']).size()
print(f"  平均每(日,组)股票数: {gs.mean():.2f}")
print(f"  只有1只股票的组占比: {(gs == 1).mean()*100:.1f}%")
print(f"  组数量: {sub['group'].nunique()}")
print("  -> 组内只有1只股票时, demean 后 excess 恒为 0, 标签被摧毁")

# ── excess_group 与 raw 的相关性 ──
print("\n=== 标签相关性 (日内截面 Spearman 均值) ===")
from scipy.stats import spearmanr
cors = {'excess_date': [], 'excess_group': []}
for dt, g in sub.groupby('date'):
    if len(g) < 10:
        continue
    for k in cors:
        c, _ = spearmanr(g[k], g['fwd_1d_ret'])
        if c == c:
            cors[k].append(c)
for k, v in cors.items():
    print(f"  corr({k:13s}, raw) = {np.mean(v):.4f}")
