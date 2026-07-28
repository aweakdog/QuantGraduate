#!/usr/bin/env python3
"""实盘可操作性审计: 检查回测与真实下单之间的差距"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

print('=' * 70)
print('  1. 股票池 (universe_pit.parquet) 时效性')
print('=' * 70)
u = pd.read_parquet(ROOT / 'data' / 'universe' / 'universe_pit.parquet')
u['effective_date'] = pd.to_datetime(u['effective_date'])
eff = sorted(u['effective_date'].unique())
print(f'  生效期 {len(eff)} 个:')
for d in eff:
    n = (u['effective_date'] == d).sum()
    print(f'    {pd.Timestamp(d).date()}  {n} 只')
last = pd.Timestamp(eff[-1])
print(f'\n  最新生效期: {last.date()}  (距今 {(pd.Timestamp("2026-07-28") - last).days} 天)')

print('\n' + '=' * 70)
print('  2. ST / 退市风险股 (涨跌幅限制 5%, 代码按 10% 处理)')
print('=' * 70)
names = {}
pm = ROOT / 'data' / 'universe' / 'pit_metadata.parquet'
if pm.exists():
    m = pd.read_parquet(pm)
    names = dict(zip(m['code'].astype(str).str.zfill(6), m['name']))
p = ROOT / 'data' / 'raw' / 'all_stock_list.parquet'
if p.exists():
    n = pd.read_parquet(p)
    names.update(dict(zip(n['code'].astype(str).str[:6], n['name'])))

cur = u[u['effective_date'] == last]['code'].astype(str).str.zfill(6).tolist()
st = [(c, names.get(c, '?')) for c in cur if 'ST' in str(names.get(c, '')).upper()]
print(f'  当期池 {len(cur)} 只, 其中 ST/*ST: {len(st)} 只')
for c, nm in st[:10]:
    print(f'    {c}  {nm}')
if not st:
    print('    (无)')

print('\n  策略实际交易过的 ST 股:')
f = list((ROOT / 'data' / 'processed').glob('wf_daily_em_t1close_s001_*_cap20000.json'))
traded_st = []
if f:
    d = json.load(open(f[0]))
    traded = sorted({str(t['code'])[:6] for t in d['trades']})
    traded_st = [(c, names.get(c, '?')) for c in traded
                 if 'ST' in str(names.get(c, '')).upper()]
    for c, nm in traded_st:
        print(f'    {c}  {nm}')
    if not traded_st:
        print('    (无)')

print('\n' + '=' * 70)
print('  3. 复权方式 (影响"买不起一手"判断)')
print('=' * 70)
print('  update_kline_akshare.py 使用 adjust="qfq" = 前复权')
print('  前复权价格会随未来除权除息【追溯改变】。')
print('  -> 回测里用复权价判断"100股买不买得起", 与当时真实股价不一致。')

kl = pd.read_parquet(ROOT / 'data' / 'raw' / 'kline' / '300750.parquet')
kl['date'] = pd.to_datetime(kl['date'])
print(f'\n  例: 300750 复权后 2023-09-20 收盘 = '
      f'{float(kl[kl["date"]=="2023-09-20"]["close"].iloc[0]):.2f}')
print('       (真实当日股价与此不同, 若期间有分红送转)')

print('\n' + '=' * 70)
print('  4. 实盘出单脚本')
print('=' * 70)
for s, desc in [('daily_pipeline.py', '旧版每日预测'),
                ('daily_orchestrator.py', '每日编排')]:
    fp = ROOT / 'scripts' / s
    if not fp.exists():
        print(f'  {s}: 不存在')
        continue
    txt = fp.read_text(errors='ignore')
    win = 'D:\\\\myAI' in txt or 'D:\\myAI' in txt
    v22 = 'training_data_v22' in txt
    print(f'  {s} ({desc}):')
    print(f'    Windows硬编码路径: {"是 <- Mac跑不了" if win else "否"}')
    print(f'    用旧训练集v22: {"是 <- 非当前v35策略" if v22 else "否"}')
