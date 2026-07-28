#!/usr/bin/env python3
"""验证 label-NaN 尾部保留修复: 日期范围延长, 且历史段逐行不变"""
import pandas as pd

df = pd.read_parquet('data/processed/training_data_pit_v24.parquet',
                     columns=['date', 'code', 'fwd_5d_ret'])
df['date'] = pd.to_datetime(df['date'])

old = df.dropna(subset=['fwd_5d_ret'])
ad_old = sorted(old['date'].unique())

lab = df['fwd_5d_ret'].notna()
last = df.loc[lab, 'date'].max()
new = df[lab | (df['date'] > last)]
ad_new = sorted(new['date'].unique())

print('OLD all_dates last:', ad_old[-1].date(), ' n =', len(ad_old))
print('NEW all_dates last:', ad_new[-1].date(), ' n =', len(ad_new))
print('新增可出信号日期:', [str(d.date()) for d in ad_new[len(ad_old):]])
print()
print('OLD rows:', len(old), ' NEW rows:', len(new), ' delta:', len(new) - len(old))
print()

h_old = old[old['date'] <= last]
h_new = new[new['date'] <= last]
print('历史段行数一致(修复不影响历史回测):', len(h_old) == len(h_new),
      f'({len(h_old)} vs {len(h_new)})')
print()
print('t1open: 最后信号日', ad_new[-2].date(), '-> 成交日', ad_new[-1].date())
