#!/usr/bin/env python3
"""诊断 select_features 的未来数据泄漏

select_features 用 s = df[df.date < TEST_START] 做特征筛选,
若 s 太小(<10000行)则退化为 s = df —— 即用【全样本含测试期标签】筛特征。
PIT 训练集恰好从 TEST_START(2022-09-01) 开始, 所以一直走的是退化分支。
"""
import pandas as pd

TEST_START = pd.Timestamp('2022-09-01')
MIN_TRAIN_DAYS, LABEL_HORIZON = 250, 5

df = pd.read_parquet('data/processed/training_data_pit_v24.parquet',
                     columns=['date', 'code', 'fwd_5d_ret'])
df['date'] = pd.to_datetime(df['date'])

lab = df['fwd_5d_ret'].notna()
last = df.loc[lab, 'date'].max()
df = df[lab | (df['date'] > last)]

pre = df[df['date'] < TEST_START]
print(f'TEST_START            : {TEST_START.date()}')
print(f'数据起始              : {df["date"].min().date()}')
print(f'TEST_START 之前行数    : {len(pre)}  -> 触发退化分支: {len(pre) < 10000}')
print('=> 特征筛选实际使用了全样本(含整个测试期的 fwd_5d_ret 标签) = 未来数据泄漏')
print()

all_dates = sorted(df['date'].unique())
first_pos = MIN_TRAIN_DAYS + LABEL_HORIZON
first_pred = all_dates[first_pos]
print(f'MIN_TRAIN_DAYS={MIN_TRAIN_DAYS}, LABEL_HORIZON={LABEL_HORIZON}')
print(f'首个可出信号日        : {first_pred.date()} (all_dates[{first_pos}])')

clean = df[df['date'] < first_pred]
print(f'该日之前可用行数      : {len(clean)}  -> 够用(>10000): {len(clean) >= 10000}')
print()
print('=> 修复方案: 特征筛选的截止日改用【首个可出信号日】而非 TEST_START,')
print('   这样有约 250 个交易日的纯样本内数据可用, 无需触发退化分支。')
