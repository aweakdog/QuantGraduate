#!/usr/bin/env python3
"""实测「每日出单」一次要多少内存和时间 —— 服务器选型依据

回测是重训 943 次(约24分钟), 但实盘每天只需:
  载入训练集 -> 筛特征 -> 训一个模型 -> 预测当天
本脚本按 wf_v35 的真实配置跑这一条链路, 记录各阶段耗时与峰值内存。
"""
import resource
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd

TRAIN = 'data/processed/training_data_pit_v24.parquet'
UNI = 'data/universe/universe_pit.parquet'
LABEL_RAW = 'fwd_5d_ret'
LABEL = 'y_target'
N_FEATURES = 80
CORR_THRESH = 0.9
RANK_BUCKETS = 10
LOCKED = dict(n_estimators=151, max_depth=4, learning_rate=0.03,
              num_leaves=15, subsample=0.8, colsample_bytree=0.8,
              min_child_samples=50, random_state=42, n_jobs=10,
              verbosity=-1, boosting_type='dart')
SKIP = {'date', 'code'}


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1048576 if sys.platform == 'darwin' else r / 1024


_t0 = time.time()
_marks = []


def mark(label):
    _marks.append((label, time.time() - _t0, rss_mb()))
    print(f'  {label:32s} 累计 {time.time()-_t0:6.1f}s   峰值内存 {rss_mb():6.0f} MB',
          flush=True)


print('=' * 70)
print('  每日出单一次的资源开销实测')
print('=' * 70)

df = pd.read_parquet(TRAIN)
df['date'] = pd.to_datetime(df['date'])
mark('1. 载入训练集')

u = pd.read_parquet(UNI)
u['effective_date'] = pd.to_datetime(u['effective_date'])
u['code'] = u['code'].astype(str).str[:6]
df['code'] = df['code'].astype(str).str[:6]
eff = sorted(u['effective_date'].unique())
df['_p'] = pd.cut(df['date'], bins=list(eff) + [pd.Timestamp('2099-01-01')],
                  labels=eff, right=False)
allowed = u.groupby('effective_date')['code'].apply(set).to_dict()
keep = [c in allowed.get(p, set()) for c, p in zip(df['code'], df['_p'])]
df = df[keep].drop(columns='_p')
mark('2. PIT 股票池过滤')

df = df[df[LABEL_RAW].notna()].copy()
df[LABEL] = df.groupby('date')[LABEL_RAW].transform(lambda x: x - x.mean())

feats = [c for c in df.columns
         if c not in SKIP and c != LABEL and not c.startswith('fwd_')
         and '_21d' not in c and not c.endswith('_cross')
         and pd.api.types.is_numeric_dtype(df[c])]
sub = df.dropna(subset=[LABEL])
X = sub.groupby('code')[feats].transform(lambda c: c.ffill().fillna(0))
m = lgb.LGBMRegressor(**dict(LOCKED, n_estimators=50, boosting_type='gbdt'))
m.fit(X, sub[LABEL])
imp = pd.DataFrame({'f': feats, 'i': m.feature_importances_}).sort_values('i', ascending=False)
top = imp.head(N_FEATURES)['f'].tolist()
cm = X[top].corr().abs()
sel = []
for f in top:
    if all(cm.loc[f, s] < CORR_THRESH for s in sel):
        sel.append(f)
del X, sub
mark(f'3. 特征筛选 ({len(feats)}->{len(sel)})')

tdf = df.dropna(subset=[LABEL]).sort_values(['date', 'code'])
Xt = tdf.groupby('code')[sel].transform(lambda c: c.ffill().fillna(0))
y = tdf.groupby('date')[LABEL].transform(
    lambda s: np.clip((s.rank(method='first', pct=True) * RANK_BUCKETS).astype(int),
                      0, RANK_BUCKETS - 1))
groups = tdf.groupby('date', sort=True).size().values
mark('4. 构造训练矩阵')

t = time.time()
model = lgb.LGBMRanker(**LOCKED, label_gain=list(range(RANK_BUCKETS)))
model.fit(Xt, y, group=groups)
train_s = time.time() - t
mark(f'5. 训练 LGBMRanker ({train_s:.1f}s)')

last = df['date'].max()
pm = df['date'] == last
Xp = df.loc[pm, sel].fillna(0)
preds = model.predict(Xp)
out = pd.DataFrame({'code': df.loc[pm, 'code'].values, 'score': preds}) \
    .sort_values('score', ascending=False)
mark('6. 预测最新交易日')

print()
print('=' * 70)
print('  结论')
print('=' * 70)
print(f'  训练样本      {len(tdf):,} 行 x {len(sel)} 特征')
print(f'  预测日期      {last.date()} ({pm.sum()} 只可选)')
print(f'  单次出单总耗时 {time.time()-_t0:.1f} 秒')
print(f'  峰值内存      {rss_mb():.0f} MB')
print(f'  建议服务器内存 {max(4, int(rss_mb()/1024*2+1))} GB 以上 (峰值x2 留余量)')
print()
print('  今日 Top5 (仅验证链路可用, 非交易建议):')
for i, r in enumerate(out.head(5).itertuples(), 1):
    print(f'    #{i}  {r.code}  {r.score:+.4f}')
