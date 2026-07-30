#!/usr/bin/env python3
"""日线更新提速方案实测

现状: 每只股票重拉 2021-至今全量 + 强制单线程 -> 全市场约 3.4 小时
候选:
  A. 全市场快照单次调用   ak.stock_zh_a_spot_em / stock_zh_a_spot
  B. 单只增量短区间拉取   只取最近几天
  C. 多进程并发           py_mini_racer 非线程安全, 但进程间隔离
"""
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import akshare as ak

N_SAMPLE = 5
CODES = ['sh600000', 'sz000001', 'sz300750', 'sh601318', 'sz002594']

TOTAL_STEPS = 2 + N_SAMPLE * 2
_step = [0]


def timed(label, fn):
    _step[0] += 1
    i, n_all = _step[0], TOTAL_STEPS
    filled = int(28 * i / n_all)
    bar = '#' * filled + '-' * (28 - filled)
    sys.stdout.write(f'\r[{bar}] {i}/{n_all}  {label[:34]:34s}')
    sys.stdout.flush()
    t = time.time()
    try:
        r = fn()
        dt = time.time() - t
        n = len(r) if r is not None else 0
        sys.stdout.write(f'\r[{bar}] {i}/{n_all}  {label:38s} {dt:6.2f}s  {n:>6} 行\n')
        sys.stdout.flush()
        return dt, r
    except Exception as e:
        sys.stdout.write(f'\r[{bar}] {i}/{n_all}  {label:38s} 失败: '
                         f'{type(e).__name__}: {str(e)[:50]}\n')
        sys.stdout.flush()
        return None, None


print('=' * 68)
print('  A. 全市场快照 (一次调用拿全部 A 股当日行情)')
print('=' * 68)
dt_em, df_em = timed('东财 stock_zh_a_spot_em', ak.stock_zh_a_spot_em)
if df_em is not None:
    print(f'     列: {list(df_em.columns)[:12]}')
dt_sina, df_sina = timed('新浪 stock_zh_a_spot', ak.stock_zh_a_spot)
if df_sina is not None:
    print(f'     列: {list(df_sina.columns)[:12]}')

print()
print('=' * 68)
print('  B. 单只拉取: 全量 vs 增量短区间')
print('=' * 68)
full_times, inc_times = [], []
for c in CODES[:N_SAMPLE]:
    dt1, _ = timed(f'{c} 全量 2021-01-01 起',
                   lambda c=c: ak.stock_zh_a_daily(symbol=c, start_date='20210101',
                                                   end_date='20260728', adjust='qfq'))
    dt2, _ = timed(f'{c} 增量 最近10天',
                   lambda c=c: ak.stock_zh_a_daily(symbol=c, start_date='20260715',
                                                   end_date='20260728', adjust='qfq'))
    if dt1:
        full_times.append(dt1)
    if dt2:
        inc_times.append(dt2)
    time.sleep(0.3)

print()
print('=' * 68)
print('  测算 (全市场 5533 只)')
print('=' * 68)
if full_times:
    avg = sum(full_times) / len(full_times)
    print(f'  当前做法 全量+单线程:  {avg:.2f}s/只 -> {avg*5533/3600:.2f} 小时')
if inc_times:
    avg_i = sum(inc_times) / len(inc_times)
    print(f'  增量+单线程:           {avg_i:.2f}s/只 -> {avg_i*5533/3600:.2f} 小时')
    for w in (8, 16):
        print(f'  增量+{w}进程:           {avg_i*5533/w/60:>6.1f} 分钟')
if dt_em:
    print(f'  全市场快照单次调用:    {dt_em:.2f}s  -> 覆盖全部 {len(df_em)} 只')
