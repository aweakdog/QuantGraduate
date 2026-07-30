#!/usr/bin/env python3
"""验证多进程拉新浪日线是否可行

要回答两件事:
  1. py_mini_racer 是线程不安全, 多进程能否绕开(不崩溃)
  2. 并发会不会被新浪限流(成功率下降 / 耗时暴涨)
"""
import argparse
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
KLINE_DIR = ROOT / 'data' / 'raw' / 'kline'
START, END = '20260715', '20260728'


def sina_symbol(code):
    c = str(code)[:6]
    if c.startswith('6'):
        return 'sh' + c
    if c.startswith(('4', '8', '9')):
        return 'bj' + c
    return 'sz' + c


def fetch(code):
    import warnings as w
    w.filterwarnings('ignore')
    import akshare as ak
    t = time.time()
    try:
        df = ak.stock_zh_a_daily(symbol=sina_symbol(code), start_date=START,
                                 end_date=END, adjust='qfq')
        return code, True, len(df) if df is not None else 0, time.time() - t, ''
    except Exception as e:
        return code, False, 0, time.time() - t, f'{type(e).__name__}'


def run(codes, workers):
    ok = fail = 0
    errs = {}
    lat = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch, c) for c in codes]
        for i, f in enumerate(as_completed(futs), 1):
            _, good, _, dt, err = f.result()
            if good:
                ok += 1
                lat.append(dt)
            else:
                fail += 1
                errs[err] = errs.get(err, 0) + 1
            done = int(28 * i / len(codes))
            sys.stdout.write(f'\r  [{"#"*done}{"-"*(28-done)}] {i}/{len(codes)} '
                             f'成功{ok} 失败{fail}')
            sys.stdout.flush()
    total = time.time() - t0
    avg = sum(lat) / len(lat) if lat else 0
    rate = ok / len(codes) * 100
    print(f'\r  [{"#"*28}] {len(codes)}/{len(codes)} '
          f'成功{ok} 失败{fail}  用时{total:.1f}s  '
          f'成功率{rate:.0f}%  单只均{avg:.2f}s')
    if errs:
        print(f'      错误: {errs}')
    return total, rate, len(codes)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32, help='每档抽样只数')
    ap.add_argument('--workers', default='1,4,8', help='逗号分隔的并发档位')
    a = ap.parse_args()

    all_codes = sorted(p.stem for p in KLINE_DIR.glob('*.parquet'))
    random.seed(42)
    sample = random.sample(all_codes, min(a.n, len(all_codes)))
    print(f'样本 {len(sample)} 只, 区间 {START}~{END} (约10个交易日)\n')

    results = []
    for w in [int(x) for x in a.workers.split(',')]:
        print(f'{w} 进程:')
        total, rate, n = run(sample, w)
        results.append((w, total, rate, n))
        time.sleep(2)

    print('\n' + '=' * 62)
    print('  推算: 全市场 5533 只增量更新')
    print('=' * 62)
    print(f"  {'并发':>6}{'样本用时':>10}{'成功率':>8}{'全市场预计':>14}")
    for w, total, rate, n in results:
        est = total / n * 5533 / 60
        flag = '' if rate >= 98 else '  <- 成功率偏低, 疑似限流'
        print(f'  {w:>6}{total:>9.1f}s{rate:>7.0f}%{est:>12.1f} 分钟{flag}')
