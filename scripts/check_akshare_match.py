"""验证 akshare 拉的日K线能否与现有 data/raw/kline 对齐(复权口径)"""
import pandas as pd
import akshare as ak

TESTS = ['000063', '600519', '300750']
OVERLAP_START, OVERLAP_END = '20260601', '20260717'

for code in TESTS:
    try:
        old = pd.read_parquet(f'data/raw/kline/{code}.parquet')
    except Exception as e:
        print(f'{code}: 本地无数据 {e}')
        continue
    old['date'] = pd.to_datetime(old['date'])
    old = old.set_index('date')

    print(f"\n=== {code} ===")
    for adj, label in [('qfq', '前复权'), ('hfq', '后复权'), ('', '不复权')]:
        try:
            new = ak.stock_zh_a_hist(symbol=code, period='daily',
                                     start_date=OVERLAP_START, end_date=OVERLAP_END,
                                     adjust=adj)
        except Exception as e:
            print(f'  {label:5s}: 拉取失败 {e}')
            continue
        if new is None or not len(new):
            print(f'  {label:5s}: 空')
            continue
        new = new.rename(columns={'日期': 'date', '收盘': 'close'})
        new['date'] = pd.to_datetime(new['date'])
        new = new.set_index('date')
        j = old[['close']].join(new[['close']], how='inner', rsuffix='_ak').dropna()
        if not len(j):
            print(f'  {label:5s}: 无重叠日期')
            continue
        diff = (j['close_ak'] / j['close'] - 1).abs()
        print(f'  {label:5s}: 重叠 {len(j)} 天  最大偏差 {diff.max()*100:8.4f}%  '
              f'平均 {diff.mean()*100:7.4f}%  {"✓ 一致" if diff.max() < 0.001 else ""}')

print("\n--- akshare 返回的列 ---")
s = ak.stock_zh_a_hist(symbol='000063', period='daily',
                       start_date='20260710', end_date='20260717', adjust='qfq')
print(list(s.columns))
print(s.tail(3).to_string(index=False))
