#!/usr/bin/env python3
"""本金规模造成的选股价格偏差: 小资金买不起高价股, 被迫改买便宜票"""
import json, glob
import pandas as pd

def load(cap):
    f = glob.glob(f'data/processed/wf_daily_cap{cap}_B35C1_*.json')
    return json.load(open(f[0])) if f else None

res = {}
for cap in (20000, 100000):
    d = load(cap)
    if not d:
        continue
    buys = pd.DataFrame([t for t in d['trades'] if t['action'] == 'buy'])
    res[cap] = buys
    alloc = cap / 3
    print(f'=== ¥{cap:,} (单只预算约 ¥{alloc:,.0f} -> 可买最高价约 ¥{alloc/100:.0f}) ===')
    print(f'  买入 {len(buys)} 笔, 涉及 {buys["code"].nunique()} 只')
    print(f'  成交价: 最低 ¥{buys["price"].min():.2f} | 中位 ¥{buys["price"].median():.2f} '
          f'| 最高 ¥{buys["price"].max():.2f}')
    print(f'  单笔金额: 中位 ¥{buys["gross"].median():,.0f} | 最大 ¥{buys["gross"].max():,.0f}')
    print(f'  手续费触及¥5下限的比例: {(buys["fee"] <= 5.001).mean()*100:.1f}%')
    print()

if len(res) == 2:
    a, b = set(res[20000]['code']), set(res[100000]['code'])
    print('=== 选股重合度 ===')
    print(f'  2万买过 {len(a)} 只 | 10万买过 {len(b)} 只 | 共同 {len(a & b)} 只')
    print(f'  重合率(占10万) {len(a & b)/len(b)*100:.0f}%')
    only_big = b - a
    if only_big:
        px = res[100000].groupby('code')['price'].mean()
        top = px[list(only_big)].sort_values(ascending=False).head(8)
        print(f'\n  仅10万能买、2万买不起的高价股 (共{len(only_big)}只), 均价最高的:')
        for c, p in top.items():
            print(f'    {c:>10}  ¥{p:>8.2f}')
