#!/usr/bin/env python3
"""检查最大盈利交易是否由异常价格跳变(未复权/脏数据)造成"""
import json, glob, sys
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pair_trades(trades):
    pending, rows = defaultdict(deque), []
    for t in sorted(trades, key=lambda x: (x['date'], x['action'] != 'sell')):
        code = str(t['code'])
        if t['action'] == 'buy':
            pending[code].append(t)
            continue
        if not pending[code]:
            continue
        b = pending[code].popleft()
        ret = t['price'] / b['price'] - 1
        rows.append({'code': code, 'buy_date': b['date'], 'sell_date': t['date'],
                     'buy_px': b['price'], 'sell_px': t['price'],
                     'ret_pct': ret * 100, 'shares': b['shares'],
                     'pnl': t['net'] - b['gross'] - b['fee']})
    return pd.DataFrame(rows)


def check_kline(code, d0, d1):
    """回看该股在持仓窗口内的原始K线, 看是否有异常跳空"""
    c6 = str(code)[:6]
    for p in ROOT.glob(f'data/raw/kline/*{c6}*.parquet'):
        kl = pd.read_parquet(p).rename(columns={
            '时间': 'date', '收盘价': 'close', '开盘价': 'open',
            '最高价': 'high', '最低价': 'low'})
        kl['date'] = pd.to_datetime(kl['date'])
        w = kl[(kl['date'] >= pd.Timestamp(d0)) & (kl['date'] <= pd.Timestamp(d1))]
        if w.empty:
            return None
        w = w.sort_values('date')
        jump = (w['open'] / w['close'].shift(1) - 1).abs().max()
        return {'n_bars': len(w), 'max_overnight_jump_pct': float(jump * 100) if pd.notna(jump) else 0.0,
                'first_close': float(w['close'].iloc[0]), 'last_close': float(w['close'].iloc[-1])}
    return None


tag = sys.argv[1] if len(sys.argv) > 1 else 'clean_h10g00_'
f = glob.glob(f'data/processed/wf_daily_{tag}*_cap100000.json')[0]
d = json.load(open(f))
tr = pair_trades(d['trades'])
tr['year'] = pd.to_datetime(tr['sell_date']).dt.year

print(f'=== {tag} 配对交易 {len(tr)} 笔 ===')
print('\n逐年盈亏:')
for y, g in tr.groupby('year'):
    print(f"  {y}: {len(g):>3} 笔 | 胜率 {(g['ret_pct']>0).mean()*100:>5.1f}% | "
          f"平均 {g['ret_pct'].mean():>+6.2f}% | 合计盈亏 ¥{g['pnl'].sum():>+12,.0f}")

print('\n单笔收益率 Top 10:')
top = tr.nlargest(10, 'ret_pct')
for _, r in top.iterrows():
    k = check_kline(r['code'], r['buy_date'], r['sell_date'])
    jm = f"{k['max_overnight_jump_pct']:.1f}%" if k else 'n/a'
    flag = ' <<< 异常跳空' if k and k['max_overnight_jump_pct'] > 25 else ''
    print(f"  {r['code']:>10} {r['buy_date']}~{r['sell_date']} "
          f"{r['buy_px']:>8.2f}->{r['sell_px']:>8.2f} {r['ret_pct']:>+7.1f}% "
          f"pnl ¥{r['pnl']:>+10,.0f} | 窗口内最大隔夜跳空 {jm}{flag}")

print(f"\n最大单笔占总盈利比重:")
tot = tr[tr['pnl'] > 0]['pnl'].sum()
for _, r in tr.nlargest(5, 'pnl').iterrows():
    print(f"  {r['code']:>10} {r['sell_date']} ¥{r['pnl']:>+12,.0f}  = 总盈利的 {r['pnl']/tot*100:.1f}%")
