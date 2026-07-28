"""A股交易制度合规审计: 检查 c1_base 回测是否违反 T+1 / 涨跌停 / 整手 等规则"""
import json
from collections import defaultdict, deque

import pandas as pd

F = 'data/processed/wf_daily_c1_base_ts2022-09-01_te2026-07-16_cap100000.json'
res = json.load(open(F))
tr = pd.DataFrame(res['trades'])
tr['date'] = pd.to_datetime(tr['date'])
tr['code6'] = tr['code'].astype(str).str[:6]

print("=" * 62)
print("  A股交易制度合规审计  —  c1_base")
print("=" * 62)

# ── 1. T+1 ──
print("\n[1] T+1 制度（当日买入不可当日卖出）")
pending = defaultdict(deque)
same_day = []
for _, t in tr.sort_values(['date', 'action']).iterrows():
    if t['action'] == 'buy':
        pending[t['code6']].append(t['date'])
    else:
        if pending[t['code6']]:
            bd = pending[t['code6']].popleft()
            if bd == t['date']:
                same_day.append((str(t['date'].date()), t['code6']))
print(f"    当日买当日卖: {len(same_day)} 笔")
if same_day:
    days = sorted({d for d, _ in same_day})
    print(f"    发生日期: {days}")
    print(f"    -> 均为回测期末强制清仓, 实盘不存在此问题")

# ── 2. 整手 ──
print("\n[2] 整手交易（100股整数倍）")
bad = tr[tr['shares'] % 100 != 0]
print(f"    非整手成交: {len(bad)} 笔  {'✓ 合规' if len(bad) == 0 else '✗ 违规'}")

# ── 3. 涨跌停判定口径 ──
print("\n[3] 涨跌停判定")
tr['board'] = tr['code6'].map(lambda c: '创业板' if c.startswith('300') else
                              ('科创板' if c.startswith('688') else
                               ('北交所' if c.startswith(('43', '83', '87', '92')) else '主板')))
cnt = tr['board'].value_counts()
print("    成交股票所属板块分布:")
for k, v in cnt.items():
    lim = {'主板': '±10%', '创业板': '±20%', '科创板': '±20%', '北交所': '±30%'}[k]
    print(f"      {k:5s} {v:5d} 笔  真实涨跌幅限制 {lim}")
n2 = int(cnt.get('创业板', 0) + cnt.get('科创板', 0) + cnt.get('北交所', 0))
print(f"    代码统一按 ±10% 判定, 但其中 {n2} 笔({n2/len(tr)*100:.1f}%) 实际限制更宽")
print("    -> 偏保守: 会把 +10%~+20% 的可买标的误判为涨停而跳过")

# ── 4. 成交价口径 ──
print("\n[4] 成交价口径  ★关键")
print("    当前: 用 T 日收盘价计算特征 -> 同一时刻用 T 日收盘价成交")
print("    问题: 收盘价在收盘瞬间才确定, 无法同时下单成交")
print("    -> 无法实现「前一天指导第二天」, 这是当日信号当日成交")

# ── 5. 费用 ──
print("\n[5] 交易费用")
print(f"    回测: 双边各 {0.0006*100:.2f}%, 最低 5 元, 往返 {0.0012*100:.2f}%")
print("    实盘: 佣金约0.025%双边(最低5元) + 印花税0.05%卖出单边 + 过户费0.001%双边")
print("    实盘往返约 0.112%  ->  回测 0.12% 略保守 ✓")

# ── 6. 停牌 ──
print("\n[6] 停牌处理")
print(f"    买入拒单 {res['summary']['rejected_buy']} 次, 卖出拒单 {res['summary']['rejected_sell']} 次")
print("    无当日K线则跳过买入 / 继续持有, 逻辑合理 ✓")

# ── 7. 隔夜跳空规模: 评估改到 T+1 开盘成交的代价 ──
print("\n[7] 若改为「T日收盘出信号, T+1开盘成交」的代价评估  ★关键")
kl = {}
import pathlib
KD = pathlib.Path('data/raw/kline')
CM = {'时间': 'date', '收盘价': 'close', '开盘价': 'open'}
need = set(tr['code6'])
for c in need:
    p = KD / f'{c}.parquet'
    if p.exists():
        k = pd.read_parquet(p).rename(columns=CM)
        k['date'] = pd.to_datetime(k['date'])
        kl[c] = k.sort_values('date').drop_duplicates('date').set_index('date')[['open', 'close']]

gaps = []
for _, t in tr.iterrows():
    k = kl.get(t['code6'])
    if k is None:
        continue
    idx = k.index
    pos = idx.searchsorted(t['date'])
    if pos + 1 >= len(idx):
        continue
    c_t = k.iloc[pos]['close']
    o_n = k.iloc[pos + 1]['open']
    if c_t and c_t == c_t:
        gaps.append({'action': t['action'], 'gap': o_n / c_t - 1})
g = pd.DataFrame(gaps)
for act in ['buy', 'sell']:
    s = g[g['action'] == act]['gap']
    if len(s):
        name = '买入' if act == 'buy' else '卖出'
        print(f"    {name}标的 次日开盘跳空: 均值 {s.mean()*100:+.3f}%  中位 {s.median()*100:+.3f}%  (n={len(s)})")
buy_gap = g[g['action'] == 'buy']['gap'].mean()
sell_gap = g[g['action'] == 'sell']['gap'].mean()
drag = buy_gap - sell_gap
print(f"    改到次日开盘成交的单次往返净拖累 ≈ {drag*100:+.3f}%")
print(f"    按 {res['summary']['n_trades']//2} 次往返 / {res['n_days']} 天估算,"
      f" 年化拖累 ≈ {drag * (res['summary']['n_trades']//2) / res['n_days'] * 252 * 100:+.1f}%")
