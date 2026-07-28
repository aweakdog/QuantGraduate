#!/usr/bin/env python3
"""对比修复特征筛选泄漏前后的结果"""
import json, glob

def load(pat):
    f = glob.glob(f'data/processed/wf_daily_{pat}*_cap100000.json')
    return json.load(open(f[0])) if f else None

def row(name, d):
    s = d['summary']
    return (name, s['total_return_pct'], s['annualized_return_pct'], s['sharpe'],
            s['max_dd_pct'], s['excess_annual_pct'], s['information_ratio'],
            s['total_cost_pct'], s['n_trades'], s.get('cash_days_pct', 0))

def show(title, rows):
    print(f'\n=== {title} ===')
    hdr = (f"{'plan':>14}{'tot%':>8}{'ann%':>7}{'sharpe':>7}{'dd%':>7}"
           f"{'exc_ann%':>9}{'IR':>6}{'cost%':>7}{'trd':>5}{'cash%':>7}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r[0]:>14}{r[1]:>8.1f}{r[2]:>7.1f}{r[3]:>7.2f}{r[4]:>7.1f}'
              f'{r[5]:>9.1f}{r[6]:>6.2f}{r[7]:>7.1f}{r[8]:>5}{r[9]:>7.1f}')

# ── 无泄漏 ──
clean = []
for h in (5, 10):
    for g, lab in (('00', '0%'), ('10', '10%'), ('15', '15%')):
        d = load(f'clean_h{h}g{g}_')
        if d:
            clean.append(row(f'h{h} guard{lab}', d))
show('修复后 (无泄漏特征集)', clean)

# ── 泄漏版 ──
leaky = []
for pat, name in [('pit_cache_base', 'h5 guard0%'), ('pit_g10', 'h5 guard10%'),
                  ('pit_h10_', 'h10 guard0%'), ('pit_h10full', 'h10 guard0% (07-27)')]:
    d = load(pat)
    if d:
        leaky.append(row(name, d))
show('修复前 (泄漏特征集, 仅供对照)', leaky)

# ── 特征集差异 ──
a = load('pit_h10_')
b = load('clean_h5g00_')
if a and b:
    fa, fb = set(a['selected_features']), set(b['selected_features'])
    print(f'\n=== 特征集差异 ===')
    print(f'泄漏版 {len(fa)} 个 | 无泄漏版 {len(fb)} 个 | 交集 {len(fa & fb)} 个')
    print(f'仅泄漏版有 ({len(fa - fb)}): {sorted(fa - fb)}')
    print(f'仅无泄漏版有 ({len(fb - fa)}): {sorted(fb - fa)}')
    print(f'筛选截止日: {b.get("feat_select_cutoff")}')

# ── 分段稳健性 ──
print('\n=== 分段稳健性 (无泄漏) ===')
for h in (5, 10):
    for g, lab in (('00', '0%'), ('10', '10%'), ('15', '15%')):
        d = load(f'clean_h{h}g{g}_')
        if not d or not d.get('stability'):
            continue
        st = d['stability']
        segs = ' | '.join(f"{x['segment']} {x['excess_annual_pct']:+.1f}%" for x in st)
        ok = all(x['excess_annual_pct'] > 0 for x in st)
        print(f"  h{h} guard{lab:>4}: {segs}   {'两段都赢 ✓' if ok else '✗'}")
