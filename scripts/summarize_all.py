#!/usr/bin/env python3
"""Summarize all experiment results into a comparison table."""
import json, glob

def load(tag):
    files = glob.glob(f'data/processed/wf_daily_pit_{tag}*_cap100000.json')
    if not files:
        return None
    return json.load(open(files[0]))['summary']

rows = []

tags = [
    ('h5 no_guard', 'cache_base'),
    ('h5 guard10%', 'g10'),
    ('h5 guard20%', 'g20'),
    ('h10 no_guard', 'h10'),
    ('h10 guard10%', 'h10g10'),
    ('h10 guard15%', 'h10g15'),
    ('grid 0.40/3', 'gridB40C3'),
    ('guard 0%', 'guard00pct'),
    ('guard 5%', 'guard05pct'),
    ('guard 10%', 'guard10pct'),
    ('guard 15%', 'guard15pct'),
    ('guard 20%', 'guard20pct'),
    ('guard 30%', 'guard30pct'),
]

for name, tag in tags:
    s = load(tag)
    if s:
        rows.append((name, s))

hdr = f"{'plan':>16}{'tot%':>8}{'ann%':>7}{'sharpe':>7}{'dd%':>7}{'exc_ann%':>9}{'IR':>6}{'cost%':>7}{'trd':>5}"
print(hdr)
print('-' * len(hdr))
for name, s in rows:
    print(f"{name:>16}{s['total_return_pct']:>8.1f}{s['annualized_return_pct']:>7.1f}"
          f"{s['sharpe']:>7.2f}{s['max_dd_pct']:>7.1f}{s['excess_annual_pct']:>9.1f}"
          f"{s['information_ratio']:>6.2f}{s['total_cost_pct']:>7.1f}{s['n_trades']:>5}")
