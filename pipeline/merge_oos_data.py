"""Merge OOS data into main data files (schema-safe)"""
import pandas as pd

BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy/data/raw'

# 1. Events merge - ensure schema match
ev_v2 = pd.read_parquet(f'{BASE}/events_ifind/events_v2.parquet')
ev_oos = pd.read_parquet(f'{BASE}/events_ifind/oos_events.parquet')

# Ensure oos has all v2 columns
for c in ev_v2.columns:
    if c not in ev_oos.columns:
        ev_oos[c] = None

# Ensure dtypes match
for c in ev_v2.columns:
    if c in ev_oos.columns:
        ev_oos[c] = ev_oos[c].astype(ev_v2[c].dtype)

print(f'Events v2: {ev_v2.shape} ({ev_v2["code"].nunique()} codes, cols={list(ev_v2.columns)})')
print(f'Events oos: {ev_oos.shape} ({ev_oos["code"].nunique()} codes, cols={list(ev_oos.columns)})')

ev_combined = pd.concat([ev_v2, ev_oos], ignore_index=True)
ev_combined = ev_combined.drop_duplicates(subset=['code','date','event_type'], keep='last')
ev_combined.to_parquet(f'{BASE}/events_ifind/events_v2.parquet', index=False)
print(f'Events merged: {ev_combined.shape} ({ev_combined["code"].nunique()} codes)')

# 2. Fund flow merge
ff_old = pd.read_parquet(f'{BASE}/fund_flow_full/fundflow_history.parquet')
ff_new = pd.read_parquet(f'{BASE}/fund_flow_full/oos_fundflow_full.parquet')
for c in ff_old.columns:
    if c in ff_new.columns:
        ff_new[c] = ff_new[c].astype(ff_old[c].dtype, errors='ignore')
ff_combined = pd.concat([ff_old, ff_new], ignore_index=True)
ff_combined = ff_combined.drop_duplicates(subset=['date','code'], keep='last').sort_values(['code','date'])
ff_combined.to_parquet(f'{BASE}/fund_flow_full/fundflow_history.parquet', index=False)
print(f'\nFund flow old: {ff_old.shape} ({ff_old["code"].nunique()} codes)')
print(f'Fund flow new: {ff_new.shape} ({ff_new["code"].nunique()} codes)')
print(f'Fund flow merged: {ff_combined.shape} ({ff_combined["code"].nunique()} codes)')

# 3. Margin merge
mt_old = pd.read_parquet(f'{BASE}/MainNetFlow/margintrade_history.parquet')
mt_new = pd.read_parquet(f'{BASE}/MainNetFlow/oos_margintrade.parquet')
for c in mt_old.columns:
    if c in mt_new.columns:
        mt_new[c] = mt_new[c].astype(mt_old[c].dtype, errors='ignore')
mt_combined = pd.concat([mt_old, mt_new], ignore_index=True)
mt_combined = mt_combined.drop_duplicates(subset=['date','code'], keep='last').sort_values(['code','date'])
mt_combined.to_parquet(f'{BASE}/MainNetFlow/margintrade_history.parquet', index=False)
print(f'\nMargin old: {mt_old.shape} ({mt_old["code"].nunique()} codes)')
print(f'Margin new: {mt_new.shape} ({mt_new["code"].nunique()} codes)')
print(f'Margin merged: {mt_combined.shape} ({mt_combined["code"].nunique()} codes)')

print('\n=== All merged ✅ ===')
