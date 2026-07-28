"""拉取全关注圈限售解禁数据"""
import sys, time, json, os
sys.path.insert(0, 'C:/Users/admin/.workbuddy/skills/unified-finance-data/scripts')
from astock_source import cmd_unlock
import pandas as pd

BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy/data'

# Get all stock codes
td = pd.read_parquet(f'{BASE}/processed/training_data_v6.parquet')
codes = sorted(set(c[:6] for c in td['code'].unique()))
print(f"Total stocks: {len(codes)}")

# Batch pull
all_hist = []
all_future = []

for i, code in enumerate(codes):
    try:
        r = cmd_unlock(code)
        for h in r.get('history', []):
            ratio = float(h.get('ratio', 0) or 0)
            all_hist.append({'code': code, 'date': h['date'], 'ratio': ratio, 'type': h.get('type','')})
        for u in r.get('upcoming', []):
            ratio = float(u.get('ratio', 0) or 0)
            all_future.append({'code': code, 'date': u['date'], 'ratio': ratio, 'type': u.get('type','')})
    except Exception as e:
        print(f"  ERR {code}: {e}")
    
    time.sleep(0.15)
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(codes)}] hist={len(all_hist)}, future={len(all_future)}", flush=True)

# Save
hist_df = pd.DataFrame(all_hist)
fut_df = pd.DataFrame(all_future)

os.makedirs(f'{BASE}/raw/cross_holdings', exist_ok=True)
hist_df.to_parquet(f'{BASE}/raw/cross_holdings/unlock_history.parquet', index=False)
fut_df.to_parquet(f'{BASE}/raw/cross_holdings/unlock_upcoming.parquet', index=False)

print(f"\nDone:")
print(f"  History: {len(hist_df)} rows, {hist_df['code'].nunique()} stocks")
print(f"  Upcoming: {len(fut_df)} rows, {fut_df['code'].nunique()} stocks")
print(f"  Hist >0.5%: {len(hist_df[hist_df['ratio'] > 0.5])}")
print(f"  Fut >0.5%: {len(fut_df[fut_df['ratio'] > 0.5])}")

if len(fut_df) > 0:
    print(f"\nUpcoming >0.5%:")
    big = fut_df[fut_df['ratio'] > 0.5].sort_values('ratio', ascending=False)
    for _, r in big.iterrows():
        print(f"  {r['date']} | {r['code']} | {r['ratio']:.2f}% | {r['type']}")
