"""Build dynamic holding-event signals with timestamps"""
import pandas as pd, numpy as np, json, os
from glob import glob

BASE = 'data/raw/cross_holdings'
ANN_DIR = 'data/raw/announcements'

# Reload IPO events with correct dates from announcements
ipo_rows = []
for f in glob(f'{ANN_DIR}/*.parquet'):
    code = os.path.basename(f).replace('.parquet','')
    try:
        df = pd.read_parquet(f)
        tc = 'title' if 'title' in df.columns else None
        if not tc: continue
        for _, r in df.iterrows():
            title = str(r.get(tc, ''))
            if not any(kw in title for kw in ['参股公司','分拆所属','分拆.*上市']):
                continue
            dt = str(r.get('date', ''))[:10]
            ipo_rows.append({'code': code, 'title': title[:200], 'date': dt})
    except: pass

ipo_df = pd.DataFrame(ipo_rows).drop_duplicates(subset=['code','title'])
# Filter future dates
ipo_df['date_parsed'] = pd.to_datetime(ipo_df['date'], errors='coerce')
ipo_df = ipo_df[ipo_df['date_parsed'] <= '2026-07-03']
print(f"IPO events: {len(ipo_df)}, with dates: {(ipo_df['date']!='').sum()}")

# Load holdings + events
holdings = pd.read_parquet(f'{BASE}/cross_holdings.parquet')
holdings['ratio'] = pd.to_numeric(holdings['ratio'], errors='coerce')
ev2 = pd.read_parquet('data/raw/events_ifind/events_v2.parquet')
ev2['date'] = pd.to_datetime(ev2['date'], errors='coerce')
ev2 = ev2[ev2['date'].dt.year <= 2026]
ev2 = ev2[ev2['date'] <= '2026-07-03']  # no future dates
ev2 = ev2.dropna(subset=['date'])
ev2['code6'] = ev2['code'].str[:6]

# Supply chain map
with open('data/universe/supply_chain_map.json', encoding='utf-8') as f:
    sc = json.load(f)

# Helper: get latest event date for a stock code
def latest_event(code6):
    sub = ev2[ev2['code6'] == code6].sort_values('date', ascending=False)
    return str(sub.iloc[0]['date'])[:10] if len(sub) > 0 else ''

all_signals = []

# 1. IPO events (already have dates)
for _, r in ipo_df.iterrows():
    all_signals.append({
        'holder_code': r['code'], 'investee': '',
        'signal_type': 'portfolio_ipo', 'signal_detail': r['title'][:200],
        'signal_date': r['date'], 'p_level': 1,
    })

# 2. Cross holdings (listed->listed)
for _, r in holdings[holdings['is_listed'] == '是'].iterrows():
    all_signals.append({
        'holder_code': r['stock_code'], 'investee': r['investee'],
        'signal_type': 'cross_holding',
        'signal_detail': f'Holds {r["investee"]} ({r["ratio"]}%, {r["relation"]})',
        'signal_date': latest_event(r['stock_code']), 'p_level': 2,
    })

# 3. Major holdings (>1% non-subsidiary)
sub_mask = holdings['relation'].isin(['子公司','孙公司'])
sig_h = holdings[~sub_mask & ((holdings['ratio'] >= 1) | (holdings['ratio'].isna()))]
for code in sig_h['stock_code'].unique():
    dt = latest_event(code)
    if not dt: continue
    for _, r in sig_h[sig_h['stock_code'] == code].iterrows():
        all_signals.append({
            'holder_code': code, 'investee': r['investee'],
            'signal_type': 'major_holding',
            'signal_detail': f'Holds {r["investee"]} ({r["ratio"]}%, {r["relation"]})',
            'signal_date': dt, 'p_level': 2,
        })

# 4. Supply chain + peer supplier (dynamic)
for chain in sc['chains']:
    leader = chain.get('chain_leader', chain.get('leader', {}))
    leader_name = leader.get('name','?')
    for link in chain.get('demand_links', []):
        suppliers = link.get('a_share_suppliers', [])
        for s in suppliers:
            scode = s['code'][:6]
            sname = s.get('name','')
            dt = latest_event(scode)
            all_signals.append({
                'holder_code': scode,
                'investee': leader_name + ' chain',
                'signal_type': 'supply_chain',
                'signal_detail': 'Supplier(%s) in %s/%s' % (sname, leader_name, link['component']),
                'signal_date': dt, 'p_level': 2,
            })

# 5. Peer supplier
for chain in sc['chains']:
    leader = chain.get('chain_leader', chain.get('leader', {}))
    leader_name = leader.get('name','?')
    for link in chain.get('demand_links', []):
        suppliers = link.get('a_share_suppliers', [])
        for i, s1 in enumerate(suppliers):
            sc1 = s1['code'][:6]
            dt = latest_event(sc1)
            if not dt: continue
            for j, s2 in enumerate(suppliers):
                if i == j: continue
                all_signals.append({
                    'holder_code': s2['code'][:6],
                    'investee': s1.get('name','')[:30],
                    'signal_type': 'peer_supplier',
                    'signal_detail': '%s(%s/%s) had event' % (s1.get('name',''), leader_name, link['component']),
                    'signal_date': dt, 'p_level': 2,
                })

# 6. Fund holdings (>2%)
for _, r in holdings[holdings['investee'].str.contains('基金|投资.*合伙', na=False)].iterrows():
    if pd.notna(r['ratio']) and float(r['ratio']) < 2: continue
    dt = latest_event(r['stock_code'])
    all_signals.append({
        'holder_code': r['stock_code'], 'investee': r['investee'],
        'signal_type': 'fund_holding',
        'signal_detail': 'Fund(%.0f%%)' % float(r['ratio']) if pd.notna(r['ratio']) else 'Fund',
        'signal_date': dt, 'p_level': 3,
    })

# 7. Unlock events (限售解禁) — filter >0.5%流通股比例, keep all dates
import re
unlock_raw = pd.read_parquet('data/raw/events_ifind/events_v2.parquet')
unlock_raw['date'] = pd.to_datetime(unlock_raw['date'], errors='coerce')
unlock_raw = unlock_raw[unlock_raw['event_type'] == 'unlock'].dropna(subset=['date'])

# Extract ratio from content: "占流通股比例X.XXXX%"
def extract_ratio(content):
    if pd.isna(content) or not isinstance(content, str): return 0
    m = re.search(r'占流通股比例([\d.]+)%', content)
    return float(m.group(1)) if m else 0

unlock_raw['ratio_pct'] = unlock_raw['content'].apply(extract_ratio)
unlock_sig = unlock_raw[unlock_raw['ratio_pct'] > 0.5]
print(f"\nUnlock events: {len(unlock_raw)} total, {len(unlock_sig)} > 0.5% threshold")

for _, r in unlock_sig.iterrows():
    all_signals.append({
        'holder_code': r['code'][:6], 'investee': '',
        'signal_type': 'unlock',
        'signal_detail': '%.1f%% unlock on %s' % (r['ratio_pct'], str(r['date'])[:10]),
        'signal_date': str(r['date'])[:10],
        'p_level': 2,
    })

df = pd.DataFrame(all_signals).drop_duplicates(subset=['holder_code','investee','signal_type','signal_date'])
df['p_level'] = df['p_level'].astype(int)
df = df.sort_values('signal_date', ascending=False).reset_index(drop=True)
df.to_parquet(f'{BASE}/holding_event_signals.parquet', index=False)

dates_count = (df['signal_date'] != '').sum()
print(f"\nTotal signals: {len(df)}")
print(f"With dates: {dates_count}/{len(df)} ({dates_count/len(df)*100:.0f}%)")
print(f"By type: {dict(df['signal_type'].value_counts())}")

print("\n=== Latest 20 signals ===")
for _, r in df.head(20).iterrows():
    print("  %s | %s | %s | P%d | %s" % (
        r['signal_date'][:10], r['holder_code'],
        r['signal_detail'][:60], r['p_level'],
        r['signal_type']
    ))

print("\n=== IPO events sample ===")
for _, r in df[df['signal_type']=='portfolio_ipo'].head(10).iterrows():
    print("  %s | %s | %s" % (r['signal_date'], r['holder_code'], r['signal_detail'][:80]))
