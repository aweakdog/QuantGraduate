"""OOS 泛化测试: 100 只双创股 (完全未见)"""
import sys, os, json, joblib
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.feature_engine import read_kline, calc_technical_features, calc_labels, calc_fund_features
from pipeline.feature_engine import _load_events, calc_event_v2_features, _precompute_theme_events, _load_theme_map

BASE = 'D:/myAI/WorkBuddy-workspace/quant-strategy'

# Load model
model = joblib.load(f'{BASE}/data/processed/model_merged_v2.pkl')
features = model.feature_name_

# Load test stocks
with open(f'{BASE}/data/universe/oos_100_list.json') as f:
    codes6 = json.load(f)

# Load fund flow
ff = pd.read_parquet(f'{BASE}/data/raw/fund_flow_full/oos_100_ff.parquet')
ff['date'] = pd.to_datetime(ff['date'])

# Load events
ev = _load_events()

# Precompute theme events (macro)
test_dates_for_theme = pd.Series(pd.date_range('2025-03-01', '2026-07-03'))
theme_events = _precompute_theme_events(test_dates_for_theme)
theme_map = _load_theme_map()

# Macro data
cn_pmi = pd.read_parquet(f'{BASE}/data/raw/macro/中国PMI.parquet')
cn_pmi['date'] = pd.to_datetime(cn_pmi['月份'].str.replace('年','-').str.replace('月份','-01'), errors='coerce')
cn_pmi['date'] = cn_pmi['date'] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
cn_pmi_v = cn_pmi[['date','制造业-指数']].rename(columns={'制造业-指数':'cn_pmi'})

us_ism = pd.read_parquet(f'{BASE}/data/raw/macro/美国ISM制造业PMI.parquet')
us_ism['date'] = pd.to_datetime(us_ism['日期'], errors='coerce')
us_ism['date'] = us_ism['date'] + pd.Timedelta(days=1)
us_ism_v = us_ism[['date','今值']].rename(columns={'今值':'us_ism_pmi'})

# Supply chain
with open(f'{BASE}/data/universe/supply_chain_map.json', encoding='utf-8') as f:
    sc = json.load(f)
leader_map = {}
for chain in sc['chains']:
    for link in chain['demand_links']:
        for s in link['a_share_suppliers']:
            lk = s['code'][:6]
            if lk not in leader_map:
                leader_map[lk] = {"cnt": 0, "exp": 0, "binding_sum": 0.0}
            leader_map[lk]["cnt"] += 1
            leader_map[lk]["exp"] = max(leader_map[lk]["exp"], {"核心": 3, "高": 2, "中": 1}.get(s.get('exposure',''), 0))
            leader_map[lk]["binding_sum"] += s.get('scoring', {}).get('binding', 0)

# Build features for each stock
all_rows = []
for i, c6 in enumerate(codes6):
    df_k = read_kline(c6)
    if df_k is None or len(df_k) < 60:
        continue

    tech = calc_technical_features(df_k)
    if tech is None:
        continue
    labels = calc_labels(df_k)
    result = tech.merge(labels, on='date', how='left')
    for c in [c for c in tech.columns if c != 'date']:
        result[c] = result[c].shift(1)

    # Fund flow
    sub_ff = ff[ff['code'] == c6].copy()
    if len(sub_ff) > 0:
        fund = calc_fund_features(sub_ff)
        if fund is not None:
            fund['date'] = pd.to_datetime(fund['date']) + pd.Timedelta(days=1)
            result = result.merge(fund, on='date', how='left')

    # Macros (ffill)
    result = result.merge(cn_pmi_v, on='date', how='left')
    result['cn_pmi'] = result['cn_pmi'].ffill()
    result = result.merge(us_ism_v, on='date', how='left')
    result['us_ism_pmi'] = result['us_ism_pmi'].ffill()

    # Events
    sub_ev = ev[ev['code'] == c6].copy()
    if len(sub_ev) > 0:
        sub_ev['date'] = pd.to_datetime(sub_ev['date'], errors='coerce')
        sub_ev = sub_ev.dropna(subset=['date'])
        ev2 = calc_event_v2_features(sub_ev, result['date'])
        if ev2 is not None:
            ev2['date'] = pd.to_datetime(ev2['date']) + pd.Timedelta(days=1)
            result = result.merge(ev2, on='date', how='left')

    # Chain leader
    info = leader_map.get(c6, {})
    result['has_leader'] = 1 if info.get('cnt', 0) > 0 else 0
    result['leader_count'] = info.get('cnt', 0)
    result['leader_exp'] = info.get('exp', 0)
    result['leader_binding_sum'] = info.get('binding_sum', 0.0)

    # Theme events
    themes = theme_map.get(c6, [])
    for theme in themes:
        if theme in theme_events:
            tev = theme_events[theme].copy()
            tev['date'] = pd.to_datetime(tev['date']) + pd.Timedelta(days=1)
            result = result.merge(tev, on='date', how='left')

    # Market-wide events
    if '__all__' in theme_events:
        tev_all = theme_events['__all__'].copy()
        tev_all['date'] = pd.to_datetime(tev_all['date']) + pd.Timedelta(days=1)
        result = result.merge(tev_all, on='date', how='left')

    # MA3/8/21 for non-event columns
    _roll_cols = [c for c in result.columns
                  if c not in ('date', 'code')
                  and not c.startswith(('fwd_', 'ev_', 'tev_', 'ann_', 'has_', 'leader_'))
                  and c not in ('cn_pmi', 'us_ism_pmi')
                  and result[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    for w in (3, 8, 21):
        for c in _roll_cols:
            new_name = f'{c}_ma{w}'
            if new_name not in result.columns:
                result[new_name] = result[c].rolling(w, min_periods=w // 2 + 1).mean()

    result['code'] = f"{c6}.SZ" if c6.startswith('3') else f"{c6}.SH"
    all_rows.append(result)

    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/{len(codes6)}] {c6}', flush=True)

# Combine
df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
if 'code' not in df.columns or len(df) == 0:
    print('No stocks built')
    sys.exit(1)

print(f'Built: {df.shape}, {df["code"].nunique()} stocks')

# Filter to test period (2025-03-11 ~ 2026-06-25)
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= '2025-03-11') & (df['date'] <= '2026-06-25')]
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['fwd_1d_ret'])

print(f'Test period: {df.shape}, {df["code"].nunique()} stocks')

# Predict with existing model
# Align features: fill missing with 0
for f in features:
    if f not in df.columns:
        df[f] = 0.0

X = df[[f for f in features if f in df.columns]].values
med = np.nanmedian(X, axis=0)
med = np.where(np.isnan(med), 0, med)
X = np.where(np.isnan(X), med, X)
df['pred'] = model.predict(X)

# Backtest: 去弱留强, T+1, TC=0.06%
dates_list = sorted(df['date'].unique())
TC = 0.0006

print(f'\n{"="*55}')
print(f'  OOS 100 — ChiNext/STAR (完全未见)')
print(f'  Period: {dates_list[0].date()} ~ {dates_list[-1].date()} ({len(dates_list)} days)')
print(f'{"="*55}')
print(f'  {"Top N":>6} {"Return":>10} {"Ann.":>10} {"Sharpe":>8} {"MDD":>8} {"Win%":>6}')
print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*8} {"-"*8} {"-"*6}')

for top_n in [2, 3, 4, 5, 7]:
    picks = {}
    for d in dates_list:
        g = df[df['date'] == d]
        picks[d] = set(g.sort_values('pred', ascending=False).head(top_n)['code'].values)

    h = set()
    rets = []
    for i in range(len(dates_list) - 1):
        t = dates_list[i]
        pt = picks[t]
        if i == 0:
            h = pt
            continue
        t1 = dates_list[i + 1]
        a = df[(df['date'] == t1) & (df['code'].isin(h))]
        if len(a) == 0:
            h = picks[t1]
            continue
        gr = a['fwd_1d_ret'].mean()
        nh = picks[t1]
        sp = len(h - nh) / len(h) if h else 0
        rets.append(gr - sp * TC)
        h = nh

    if len(rets) < 10:
        continue
    rr = pd.Series(rets)
    cum = (1 + rr).prod() - 1
    ann = (1 + cum) ** (252 / len(rr)) - 1
    peak_r = (1 + rr).cumprod().expanding().max()
    mdd = ((1 + rr).cumprod() / peak_r - 1).min()
    sharpe = rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else 0
    win = (rr > 0).mean()
    print(f'  {top_n:>6} {cum*100:>9.1f}% {ann*100:>9.1f}% {sharpe:>7.2f} {mdd*100:>7.1f}% {win*100:>5.1f}%')
