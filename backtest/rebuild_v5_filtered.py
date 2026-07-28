"""重建 v5 训练数据 → 排除 13 只无事件股票 → 回测"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.feature_engine import (
    read_kline, calc_technical_features, calc_labels,
    calc_fund_features, calc_event_v2_features,
)
import pandas as pd, numpy as np, lightgbm as lgb, joblib

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data")

EXCLUDE = {'300449','688561','688206','300220','688039','300423','601828','000301','300940','002862','600463','600036','002573'}

# Load data
ff = pd.read_parquet(os.path.join(DATA, "raw/fund_flow_full/fundflow_history.parquet"))
ff["date"] = pd.to_datetime(ff["date"])
ev = pd.read_parquet(os.path.join(DATA, "raw/events_ifind/events_v2.parquet"))

orig = pd.read_parquet(os.path.join(DATA, "processed/training_data_v6.parquet"))
orig["date"] = pd.to_datetime(orig["date"])
orig_codes6 = set(c[:6] for c in orig['code'].unique())

all_ff = set(ff['code'].unique())
new_codes6 = sorted([c for c in all_ff if c not in orig_codes6 and c not in EXCLUDE])
print(f"New stocks (with events): {len(new_codes6)}")

# Chain leader map
sc_path = os.path.join(DATA, "universe/supply_chain_map.json")
with open(sc_path, encoding='utf-8') as f:
    sc = json.load(f)
lm = {}
ev_map = {"核心": 3, "高": 2, "中": 1}
for chain in sc['chains']:
    for link in chain['demand_links']:
        for s in link['a_share_suppliers']:
            sc6 = s['code'][:6]
            if sc6 not in lm: lm[sc6] = {"cnt": 0, "exp": 0, "binding_sum": 0.0}
            lm[sc6]["cnt"] += 1; lm[sc6]["exp"] = max(lm[sc6]["exp"], ev_map.get(s['exposure'], 0))
            lm[sc6]["binding_sum"] += s.get('scoring', {}).get('binding', 0)

# Macro
pmip = os.path.join(DATA, "raw/macro/中国PMI.parquet")
if os.path.exists(pmip):
    df_pmi = pd.read_parquet(pmip)
    df_pmi['date'] = pd.to_datetime(df_pmi['月份'].str.replace('年','-').str.replace('月份','-01'), errors='coerce') + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    pmi_v = df_pmi[['date','制造业-指数']].rename(columns={'制造业-指数':'cn_pmi'})
ismp = os.path.join(DATA, "raw/macro/美国ISM制造业PMI.parquet")
if os.path.exists(ismp):
    df_ism = pd.read_parquet(ismp)
    df_ism['date'] = pd.to_datetime(df_ism['日期'], errors='coerce') + pd.Timedelta(days=1)
    ism_v = df_ism[df_ism['商品']=='美国ISM制造业PMI报告'][['date','今值']].rename(columns={'今值':'us_ism_pmi'})

# Build features for new stocks
new_rows = []
for i, c6 in enumerate(new_codes6):
    mkt = ".SZ" if c6.startswith(('0','3')) else ".SH"
    cf = f"{c6}{mkt}"
    
    df_k = read_kline(c6)
    if df_k is None or len(df_k) < 60: continue
    tech = calc_technical_features(df_k)
    if tech is None: continue
    labels = calc_labels(df_k)
    result = tech.merge(labels, on='date', how='left')
    for c in [x for x in tech.columns if x != 'date']:
        result[c] = result[c].shift(1)
    
    # Fund flow
    sub_ff = ff[ff['code'] == c6].copy()
    if len(sub_ff) > 0:
        fund = calc_fund_features(sub_ff)
        if fund is not None:
            fund['date'] = pd.to_datetime(fund['date']) + pd.Timedelta(days=1)
            result = result.merge(fund, on='date', how='left')
    
    # Macro
    if os.path.exists(pmip):
        result = result.merge(pmi_v, on='date', how='left'); result['cn_pmi'] = result['cn_pmi'].ffill()
    if os.path.exists(ismp):
        result = result.merge(ism_v, on='date', how='left'); result['us_ism_pmi'] = result['us_ism_pmi'].ffill()
    
    # Events
    sub_ev = ev[ev['code'] == c6].copy()
    if len(sub_ev) > 0:
        sub_ev['date'] = pd.to_datetime(sub_ev['date'], errors='coerce')
        sub_ev = sub_ev.dropna(subset=['date'])
        # Add required columns expected by calc_event_v2_features
        sub_ev['dir_hard'] = sub_ev['direction'].fillna(0).astype(int)
        PWEIGHT = {0: 10, 1: 5, 2: 2, 3: 1}
        sub_ev['p_w'] = sub_ev['p_level'].map(PWEIGHT).fillna(1).astype(float)
        sub_ev['impact'] = sub_ev['p_w'] * sub_ev['dir_hard']
        ev2 = calc_event_v2_features(sub_ev, result['date'])
        if ev2 is not None:
            ev2['date'] = pd.to_datetime(ev2['date']) + pd.Timedelta(days=1)
            result = result.merge(ev2, on='date', how='left')
    
    # Chain leader
    info = lm.get(c6, {})
    result['has_leader'] = 1 if info.get('cnt',0) > 0 else 0
    result['leader_count'] = info.get('cnt',0)
    result['leader_exp'] = info.get('exp',0)
    result['leader_binding_sum'] = info.get('binding_sum',0.0)
    
    # Rolling MA features (same as feature_engine v6)
    _roll_cols = [c for c in result.columns
                  if c not in ('date', 'code')
                  and not c.startswith(('fwd_', 'ev_', 'tev_', 'ann_', 'has_', 'leader_'))
                  and c not in ('cn_pmi', 'us_ism_pmi')
                  and result[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    for w in (3, 8, 21):
        for c in _roll_cols:
            new_name = f'{c}_ma{w}'
            if new_name not in result.columns:
                result[new_name] = result[c].rolling(w, min_periods=w//2+1).mean()
    
    result['code'] = cf
    new_rows.append(result)
    
    if (i+1) % 10 == 0:
        print(f'  [{i+1}/{len(new_codes6)}] {cf} ({len(result)} rows)', flush=True)

new_df = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
print(f"New: {new_df.shape}, {new_df['code'].nunique() if 'code' in new_df.columns else 0} stocks")

# Combine & clean
combined = pd.concat([orig, new_df], ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
combined = combined[combined['date'] >= '2020-01-01']
for c in combined.select_dtypes(include=[np.number]).columns:
    combined[c] = combined[c].replace([np.inf, -np.inf], np.nan)
combined = combined.dropna(subset=['fwd_1d_ret'])
print(f"Combined: {combined.shape}, {combined['code'].nunique()} stocks")

# Train/test split
dates = sorted(combined['date'].unique())
split_i = int(len(dates) * 0.8)
train_end, test_start = dates[split_i-1], dates[split_i]
skip = {'date','code','fwd_1d_ret','fwd_5d_ret','fwd_10d_ret','fwd_20d_ret'}
feats = [c for c in combined.columns if c not in skip]
label = 'fwd_1d_excess'
combined[label] = combined.groupby('date')['fwd_1d_ret'].transform(lambda x: x - x.mean())

train = combined[combined['date'] <= train_end]
test = combined[combined['date'] >= test_start]
X_tr = train[feats].values; y_tr = train[label].values
med = np.nanmedian(X_tr, axis=0); med = np.where(np.isnan(med), 0, med)
X_tr = np.where(np.isnan(X_tr), med, X_tr)

model = lgb.LGBMRegressor(n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.05,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbose=-1)
model.fit(X_tr, y_tr)

test = test.copy()
test['pred'] = model.predict(test[feats].values)

# Backtest
dl = sorted(test['date'].unique())
TC = 0.0006
print(f"\n{'='*55}")
print(f"  FILTERED - {combined['code'].nunique()} stocks, test={test_start.date()}~{dl[-1].date()}")
print(f"{'='*55}")
print(f"  {'Top N':>6} {'Return':>10} {'Ann.':>10} {'Sharpe':>8} {'MDD':>8} {'Win%':>6}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

for top_n in [2, 3, 4, 5, 7]:
    picks = {d: set(test[test['date']==d].sort_values('pred',ascending=False).head(top_n)['code'].values) for d in dl}
    h = set(); rets = []
    for i in range(len(dl)-1):
        t = dl[i]; pt = picks[t]
        if i == 0: h = pt; continue
        t1 = dl[i+1]
        a = test[(test['date']==t1) & (test['code'].isin(h))]
        if len(a)==0: h = picks[t1]; continue
        gr = a['fwd_1d_ret'].mean()
        nh = picks[t1]; sp = len(h-nh)/len(h) if h else 0
        rets.append(gr - sp*TC); h = nh
    if len(rets) < 10: continue
    rr = pd.Series(rets)
    cum = (1+rr).prod()-1; ann = (1+cum)**(252/len(rr))-1
    peak_r = (1+rr).cumprod().expanding().max()
    mdd = ((1+rr).cumprod()/peak_r-1).min()
    sharpe = rr.mean()/rr.std()*np.sqrt(252) if rr.std()>0 else 0
    win = (rr>0).mean()
    print(f"  {top_n:>6} {cum*100:>9.1f}% {ann*100:>9.1f}% {sharpe:>7.2f} {mdd*100:>7.1f}% {win*100:>5.1f}%")

# Feature importance
imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
total = imp.sum()
ev_cols = [c for c in imp.index if 'ev_' in c or 'tev_' in c]
ev_wt = sum(imp[c] for c in ev_cols if c in imp.index)
print(f"\n  Event weight: {ev_wt}/{total} = {ev_wt/total*100:.2f}%")
print(f"  Top5 features: {dict(imp.head(5))}")

# Save model
model_path = os.path.join(DATA, "processed/model_merged_v2.pkl")
joblib.dump(model, model_path)
print(f"  Model saved: {model_path}")
