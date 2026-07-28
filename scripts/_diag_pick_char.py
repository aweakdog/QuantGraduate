"""诊断: 模型选出来的票是什么特征? 是不是在追已经暴涨的票?"""
import pandas as pd, numpy as np, json
from pathlib import Path

res = json.loads(Path('data/processed/wf_daily_v33_select_timing_dart_ts2022-01-01_te2026-07-16_cap100000.json').read_text())
d = pd.DataFrame(res['daily'])
d['date'] = pd.to_datetime(d['date'])

DF = pd.read_parquet('data/processed/training_data_v24.parquet')
DF['date'] = pd.to_datetime(DF['date'])
DF['code'] = DF['code'].astype(str)

cols = ['ret_1d', 'ret_5d', 'fwd_1d_ret', 'atr_pct', 'pos_20', 'vol_ratio']
cols = [c for c in cols if c in DF.columns]
idx = DF.set_index(['date', 'code'])

# 全市场每日均值
mkt_mean = DF.groupby('date')[cols].mean()

recs = []
for _, row in d.iterrows():
    dt = row['date']
    picks = row['holdings'] if isinstance(row['holdings'], list) else []
    for c in picks:
        try:
            r = idx.loc[(dt, str(c)), cols]
        except KeyError:
            continue
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rec = {'date': dt}
        for col in cols:
            rec[col] = r[col]
            rec[col + '_mkt'] = mkt_mean.loc[dt, col] if dt in mkt_mean.index else np.nan
        recs.append(rec)

pk = pd.DataFrame(recs)
print(f"样本: {len(pk)} 个持仓记录\n")
print(f"{'指标':<12} {'选中票均值':>12} {'全市场均值':>12} {'差异':>12}")
print("-" * 52)
for col in cols:
    a = pk[col].mean()
    b = pk[col + '_mkt'].mean()
    print(f"{col:<12} {a:>12.4f} {b:>12.4f} {a-b:>+12.4f}")

print("\n=== 关键: 选中票【当日】涨幅分布 ===")
if 'ret_1d' in pk.columns:
    q = pk['ret_1d'].quantile([.1, .25, .5, .75, .9])
    for k, v in q.items():
        print(f"  P{int(k*100):<3d}: {v*100:+.2f}%")
    print(f"  当日涨停(>9.5%)占比: {(pk['ret_1d'] > 0.095).mean()*100:.1f}%")
    print(f"  当日涨幅>5% 占比   : {(pk['ret_1d'] > 0.05).mean()*100:.1f}%")

print("\n=== 选中票次日收益 按【当日涨幅】分组 ===")
if 'ret_1d' in pk.columns and 'fwd_1d_ret' in pk.columns:
    pk['bucket'] = pd.cut(pk['ret_1d'], [-1, -0.05, -0.02, 0.02, 0.05, 0.095, 1],
                          labels=['<-5%', '-5~-2%', '-2~2%', '2~5%', '5~9.5%', '>9.5%'])
    g = pk.groupby('bucket', observed=True)['fwd_1d_ret'].agg(['mean', 'count'])
    g['mean'] = (g['mean'] * 100).round(3)
    print(g.to_string())
