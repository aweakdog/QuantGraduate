"""校验公告更新结果 (容忍个别文件 schema 异常)"""
import pathlib

import pandas as pd

fs = sorted(pathlib.Path('data/raw/announcements').glob('*.parquet'))
mx, tot, bad, badcols = [], 0, [], {}
for f in fs:
    try:
        d = pd.read_parquet(f)
    except Exception as e:
        bad.append((f.name, type(e).__name__))
        continue
    if 'date' not in d.columns:
        badcols[f.name] = list(d.columns)[:6]
        continue
    tot += len(d)
    m = pd.to_datetime(d['date'], errors='coerce').max()
    if pd.notna(m):
        mx.append(m)

s = pd.Series(mx)
print(f"公告文件 {len(fs)} 个 | 可读 {len(mx)} | 读取失败 {len(bad)} | 无date列 {len(badcols)}")
print(f"总计 {tot:,} 条")
print(f"\n最新日期分布 top6:\n{s.value_counts().head(6).to_string()}")
print(f"\n整体最新: {s.max().date()}   中位: {s.median().date()}   最旧: {s.min().date()}")
stale = (s < pd.Timestamp('2026-07-01')).sum()
print(f"仍停留在 7月前的股票数: {stale}")

if bad:
    print(f"\n读取失败样例: {bad[:5]}")
if badcols:
    print(f"\n无date列样例: {list(badcols.items())[:5]}")

print("\n--- 抽查 000063 最近 6 条 ---")
d = pd.read_parquet('data/raw/announcements/000063.parquet')
d['date'] = pd.to_datetime(d['date'])
print(d.sort_values('date').tail(6)[['date', 'type', 'title']].to_string(index=False)[:700])
