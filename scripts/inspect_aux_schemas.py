"""摸清 资金流/两融/事件/公告/宏观/商品 各文件的真实 schema 与覆盖度"""
import pathlib

import pandas as pd

pd.set_option('display.width', 200)


def show(label, path, n=3):
    p = pathlib.Path(path)
    print(f"\n{'='*70}\n{label}\n  {path}")
    if not p.exists():
        print("  [缺失]")
        return None
    df = pd.read_parquet(p)
    print(f"  行数 {len(df):,} | 列 {len(df.columns)}")
    print(f"  列名: {list(df.columns)}")
    if 'date' in df.columns:
        d = pd.to_datetime(df['date'], errors='coerce')
        print(f"  日期: {d.min()} ~ {d.max()}")
    if 'code' in df.columns:
        print(f"  股票数: {df['code'].nunique()}")
    print(df.head(n).to_string(index=False)[:800])
    return df


ff = show("【资金流 consolidated】feature_engine 直接读",
          "data/raw/fund_flow_full/fundflow_history.parquet")
if ff is not None:
    need = ["main_force_net", "main_force_pct", "dde_net", "mtss_balance", "fund_flow"]
    print(f"  feature_engine 需要的列: {need}")
    for c in need:
        if c in ff.columns:
            nn = ff[c].notna().sum()
            print(f"    {c:16s} 存在  非空 {nn:>8,} ({nn/len(ff)*100:5.1f}%)")
        else:
            print(f"    {c:16s} 缺失")

show("【两融 consolidated】", "data/raw/MainNetFlow/margintrade_history.parquet")
show("【事件 v2 (iFinD)】", "data/raw/events_ifind/events_v2.parquet")

print(f"\n{'='*70}\n【公告】data/raw/announcements/")
ad = pathlib.Path('data/raw/announcements')
fs = sorted(ad.glob('*.parquet'))
print(f"  {len(fs)} 个文件")
if fs:
    d = pd.read_parquet(fs[0])
    print(f"  样例 {fs[0].name}: {len(d)} 行, 列={list(d.columns)}")
    print(d.head(2).to_string(index=False)[:500])

print(f"\n{'='*70}\n【宏观】data/raw/macro/")
md = pathlib.Path('data/raw/macro')
for f in sorted(md.glob('*.parquet')):
    try:
        d = pd.read_parquet(f)
        dc = next((c for c in ('日期', 'date', '月份', '时间') if c in d.columns), None)
        last = ''
        if dc:
            last = f"  最新={pd.to_datetime(d[dc], errors='coerce').max()}"
        print(f"  {f.name:34s} {len(d):>6} 行  列={list(d.columns)[:5]}{last}")
    except Exception as e:
        print(f"  {f.name:34s} 读取失败 {e}")
