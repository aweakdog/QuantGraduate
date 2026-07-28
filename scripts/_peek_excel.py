import pandas as pd

F = 'data/processed/wf_daily_c1_base_ts2022-09-01_te2026-07-16_cap100000.xlsx'
x = pd.ExcelFile(F)
print('文件:', F)
print('Sheets:', x.sheet_names)
for sh in x.sheet_names:
    print(f'  {sh:8s} {len(pd.read_excel(F, sheet_name=sh)):5d} 行')

ops = pd.read_excel(F, sheet_name='操作清单')
cols = ['买入日期', '卖出日期', '股票代码', '股票名称', '股数', '买入价', '卖出价', '净收益', '收益率%']
print('\n=== 操作清单 最近 10 笔 ===')
print(ops[cols].tail(10).to_string(index=False))
print(f"\n共 {len(ops)} 笔 | 胜率 {(ops['净收益']>0).mean()*100:.1f}% | "
      f"平均收益率 {ops['收益率%'].mean():.2f}% | 净收益合计 ¥{ops['净收益'].sum():,.0f}")

print('\n=== 最赚钱 5 笔 ===')
print(ops.nlargest(5, '净收益')[cols].to_string(index=False))
print('\n=== 最亏 5 笔 ===')
print(ops.nsmallest(5, '净收益')[cols].to_string(index=False))

d = pd.read_excel(F, sheet_name='每日汇总')
print('\n=== 每日汇总 最后 5 天 ===')
print(d[['日期', '组合总资产', '仓位占比', '持仓只数', '实际日收益', '基准日收益', '超额收益']].tail(5).to_string(index=False))
print('\n=== 最后一天持仓 ===')
print(d['持仓明细'].iloc[-1])
