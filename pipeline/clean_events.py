"""
清洗 events_ifind — 修复 P0 financial_fraud 误分类

问题: 1015条 "financial_fraud P0" 中大部分是常规公告(审计报告/内控制度/招股书)
      关键词匹配过宽, 把任何提到"差错""舞弊""虚假"的制度文件都标为欺诈

修复规则:
  - 标题含 "制度|审计报告|内控|招股|年报|季报|半年报|财务报告|业绩预告|业绩快报|责任追究|自我评价|内部控制|管理制度"
    → 这些是例行披露文件, 不是实际欺诈事件
    → 重分类为 P3 + event_type="routine_filing"
  
  - 标题含 "立案|处罚|警示|监管措施|违规|造假|虚增|舞弊|行政处罚|证监局|警示函"
    (且不含上面例行文件关键词)
    → 保留 P0 + event_type="regulatory_action"
  
  - 其余 financial_fraud → 降为 P2 + event_type="regulatory_filing"

输出: data/raw/events_ifind/events_clean.parquet
"""
import pandas as pd
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = settings.DATA_DIR
events_path = DATA / "raw" / "events_ifind" / "events.parquet"
out_path = DATA / "raw" / "events_ifind" / "events_clean.parquet"

df = pd.read_parquet(events_path)
print(f"Original: {len(df)} events")

# Routine filing patterns
ROUTINE = re.compile(
    r"制度|审计报告|内控|招股|年报|季报|半年报|财务报告|业绩预告|业绩快报|"
    r"责任追究|自我评价|内部控制|管理制度|公司章程|股东大会议事规则|"
    r"董事会议事规则|监事会议事规则|信息披露制度|投资者关系|募集资金",
    re.IGNORECASE
)

# Actual enforcement patterns
ENFORCEMENT = re.compile(
    r"立案|处罚|警示函|监管措施|违规|造假|虚增|舞弊|行政处罚|"
    r"证监局|证监会.*?[查处罚]|交易所.*?[处分监管]|退市风险|ST戴帽",
    re.IGNORECASE
)

# Only fix financial_fraud P0
mask_ff = (df["event_type"] == "financial_fraud") & (df["p_level"] == "P0")
ff = df[mask_ff].copy()
print(f"financial_fraud P0 to clean: {len(ff)}")

routine_mask = ff["title"].apply(lambda t: bool(ROUTINE.search(str(t))))
enforcement_mask = ff["title"].apply(lambda t: bool(ENFORCEMENT.search(str(t)))) & ~routine_mask

# Apply reclassification
df.loc[mask_ff & routine_mask, "event_type"] = "routine_filing"
df.loc[mask_ff & routine_mask, "p_level"] = "P3"
df.loc[mask_ff & routine_mask, "direction"] = 0

df.loc[mask_ff & enforcement_mask, "event_type"] = "regulatory_action"
df.loc[mask_ff & enforcement_mask, "p_level"] = "P0"
df.loc[mask_ff & enforcement_mask, "direction"] = -1

# Remaining financial_fraud (not routine, not enforcement) → P2
remaining_mask = mask_ff & ~routine_mask & ~enforcement_mask
df.loc[remaining_mask, "event_type"] = "regulatory_filing"
df.loc[remaining_mask, "p_level"] = "P2"
df.loc[remaining_mask, "direction"] = 0

# Save
df.to_parquet(out_path, index=False)

# Report
print(f"\nCleaned distribution:")
print(f"  event_type x p_level:")
ct = pd.crosstab(df["event_type"], df["p_level"])
print(ct.to_string())
print(f"\n  P0 count: {(df['p_level']=='P0').sum()} (was 1015)")
print(f"  P0 samples:")
for _, r in df[df["p_level"]=="P0"].head(5).iterrows():
    print(f"    [{r['event_type']}] {r['name']} {r['date']}: {r['title'][:60]}")
print(f"\nSaved: {out_path}")
