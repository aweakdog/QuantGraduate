"""
扩展事件覆盖到全量196只: 从 announcements 标题关键词提取 P0-P4 事件

策略:
  1. 遍历全部 200 只股票的 announcements 目录
  2. 用关键词匹配 title 判断事件类型/级别/方向
  3. 合并 events_clean (109只) + announcements-keyword (196只)
  4. 输出去重后的 events_v2.parquet
"""

import sys, re, glob, os
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd

BASE = Path("D:/myAI/WorkBuddy-workspace/quant-strategy")
ANN_DIR = BASE / "data" / "raw" / "announcements"
EVENTS_CLEAN = BASE / "data" / "raw" / "events_ifind" / "events_clean.parquet"
EVENTS_DAILY_DIR = BASE / "data" / "raw" / "events_daily"
OUT = BASE / "data" / "raw" / "events_ifind" / "events_v2.parquet"

# ─── 事件关键词映射 (title → event_type, p_level, direction) ─────
EVENT_RULES = [
    # P0: 诉讼/立案/处罚/监管 — 最严重
    (r"(?:立案|被立案|证监会立案|被证监会)", "lawsuit", "P0", -1),
    (r"(?:诉讼|被起诉|法律诉讼)", "lawsuit", "P0", -1),
    (r"(?:处罚|被处罚|行政处罚|罚款)", "regulatory_action", "P0", -1),
    (r"(?:风险警示|退市风险|ST)", "regulatory_action", "P0", -1),
    (r"(?:监管措施|监管关注|监管函)", "regulatory", "P0", -1),
    # P1: 减持/预亏 — 较严重
    (r"(?:减持|减持计划|大股东减持)", "reduction", "P1", -1),
    (r"(?:业绩预亏|预亏|亏损|净利润为负)", "earnings_revise", "P1", 0),
    (r"(?:业绩下降|同比下降|大幅下降)", "earnings_revise", "P1", 0),
    # P2: 增持/回购/中标 — 利好
    (r"(?:增持|增持计划|大股东增持)", "increase", "P2", 1),
    (r"(?:回购|股份回购|回购计划)", "buyback_plan", "P2", 1),
    (r"(?:中标|重大项目|重大合同)", "big_contract", "P2", 1),
    (r"(?:业绩预增|预增|大幅增长)", "earnings_revise", "P2", 0),
    # P3: 投资/分红/激励 — 中性偏利好
    (r"(?:利润分配|分红|派息|权益分派)", "dividend", "P3", 1),
    (r"(?:股权激励|员工持股)", "equity_incentive", "P3", 1),
    (r"(?:投资|对外投资|设立子公司)", "expansion", "P3", 1),
    (r"(?:解禁|限售股|锁定期)", "pledge", "P3", -1),
    (r"(?:质押|股份质押)", "pledge", "P3", -1),
    (r"(?:募集资金|定增|配股)", "expansion", "P3", 1),
]

def classify_title(title: str) -> tuple:
    """从标题提取 (event_type, p_level, direction)"""
    for pattern, etype, plevel, direction in EVENT_RULES:
        if re.search(pattern, title):
            return (etype, plevel, direction)
    return (None, None, None)

def main():
    # 1. Load existing events_clean
    ev_clean = pd.read_parquet(str(EVENTS_CLEAN))
    existing_titles = set(ev_clean["title"].dropna().unique())
    existing_keys = set()  # (code, title) for dedup
    for _, row in ev_clean.iterrows():
        existing_keys.add((row["code"], row.get("title", "")))
    
    print(f"[EVENTS_CLEAN] {len(ev_clean)} rows, {ev_clean['code'].nunique()} stocks, {len(existing_titles)} unique titles")

    # 2. Process announcements for ALL stocks
    ann_files = sorted(glob.glob(str(ANN_DIR / "*.parquet")))
    print(f"[ANNOUNCEMENTS] {len(ann_files)} files")
    
    new_events = []
    for f in ann_files:
        code6 = os.path.basename(f).replace(".parquet", "")
        code = f"{code6}.SZ" if code6.startswith(("0", "3")) else f"{code6}.SH"
        
        df = pd.read_parquet(f)
        if "title" not in df.columns or "date" not in df.columns:
            continue
        
        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            if not title or (code, title) in existing_keys:
                continue
            
            event_type, p_level, direction = classify_title(title)
            if event_type is None:
                continue
            
            new_events.append({
                "code": code,
                "name": title.split(":")[0] if ":" in title else "",
                "event_type": event_type,
                "p_level": p_level,
                "direction": direction,
                "date": str(row["date"])[:10],
                "title": title,
                "snippet": title[:200],
            })

    print(f"[NEW] {len(new_events)} new events from announcements")

    if len(new_events) == 0:
        print("No new events found. Output unchanged.")
        return

    # 3. Merge: keep events_clean, add new with dup check
    new_df = pd.DataFrame(new_events)
    merged = pd.concat([ev_clean, new_df], ignore_index=True)
    
    # 4. Merge daily event signals (commodity, US overnight, news)
    daily_patterns = ["commodity_clean_*.parquet", "us_overnight_clean_*.parquet", "news_events_clean_*.parquet"]
    for pattern in daily_patterns:
        for f in sorted(EVENTS_DAILY_DIR.glob(pattern)):
            try:
                daily_df = pd.read_parquet(f)
                required_cols = {"date", "stock_code", "event_name", "p_level", "dir_hard", "impact"}
                if required_cols.issubset(set(daily_df.columns)):
                    daily_df = daily_df.rename(columns={"stock_code": "code", "event_name": "event_type", "dir_hard": "direction"})
                    daily_df["date"] = pd.to_datetime(daily_df["date"])
                    daily_df["title"] = daily_df.get("reason", daily_df["event_type"])
                    merged = pd.concat([merged, daily_df], ignore_index=True)
                    print(f"  + {f.name}: {len(daily_df)} rows")
            except:
                pass
    
    merged = merged.drop_duplicates(subset=["code", "title"], keep="first")
    merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    
    merged.to_parquet(str(OUT))
    print(f"\n[OUT] {OUT}")
    print(f"  Total: {len(merged)} rows, {merged['code'].nunique()} stocks")
    print(f"  New: {len(merged) - len(ev_clean)} rows added")
    print(f"  P-level: {merged['p_level'].value_counts().to_dict()}")
    print(f"  Types: {merged['event_type'].value_counts().head(10).to_dict()}")

if __name__ == "__main__":
    main()
