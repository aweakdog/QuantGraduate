"""
同花顺7×24快讯 → 事件信号

流程:
  1. fetch_ths_news.py 获取全量快讯 (~80条)
  2. 关键词匹配 → 事件分类 + P-level + 方向
  3. 提取涉及股票 → 写入 events_daily

覆盖:
  - 机构研报/评级变动 (高盛/摩根等)
  - 财报业绩预告
  - 行业新闻 (半导体/AI/新能源等)
  - 政策新闻
  - 盘后公告

运行: 交易日 09:00 / 12:00 / 17:00
"""
import json, os, sys, subprocess, re, pandas as pd
from datetime import datetime, timezone, timedelta

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY8 = NOW.strftime("%Y%m%d")
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
EVENTS_DIR = os.path.join(DATA_DIR, "raw", "events_daily")

PY = "C:/Users/admin/.workbuddy/binaries/python/envs/ths/Scripts/python.exe"
FETCH = "C:/Users/admin/.workbuddy/skills/a-stock-news/scripts/fetch_ths_news.py"

# ─── 事件分类规则 ────────────────────────────────────────────
# (pattern, event_type, p_level, direction)
RULES = [
    # P0: 重大利空
    (r'(?:立案|被立案|证监会调查)', 'lawsuit', 'P0', -1),
    (r'(?:退市|风险警示|ST|暂停上市)', 'delist_risk', 'P0', -1),
    (r'(?:处罚|行政处罚|罚款)', 'regulatory', 'P0', -1),
    # P1: 机构研报评级变动
    (r'(?:高盛|摩根士丹利|大摩|瑞银|摩根大通|小摩|花旗|中金).*(?:上调|买入|增持|超配)', 'research_upgrade', 'P1', 1),
    (r'(?:高盛|摩根士丹利|大摩|瑞银|摩根大通|小摩|花旗|中金).*(?:下调|卖出|减持|低配)', 'research_downgrade', 'P1', -1),
    (r'(?:大幅上调|强烈推荐|重点推荐)', 'research_upgrade', 'P1', 1),
    # P1: 业绩
    (r'(?:业绩预增|净利润大增|同比增长|扭亏为盈)', 'earnings_surge', 'P1', 1),
    (r'(?:业绩预亏|净利润为负|大幅亏损)', 'earnings_loss', 'P1', -1),
    # P1: 增减持
    (r'(?:减持|减持计划|大股东减持|拟减持)', 'reduction', 'P1', -1),
    (r'(?:增持|增持计划|大股东增持)', 'increase', 'P1', 1),
    # P2: 回购/中标/合同
    (r'(?:回购|股份回购|回购计划)', 'buyback', 'P2', 1),
    (r'(?:中标|重大合同|重大项目|签订.*合同)', 'contract', 'P2', 1),
    # P2: 行业景气/涨价
    (r'(?:涨价|供不应求|供需紧张|产能不足)', 'industry_boom', 'P2', 1),
    (r'(?:价格下跌|产能过剩|需求疲软|降价)', 'industry_slow', 'P2', -1),
    # P2: 政策利好/利空
    (r'(?:政策.*利好|支持.*发展|补贴|减免.*税)', 'policy_positive', 'P2', 1),
    (r'(?:政策.*收紧|监管.*加强|限制|调查)', 'policy_negative', 'P2', -1),
    # P2: AI/科技催化
    (r'(?:AI|人工智能|大模型|算力|芯片|半导体)(?:.*突破|.*发布|.*创新|.*利好)', 'tech_catalyst', 'P2', 1),
]

# 行业板块提取规则 (标题→板块概念)
SECTOR_KEYWORDS = {
    "芯片|半导体|集成电路": "芯片概念",
    "AI|人工智能|大模型|算力": "AI应用",
    "新能源|光伏|风电|储能": "新能源",
    "新能源汽车|电动车|锂电池": "新能源汽车",
    "黄金|白银|有色金属": "有色金属",
    "原油|石油|天然气|能源": "能源",
    "银行|保险|券商|金融": "金融",
    "医药|医疗|生物|创新药": "医药",
    "消费|白酒|食品|零售": "消费",
    "房地产|地产|楼市": "房地产",
    "军工|航天|国防": "军工",
    "5G|6G|通信|光通信": "5G/通信",
}

def fetch_news():
    r = subprocess.run([PY, FETCH], capture_output=True, text=True, timeout=60)
    if not r.stdout:
        return []
    try:
        data = json.loads(r.stdout)
        return data.get("items", [])
    except:
        return []

def extract_codes(title):
    """从标题提取A股代码"""
    codes = set()
    for m in re.finditer(r'(6[0-9]{5}|3[0-9]{5}|0[0-9]{5}|002[0-9]{3}|688[0-9]{3}|300[0-9]{3})', title):
        codes.add(m.group(1))
    return codes

def extract_sectors(title):
    """从标题提取涉及板块"""
    sectors = set()
    for pattern, concept in SECTOR_KEYWORDS.items():
        if re.search(pattern, title):
            sectors.add(concept)
    return sectors

def main():
    print(f"=== 同花顺快讯 → 事件 [{TODAY}] ===")
    
    items = fetch_news()
    print(f"  快讯: {len(items)}条")
    
    signals = []
    seen = set()
    
    for item in items:
        title = item.get("title", "")
        news_time = item.get("time", "")
        
        if not title or title in seen:
            continue
        seen.add(title)
        
        # 匹配事件规则
        matched = False
        for pattern, ev_type, p_level, direction in RULES:
            if re.search(pattern, title):
                matched = True
                break
        
        if not matched:
            # 本日无匹配规则，跳过纯行情/国际新闻
            if any(kw in title for kw in ["美股", "欧股", "日经", "原油", "黄金", 
                    "美元", "汇率", "期货", "指数", "国债"]):
                continue
            # 有公司名的也保留
            codes = extract_codes(title)
            sectors = extract_sectors(title)
            if not codes and not sectors:
                continue
            ev_type, p_level, direction = "industry_news", "P3", 0
        
        codes = extract_codes(title)
        sectors = extract_sectors(title)
        
        dir_label = "bullish" if direction > 0 else ("bearish" if direction < 0 else "neutral")
        marker = "🟢" if direction > 0 else ("🔴" if direction < 0 else "⚪")
        
        if codes:
            for c in codes:
                signals.append({
                    "code": c, "name": "",
                    "event_type": ev_type, "p_level": p_level,
                    "direction": dir_label, "change_pct": 0,
                    "reason": f"{ev_type}: {title[:60]}",
                    "time": news_time,
                })
            print(f"  {marker} {news_time} {list(codes)} | {p_level} | {title[:50]}")
        elif sectors:
            # 板块级信号
            for s in sectors:
                signals.append({
                    "code": f"__{s}__", "name": s,
                    "event_type": ev_type, "p_level": p_level,
                    "direction": dir_label, "change_pct": 0,
                    "reason": f"{ev_type}: {title[:60]}",
                    "time": news_time,
                })
            print(f"  {marker} {news_time} [{','.join(sectors)}] | {p_level} | {title[:50]}")
    
    if not signals:
        print("  ⚠ 无可转化事件")
        return
    
    df = pd.DataFrame(signals)
    df["date"] = TODAY
    os.makedirs(EVENTS_DIR, exist_ok=True)
    
    path = os.path.join(EVENTS_DIR, f"ths_news_{TODAY8}.parquet")
    df.to_parquet(path, index=False)
    
    clean = df.rename(columns={"code":"stock_code","event_type":"event_name","p_level":"p_level"})
    clean["dir_hard"] = clean["direction"].map({"bullish":1,"bearish":-1,"neutral":0}).fillna(0)
    clean["impact"] = clean["dir_hard"] * clean["p_level"].map({"P0":10,"P1":5,"P2":2,"P3":1}).fillna(0)
    cp = os.path.join(EVENTS_DIR, f"ths_news_clean_{TODAY8}.parquet")
    clean.to_parquet(cp, index=False)
    
    print(f"\n  ✓ {len(signals)}条 → {path}")
    by_type = df["event_type"].value_counts().to_dict()
    print(f"  类型分布: {json.dumps(by_type, ensure_ascii=False)}")

if __name__ == "__main__":
    main()
