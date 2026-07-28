"""
权威机构研报/言论 → A股事件信号

数据源: iFinD MCP search_news
覆盖: 高盛/摩根士丹利/瑞银/摩根大通/中金/花旗等

流程:
  1. 搜索各机构最新研报/评级变动
  2. 提取: 机构, 股票, 评级方向, 目标价
  3. 匹配A股标的 → 生成P-level事件
  4. 写入 events_daily

运行:
  python pull_news_events.py

自动化: 交易日 08:30 + 17:45
"""
import json, os, sys, re, pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY8 = NOW.strftime("%Y%m%d")
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
EVENTS_DIR = os.path.join(DATA_DIR, "raw", "events_daily")

MCP_TOKEN = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjtIFPAyXTksV4T-hKnzLqkW0Q1j-02SSnHyIzSMgGqD74Rj1lZto2oIAynV5gHiZXjTgX8ARvE1NtnBNLbdHMADuomjMNRHAXpPE83sCL4ehLGL6zb5_n8XVzLwr_RuJ4SZiekMR3sEMGNePywrP2flMO_K6R0suTvFTlSWU5WxOYMKqLxUOciZnZqTnxUs6_Lnj6He4XBEgul2VdJX4w6lcPq5ibDx7CDp-8SzW_FW0CBkREtIWBbyuqHaQyWdnUbg6nPoCo3sD3ipTL3ereUqX33GY8mn8dYfIFKZShADp5kGziTtqWRLQ.gQm_IZm7qxG8OKz-.mHYCbUproLUp1qLvMntUQ5rq6e27ORuzqnhXvhkIVFbA5UsTZBq_1UqJuq4XlN5EuI6j2o91dgWFz2vIHhm7482C1vcpwDTlUC48j_UymGR03dX8iiriSA-qE7ZQJLx50YFrG7aFw5sALibKzwDGVETilkI9upyDUu5s7tMg3cIhj0GUWU-8xso-AZf_frGahYyEzZsK4EHKHBxxVmE5IghBnJcTvjvB-Hs46nrhbeQ2wr2aSP82bq8JtXaHvstS6CC_63YS_jB7KWBF1sqkP25138A7y31xzOlmMEW_GIuDOElFXT2SXJ3qbGQmYBg_EwPMyGdy4rs2xk74WmQNyCzrdRIY86zPOlWqyt2EHJi9GHwxOjgAWdJ8eQ0o_kCsw7lyvYoWwQbyeqcs2rOTrhLeMbI.-AlkZXTtGcQkQqGT_JGRzw"
MCP_NEWS_URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-news-mcp"

# ─── 机构研报搜索配置 ────────────────────────────────────────
INSTITUTIONS = [
    "高盛", "摩根士丹利", "大摩", "瑞银", "摩根大通", "小摩",
    "花旗", "中金", "里昂", "美银", "野村", "中信建投", "国泰君安"
]

SEARCH_TEMPLATES = [
    lambda inst: f"{inst} 研报 评级 目标价 A股",
    lambda inst: f"{inst} 上调 评级 A股",
    lambda inst: f"{inst} 下调 评级 A股",
]

# A股关键词映射 (匹配研报中提到的公司名 → 代码)
COMPANY_MAP = {}
_map_path = os.path.join(DATA_DIR, "universe", "stock_list.parquet")
if os.path.exists(_map_path):
    try:
        df = pd.read_parquet(_map_path)
        for _, r in df.iterrows():
            name = str(r.get("股票简称", ""))
            code = str(r.get("股票代码", ""))[:6]
            if name and code:
                COMPANY_MAP[name] = code
    except: pass

def mcp_news(query, size=5):
    body = json.dumps({
        'jsonrpc': '2.0', 'method': 'tools/call',
        'params': {'name': 'search_news', 'arguments': {
            'query': query, 'size': size,
            'time_start': (NOW - timedelta(days=2)).strftime("%Y-%m-%d"),
            'time_end': TODAY
        }},
        'id': 1
    }).encode()
    req = Request(MCP_NEWS_URL, data=body,
        headers={'Authorization': MCP_TOKEN, 'Content-Type': 'application/json'},
        method='POST')
    try:
        with urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
            text = resp['result']['content'][0]['text']
            data = json.loads(text)
            return json.loads(data.get('data', '[]'))
    except Exception as e:
        return []

def extract_stock_code(text):
    """从新闻文本中提取A股代码"""
    codes = set()
    # 匹配 600xxx/300xxx/000xxx/002xxx/688xxx
    for m in re.finditer(r'(6[0-9]{5}|3[0-9]{5}|0[0-9]{5}|002[0-9]{3}|688[0-9]{3})', text):
        codes.add(m.group(1))
    
    # 匹配公司名
    for name, code in COMPANY_MAP.items():
        if name in text:
            codes.add(code)
    
    return codes

def classify_event(title, content, institution):
    """研报 → 事件类型/级别/方向"""
    text = f"{title} {content}"
    
    direction = 0  # 0=中性, 1=利好, -1=利空
    p_level = "P2"
    
    # 评级方向
    if re.search(r'上调|买入|增持|超配|推荐|看好', text):
        direction = 1
        p_level = "P2"
    if re.search(r'下调|卖出|减持|低配|看空|回避', text):
        direction = -1
        p_level = "P2"
    if re.search(r'大幅上调|强烈推荐|重点推荐', text):
        direction = 1
        p_level = "P1"
    if re.search(r'大幅下调|强烈看空', text):
        direction = -1
        p_level = "P1"
    
    # 目标价变动
    if re.search(r'目标价.*上调|上调.*目标价', text):
        direction = max(direction, 1)
    if re.search(r'目标价.*下调|下调.*目标价', text):
        direction = min(direction, -1)
    
    return p_level, direction

def main():
    print(f"=== 机构研报事件 [{TODAY}] ===")
    all_signals = []
    seen = set()
    
    for inst in INSTITUTIONS[:6]:  # 前6家重点机构
        for tmpl in SEARCH_TEMPLATES[:2]:  # 每种2个搜索模板
            query = tmpl(inst)
            news = mcp_news(query, size=5)
            if not news:
                continue
            for item in news:
                title = item.get("资讯标题", "")
                content = item.get("资讯内容", "")
                news_date = item.get("日期", TODAY)
                
                if not title or not content:
                    continue
                if title in seen:
                    continue
                seen.add(title)
                
                # 提取涉及股票
                codes = extract_stock_code(title + content)
                if not codes:
                    continue
                
                p_level, direction = classify_event(title, content, inst)
                if direction == 0:
                    continue
                
                dir_label = "bullish" if direction > 0 else "bearish"
                marker = "🟢" if direction > 0 else "🔴"
                
                for code in sorted(codes):
                    all_signals.append({
                        "code": code[:6], "name": "",
                        "event_type": f"research_{inst}_{dir_label}",
                        "p_level": p_level, "direction": dir_label,
                        "change_pct": 0,
                        "reason": f"{inst}: {title[:60]}"
                    })
                
                print(f"  {marker} {inst} → {', '.join(codes)} | {p_level} | {title[:50]}")
    
    if not all_signals:
        print("  ⚠ 无研报信号")
        return
    
    df = pd.DataFrame(all_signals)
    df["date"] = TODAY
    os.makedirs(EVENTS_DIR, exist_ok=True)
    
    path = os.path.join(EVENTS_DIR, f"research_events_{TODAY8}.parquet")
    df.to_parquet(path, index=False)
    
    clean = df.rename(columns={"code":"stock_code","event_type":"event_name","p_level":"p_level"})
    clean["dir_hard"] = clean["direction"].map({"bullish":1,"bearish":-1}).fillna(0)
    clean["impact"] = clean["dir_hard"] * clean["p_level"].map({"P0":10,"P1":5,"P2":2}).fillna(1)
    cp = os.path.join(EVENTS_DIR, f"research_clean_{TODAY8}.parquet")
    clean.to_parquet(cp, index=False)
    
    print(f"\n  ✓ {len(all_signals)}条研报信号 → {path}")

if __name__ == "__main__":
    main()
