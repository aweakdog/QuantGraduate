"""
历史消息回填 — 机构研报/新闻 → events_clean

策略:
  MCP search_news 逐周查询各机构研报 → 分类 → 追加到 events_ifind

运行:
  python scripts/backfill_news_events.py

数据源: iFinD MCP search_news (api-mcp.51ifind.com)
"""
import json, os, sys, re, pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
EVENTS_CLEAN = os.path.join(DATA_DIR, "raw", "events_ifind", "events_clean.parquet")

MCP_TOKEN = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjtIFPAyXTksV4T-hKnzLqkW0Q1j-02SSnHyIzSMgGqD74Rj1lZto2oIAynV5gHiZXjTgX8ARvE1NtnBNLbdHMADuomjMNRHAXpPE83sCL4ehLGL6zb5_n8XVzLwr_RuJ4SZiekMR3sEMGNePywrP2flMO_K6R0suTvFTlSWU5WxOYMKqLxUOciZnZqTnxUs6_Lnj6He4XBEgul2VdJX4w6lcPq5ibDx7CDp-8SzW_FW0CBkREtIWBbyuqHaQyWdnUbg6nPoCo3sD3ipTL3ereUqX33GY8mn8dYfIFKZShADp5kGziTtqWRLQ.gQm_IZm7qxG8OKz-.mHYCbUproLUp1qLvMntUQ5rq6e27ORuzqnhXvhkIVFbA5UsTZBq_1UqJuq4XlN5EuI6j2o91dgWFz2vIHhm7482C1vcpwDTlUC48j_UymGR03dX8iiriSA-qE7ZQJLx50YFrG7aFw5sALibKzwDGVETilkI9upyDUu5s7tMg3cIhj0GUWU-8xso-AZf_frGahYyEzZsK4EHKHBxxVmE5IghBnJcTvjvB-Hs46nrhbeQ2wr2aSP82bq8JtXaHvstS6CC_63YS_jB7KWBF1sqkP25138A7y31xzOlmMEW_GIuDOElFXT2SXJ3qbGQmYBg_EwPMyGdy4rs2xk74WmQNyCzrdRIY86zPOlWqyt2EHJi9GHwxOjgAWdJ8eQ0o_kCsw7lyvYoWwQbyeqcs2rOTrhLeMbI.-AlkZXTtGcQkQqGT_JGRzw"
MCP_NEWS_URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-news-mcp"

# ─── 搜索配置 ────────────────────────────────────────────────
INSTITUTIONS = ["高盛", "摩根士丹利", "瑞银", "摩根大通", "花旗", "中金"]
WEEKS_BACK = 12  # 回填3个月

STOCK_CODE_PAT = re.compile(r'(6[0-9]{5}|3[0-9]{5}|0[0-9]{5}|002[0-9]{3}|688[0-9]{3}|300[0-9]{3})')

def mcp_news(query, size, start, end):
    body = json.dumps({
        'jsonrpc': '2.0', 'method': 'tools/call',
        'params': {'name': 'search_news', 'arguments': {
            'query': query, 'size': size,
            'time_start': start, 'time_end': end
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
    except: return []

def classify_event(title, content):
    text = f"{title} {content}"
    if re.search(r'(?:上调|买入|增持|超配|推荐|看好)', text):
        return ("research_upgrade", 1)
    if re.search(r'(?:下调|卖出|减持|低配|回避|沽售)', text):
        return ("research_downgrade", -1)
    if re.search(r'(?:业绩预增|净利润大增|大幅增长)', text):
        return ("earnings_surge", 1)
    if re.search(r'(?:业绩预亏|净利润为负)', text):
        return ("earnings_loss", -1)
    return ("research_note", 0)

def main():
    print("=== 历史消息回填 ===")
    all_rows = []
    seen_titles = set()
    total_api_calls = 0

    for inst in INSTITUTIONS:
        for week_offset in range(WEEKS_BACK):
            end_d = NOW - timedelta(weeks=week_offset)
            start_d = end_d - timedelta(weeks=1)
            start_s = start_d.strftime("%Y-%m-%d")
            end_s = end_d.strftime("%Y-%m-%d")

            query = f"{inst} 研报 评级 A股"
            items = mcp_news(query, 10, start_s, end_s)
            total_api_calls += 1

            if (week_offset + 1) % 4 == 0:
                print(f"  {inst}: 第{week_offset+1}/{WEEKS_BACK}周, {total_api_calls}次调用")

            for item in items:
                title = item.get("资讯标题", "")
                content = item.get("资讯内容", "")
                news_date = item.get("日期", "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                codes = set(STOCK_CODE_PAT.findall(title + content))
                if not codes:
                    continue

                ev_type, direction = classify_event(title, content)
                if direction == 0:
                    continue

                p_level = "P1" if abs(direction) == 1 else "P2"
                for code in codes:
                    all_rows.append({
                        "code": code, "title": title[:200],
                        "event_type": ev_type, "p_level": p_level,
                        "direction": direction, "date": news_date,
                        "source": f"backfill_{inst}"
                    })

    if not all_rows:
        print("⚠ 无数据")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).drop_duplicates(subset=["code", "title"])

    # 合并到 events_clean
    if os.path.exists(EVENTS_CLEAN):
        old = pd.read_parquet(EVENTS_CLEAN)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["code", "title"], keep="first")
        merged = merged.sort_values(["code", "date"]).reset_index(drop=True)
    else:
        merged = df

    merged.to_parquet(EVENTS_CLEAN)
    print(f"\n✓ 完成: {len(df)}条新事件, 合并后共{len(merged)}条")
    print(f"  机构: {df['source'].value_counts().to_dict()}")
    print(f"  类型: {df['event_type'].value_counts().to_dict()}")
    print(f"  共{total_api_calls}次API调用")

if __name__ == "__main__":
    main()
