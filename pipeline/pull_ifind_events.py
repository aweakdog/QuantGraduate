"""
iFinD MCP 批量拉取历史事件 — 关注圈 198 只股票

用法: python pipeline/pull_ifind_events.py

策略: 逐股拉取, 时间分片 (每月一批), 自动去重保存
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = str(settings.DATA_DIR)
OUT_DIR = os.path.join(DATA, "raw", "akshare_events")
os.makedirs(OUT_DIR, exist_ok=True)

# iFinD call setup
NODE = r"C:\Users\admin\.workbuddy\binaries\node\versions\22.22.2\node.exe"
IFIND_JS = r"C:\Users\admin\.workbuddy\skills\ifind-finance-data\call-node.js"

def ifind_call(server, tool, params, timeout=30):
    """调用 iFinD API"""
    payload = json.dumps({
        "action": "call",
        "server_type": server,
        "tool_name": tool,
        "params": params,
    }, ensure_ascii=False)
    try:
        r = subprocess.run(
            [NODE, IFIND_JS, payload],
            capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(IFIND_JS)
        )
        return json.loads(r.stdout) if r.stdout else {"ok": False, "error": r.stderr[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

# 加载关注圈
with open(str(settings.WATCHLIST_PATH), encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]

print(f"关注圈: {len(stocks)} 只")

# 时间分片: 2024-01 到 2026-06, 每月一批
# 避免单次查询太慢
date_ranges = []
for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year+1}-01-01"
        else:
            end = f"{year}-{month+1:02d}-01"
        if end > "2026-07-01":
            end = "2026-07-01"
        if start >= "2026-07-01":
            continue
        date_ranges.append((start, end))

print(f"时间分片: {len(date_ranges)} 个批次")

# 逐股 + 逐月拉取
total_new = 0
errors = 0

for i, s in enumerate(stocks):
    code = s["code"]
    code6 = code[:6]
    name = s["name"]
    out_path = os.path.join(OUT_DIR, f"{code6}.parquet")
    
    existing_titles = set()
    if os.path.exists(out_path):
        existing_titles = set(pd.read_parquet(out_path)["title"].dropna().unique())
    
    for start, end in date_ranges:
        # 如果这个股票已经有这个时间段的数据, 跳过
        # (简化判断: 只要总条数超过200就跳过→避免无限拉)
        if len(existing_titles) > 150:
            break
        
        resp = ifind_call("news", "search_news", {
            "query": f"{name} 重大事件",
            "time_start": start,
            "time_end": end,
            "size": 10,
        }, timeout=60)
        
        if not resp.get("ok"):
            continue
        
        try:
            data = resp.get("data", {})
            content = data.get("result", {}).get("content", [])
            if not content:
                continue
            text = content[0].get("text", "")
            parsed = json.loads(text)
            items = parsed.get("data", {}).get("data", []) if isinstance(parsed.get("data"), dict) else parsed.get("data", [])
        except:
            continue
        
        if not items or not isinstance(items, list):
            continue
        
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("资讯标题", "")
            if not title or title in existing_titles:
                continue
            
            existing_titles.add(title)
            row = pd.DataFrame([{
                "title": title,
                "content": item.get("资讯内容", "")[:500],
                "pub_time": pd.Timestamp.now(),  # iFinD doesn't return pub_time
                "source": "iFinD",
                "url": "",
                "keywords": "",
                "code": code,
                "name": name,
                "pulled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "date_range": f"{start}~{end}",
            }])
            
            if os.path.exists(out_path):
                old = pd.read_parquet(out_path)
                row = pd.concat([old, row], ignore_index=True)
            row.to_parquet(out_path, index=False)
            total_new += 1
        
        time.sleep(0.3)  # API限速
    
    if (i+1) % 20 == 0 or i == len(stocks) - 1:
        print(f"  [{i+1}/{len(stocks)}] {name}: 累计 +{total_new}条")

print(f"\n完成: +{total_new} 条 iFinD 事件")

# 汇总
total_all = 0
for f in os.listdir(OUT_DIR):
    if f.endswith(".parquet"):
        total_all += len(pd.read_parquet(os.path.join(OUT_DIR, f)))
print(f"总事件数: {total_all}")
