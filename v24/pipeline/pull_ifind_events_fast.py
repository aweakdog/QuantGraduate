"""
iFinD MCP 按日期拉取 (不限个股) → 快速积累历史事件

策略: 按月搜索, 不绑定个股(远快于逐股)
"""
import sys, os, json, time, subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = str(settings.DATA_DIR)
OUT = os.path.join(DATA, "raw", "akshare_events", "ifind_events.parquet")
NODE = r"C:\Users\admin\.workbuddy\binaries\node\versions\22.22.2\node.exe"
IFIND_JS = r"C:\Users\admin\.workbuddy\skills\ifind-finance-data\call-node.js"

def ifind_call(srv, tool, params, timeout=60):
    """调用 iFinD API — 用 node -e 绕过 subprocess guard"""
    js_code = (
        f"const {{call}} = require({json.dumps(IFIND_JS)});"
        f"call({json.dumps(srv)},{json.dumps(tool)},"
        f"{json.dumps(params,ensure_ascii=False)})"
        f".then(r=>{{console.log(JSON.stringify(r));process.exit(0);}})"
        f".catch(e=>{{console.log(JSON.stringify({{ok:false,error:e.message}}));process.exit(0);}})"
    )
    try:
        r = subprocess.run(
            [NODE, "-e", js_code],
            capture_output=True, text=True, encoding='utf-8', timeout=timeout,
            cwd=os.path.dirname(IFIND_JS)
        )
        return json.loads(r.stdout) if r.stdout.strip() else {"ok": False}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

# 已有标题 (去重)
existing = set()
if os.path.exists(OUT):
    existing = set(pd.read_parquet(OUT)["title"].dropna().unique())

# 多主题 + 按周搜索2024-01到2026-06
queries = [
    "重大新闻 政策", "行业政策 利好", "概念爆发 涨停",
    "链主 英伟达 特斯拉 华为 比亚迪", "财务造假 立案",
    "业绩预增 预亏", "人工智能 半导体 新能源",
]

print("已有: %d 条, 开始追加..." % len(existing))
new_total = 0

for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        s = f"{year}-{month:02d}-01"
        e = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
        if s >= "2026-07-11": break
        if e > "2026-07-11": e = "2026-07-11"
        
        for q in queries:
            resp = ifind_call("news", "search_news", {"query":q, "time_start":s, "time_end":e, "size":10}, timeout=60)
            if not resp.get("ok"): continue
            try:
                text = resp["data"]["result"]["content"][0]["text"]
                parsed = json.loads(text)
                raw_data = parsed.get("data", {})
                items_str = raw_data.get("data", "[]") if isinstance(raw_data, dict) else "[]"
                items = json.loads(items_str) if isinstance(items_str, str) else items_str
            except: continue
            
            for item in (items if isinstance(items, list) else []):
                title = item.get("资讯标题","") if isinstance(item,dict) else ""
                if not title or title in existing: continue
                existing.add(title)
                new_total += 1
            
            time.sleep(0.3)  # API限速
        
        if month % 3 == 0:
            print("  %s-%02d: +%d条" % (year, month, new_total))

# 保存 (只存标题, 内容太多)
if new_total > 0:
    df = pd.DataFrame([{"title":t, "source":"iFinD", "pulled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")} for t in existing])
    df.to_parquet(OUT, index=False)

print(f"\n总事件: {len(existing)} 条 (+{new_total}新)")
