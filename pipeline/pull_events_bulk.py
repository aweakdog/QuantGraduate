"""
批量拉取事件数据 — 多渠道并行

来源:
  1. AKShare stock_news_em — 每只最近10条
  2. AKShare 百度经济新闻 — 历史日期的宏观动态
  3. 10jqka push API — 近10天快讯 (股票级)

输出: data/raw/akshare_events/{code6}.parquet (增量追加, 自动去重)
"""
import akshare as ak
import pandas as pd
import numpy as np
import os, json, time, sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = str(settings.DATA_DIR)
OUT_DIR = os.path.join(DATA, "raw", "akshare_events")
os.makedirs(OUT_DIR, exist_ok=True)

with open(str(settings.WATCHLIST_PATH), encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]

print(f"关注圈: {len(stocks)} 只")

# ═════════════════════════════════════════
# 方法1: 逐股拉新闻
# ═════════════════════════════════════════
def pull_stock_news(code6, name):
    """拉取单只股票的新闻, 去重后保存"""
    out_path = os.path.join(OUT_DIR, f"{code6}.parquet")
    
    # 已有数据去重key
    existing_titles = set()
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        existing_titles = set(old["title"].dropna().unique())
    
    try:
        df = ak.stock_news_em(symbol=code6)
        if df is None or len(df) == 0:
            return 0
        
        # 标准化
        df["title"] = df["新闻标题"].astype(str)
        df["content"] = df.get("新闻内容", "")
        df["pub_time"] = pd.to_datetime(df["发布时间"])
        df["source"] = df.get("文章来源", "东财")
        df["url"] = df.get("新闻链接", "")
        df["keywords"] = df.get("关键词", "")
        df["code"] = f"{code6}.SH" if int(code6[0]) >= 6 else f"{code6}.SZ"
        df["name"] = name
        df["pulled_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # 去重
        df = df[~df["title"].isin(existing_titles)]
        if len(df) == 0:
            return 0
        
        keep_cols = ["title","content","pub_time","source","url","keywords","code","name","pulled_at"]
        df = df[keep_cols]
        
        if os.path.exists(out_path):
            old = pd.read_parquet(out_path)
            df = pd.concat([old, df], ignore_index=True)
        
        df.to_parquet(out_path, index=False)
        return len(df) - (len(old) if os.path.exists(out_path) else 0)
    except Exception as e:
        return 0

print("\n=== 逐股拉取 AKShare 新闻 ===")
total_new = 0
for i, s in enumerate(stocks):
    n = pull_stock_news(s["code"][:6], s["name"])
    total_new += n
    if (i+1) % 30 == 0:
        print(f"  [{i+1}/{len(stocks)}] 累计 +{total_new}条")
    time.sleep(0.25)
print(f"Stock news 完成: +{total_new}条")

# ═════════════════════════════════════════
# 方法2: 百度经济新闻 (历史日期, 宏观事件)
# ═════════════════════════════════════════
print("\n=== 百度经济新闻 (近30天) ===")
macro_all = []
for d in range(30, -1, -1):
    dt = (datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
    try:
        df = ak.news_economic_baidu(date=dt)
        if df is not None and len(df) > 0:
            df["pub_date"] = dt
            macro_all.append(df)
    except:
        pass
    if d % 5 == 0:
        time.sleep(1)

if macro_all:
    macro_df = pd.concat(macro_all, ignore_index=True)
    macro_df = macro_df.rename(columns={"title":"title","content":"content"})
    macro_df["source"] = "百度经济"
    macro_df["code"] = "000000"
    macro_df["name"] = "宏观经济"
    macro_path = os.path.join(OUT_DIR, "macro_baidu.parquet")
    if os.path.exists(macro_path):
        old = pd.read_parquet(macro_path)
        macro_df = pd.concat([old, macro_df], ignore_index=True)
        macro_df = macro_df.drop_duplicates(subset=["title"])
    macro_df.to_parquet(macro_path, index=False)
    print(f"宏观经济事件: {len(macro_df)}条 (累计)")

# ═════════════════════════════════════════
# 方法3: 10jqka 快讯 (近10天, 股票级)
# ═════════════════════════════════════════
print("\n=== 同花顺快讯 (近10天) ===")
import subprocess

API = 'https://news.10jqka.com.cn/tapp/news/push/stock'
push_count = 0

for day_offset in range(10):
    dt = (datetime.now() - timedelta(days=day_offset)).strftime("%Y%m%d")
    for page in range(1, 11):  # 每天最多10页
        cmd = f'curl -s -H "User-Agent: Mozilla/5.0" -H "Referer: https://news.10jqka.com.cn/" "{API}?date={dt}&page={page}"'
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=True)
            data = json.loads(r.stdout).get('data', {}).get('list', [])
            for item in data:
                title = item.get('title', '')
                digest = item.get('digest', '')
                ctime = item.get('ctime', 0)
                tags = item.get('tags', '')
                stock_names = item.get('stock', [])
                
                # 匹配关注圈股票
                for sn in (stock_names if isinstance(stock_names, list) else []):
                    matched = [s for s in stocks if s["name"] in str(sn)]
                    for ms in matched:
                        out_path = os.path.join(OUT_DIR, f"{ms['code'][:6]}.parquet")
                        existing_titles = set()
                        if os.path.exists(out_path):
                            existing_titles = set(pd.read_parquet(out_path)["title"].unique())
                        if title in existing_titles:
                            continue
                        
                        row = pd.DataFrame([{
                            "title": title, "content": digest,
                            "pub_time": datetime.fromtimestamp(int(ctime)) if ctime else None,
                            "source": "同花顺快讯", "url": "",
                            "keywords": tags, "code": ms["code"],
                            "name": ms["name"],
                            "pulled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        }])
                        
                        if os.path.exists(out_path):
                            row = pd.concat([pd.read_parquet(out_path), row], ignore_index=True)
                        row.to_parquet(out_path, index=False)
                        push_count += 1
        except:
            pass
        time.sleep(0.5)

print(f"快讯匹配: +{push_count}条")

# 汇总
total_all = 0
for f in os.listdir(OUT_DIR):
    if f.endswith(".parquet"):
        total_all += len(pd.read_parquet(os.path.join(OUT_DIR, f)))
print(f"\n=== 总计: {total_all} 条事件 ===")
