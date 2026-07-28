"""
AKShare 历史事件/新闻拉取 — 关注圈 196 只股票

用法: python pipeline/pull_akshare_events.py

输出: data/raw/akshare_events/{code}.parquet
去重: 同股+同日+同标题 → 只保留一条
"""

import akshare as ak
import pandas as pd
import numpy as np
import os, time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import settings

DATA = str(settings.DATA_DIR)
OUT_DIR = os.path.join(DATA, "raw", "akshare_events")
WATCHLIST_PATH = str(settings.WATCHLIST_PATH)
os.makedirs(OUT_DIR, exist_ok=True)

# 加载关注圈
with open(WATCHLIST_PATH, encoding="utf-8") as f:
    watch = json.load(f)
stocks = watch.get("watchlist", [])
print(f"关注圈: {len(stocks)} 只")
print(f"输出目录: {OUT_DIR}")

total_new = 0
total_dedup = 0
errors = 0

for i, s in enumerate(stocks):
    code = s["code"]
    name = s["name"]
    code6 = code[:6]
    out_path = os.path.join(OUT_DIR, f"{code6}.parquet")
    
    # 加载已有数据(用于去重)
    existing = set()
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        # 去重 key: 标题
        existing = set(old["title"].dropna().unique())
    
    try:
        # 拉取该股的历史新闻 (东方财富源, 最多约100条)
        df = ak.stock_news_em(symbol=code6)
        
        if df is None or len(df) == 0:
            continue
        
        # 标准化列名
        df = df.rename(columns={
            "新闻标题": "title",
            "新闻内容": "content", 
            "发布时间": "pub_time",
            "文章来源": "source",
            "新闻链接": "url",
        }, errors="ignore")  # akshare v1.18 列名格式有微小差异
        
        # 去重: 同标题只保留一条
        df["title"] = df.get("title", "").astype(str)
        df = df[~df["title"].isin(existing)]
        dedup_count = len(df)
        
        if len(df) == 0:
            continue
        
        # 添加元信息
        df["code"] = code
        df["name"] = name
        df["pulled_at"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # 如果已有文件, 合并; 否则直接保存
        if os.path.exists(out_path):
            old = pd.read_parquet(out_path)
            df = pd.concat([old, df], ignore_index=True)
        
        df.to_parquet(out_path, index=False)
        total_new += dedup_count
        
        if (i + 1) % 20 == 0 or i == len(stocks) - 1:
            print(f"  [{i+1}/{len(stocks)}] {name}({code}): +{dedup_count}条 (累计{total_new})")
        
        time.sleep(0.3)  # 礼貌限速
        
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  [!] {name}({code}): {e}")

print(f"\n完成: 新增 {total_new} 条事件, {errors} 个错误")

# 去重检查
total_all = 0
from collections import Counter
for f in os.listdir(OUT_DIR):
    if f.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(OUT_DIR, f))
        total_all += len(df)

print(f"总事件数: {total_all}")
