"""重建训练数据前的前置检查: watchlist / read_kline schema / 关键数据源可读性"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from pipeline.config import settings
from pipeline.feature_engine import read_kline

print("=== 1. 路径解析 ===")
print(f"  settings.DATA_DIR = {settings.DATA_DIR}")
print(f"  存在: {Path(settings.DATA_DIR).exists()}")

print("\n=== 2. build_all 使用的 watchlist ===")
for nm in ["watchlist_top120.json", "watchlist.json", "watchlist_216.json"]:
    p = ROOT / "data" / "universe" / nm
    if p.exists():
        w = json.loads(p.read_text())
        items = w.get("watchlist", w) if isinstance(w, dict) else w
        print(f"  {nm:24s} 存在, {len(items)} 只  <- build_all 会优先用前者")
    else:
        print(f"  {nm:24s} 缺失")

print("\n=== 3. read_kline 是否兼容新K线 schema ===")
raw = pd.read_parquet(ROOT / "data/raw/kline/000063.parquet")
print(f"  原始列: {list(raw.columns)}")
dk = read_kline("000063")
if dk is None:
    print("  [!!] read_kline 返回 None — 不兼容!")
else:
    print(f"  read_kline OK: {len(dk)} 行, 列 = {list(dk.columns)}")
    print(f"  日期范围 {pd.to_datetime(dk['date']).min().date()} ~ "
          f"{pd.to_datetime(dk['date']).max().date()}")
    need = ["date", "open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in dk.columns]
    print(f"  缺失必需列: {miss if miss else '无'}")

print("\n=== 4. 现有 v24 的规模 (作为重建目标参照) ===")
v24 = pd.read_parquet(ROOT / "data/processed/training_data_v24.parquet",
                      columns=["date", "code"])
v24["date"] = pd.to_datetime(v24["date"])
print(f"  {len(v24):,} 行, {v24['code'].nunique()} 只, "
      f"{v24['date'].min().date()} ~ {v24['date'].max().date()}")

print("\n=== 5. 关键依赖数据源可读性 ===")
checks = [
    ("资金流合并档", "data/raw/fund_flow_full/fundflow_history.parquet"),
    ("事件 events_v2", "data/raw/events_ifind/events_v2.parquet"),
    ("概念映射", "data/universe/concept_stock_map.json"),
    ("供应链图谱", "data/universe/supply_chain_map.json"),
    ("中国PMI", "data/raw/macro/中国PMI.parquet"),
    ("美国ISM制造业PMI", "data/raw/macro/美国ISM制造业PMI.parquet"),
]
for label, rel in checks:
    p = ROOT / rel
    if not p.exists():
        print(f"  {label:20s} 缺失  ({rel})")
        continue
    try:
        if p.suffix == ".json":
            n = len(json.loads(p.read_text()))
            print(f"  {label:20s} OK  {n} 项")
        else:
            d = pd.read_parquet(p)
            dc = next((c for c in d.columns
                       if "date" in str(c).lower() or "日期" in str(c) or "月份" in str(c)), None)
            mx = ""
            if dc:
                s = pd.to_datetime(d[dc], errors="coerce")
                if s.notna().any():
                    mx = f", 最新 {s.max().date()}"
            print(f"  {label:20s} OK  {len(d):,} 行{mx}")
    except Exception as e:
        print(f"  {label:20s} 读取失败 {type(e).__name__}")

print("\n=== 6. features 缓存目录 (增量跳过依据) ===")
fd = ROOT / "data" / "processed" / "features"
if fd.exists():
    fs = list(fd.glob("*.parquet"))
    print(f"  {len(fs)} 个特征文件 — 全量重建需 --no-incremental 强制覆盖")
else:
    print("  不存在, 将全新构建")
