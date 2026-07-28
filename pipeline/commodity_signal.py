"""
跨资产信号模块 — 商品价格变动 → A股事件信号

流程:
  1. 从 KG 读当前 commodity 价格
  2. 对比 N 日前的价格，计算变动幅度
  3. 超过阈值(±3%) → 通过 company_commodities 查受影响股票
  4. 生成 P0/P1 事件信号，写入 events_clean 格式
  5. 同时写入 P0-P4 events 供特征工程使用

运行:
  C:/Users/admin/.workbuddy/binaries/python/envs/ths/Scripts/python.exe commodity_signal.py [--dry-run]

自动化: 交易日17:35（商品行情更新后5分钟）
"""
import json, os, sys, sqlite3, pandas as pd
from datetime import datetime, timezone, timedelta

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY8 = NOW.strftime("%Y%m%d")
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
KG_DB = os.path.expanduser("~/AppData/Local/hermes/skills/miaoxiong/knowledge-graph/trade_knowledge.db")
EVENTS_DIR = os.path.join(DATA_DIR, "raw", "events_daily")

# ─── 商品变动阈值配置 ─────────────────────────────────────────
# (commodity_name, threshold_pct, event_type, p_level, bull/bear)
SIGNAL_RULES = [
    # macro 商品
    ("黄金",     3.0, "gold_surge",     "P0",  "bullish"),
    ("黄金",    -3.0, "gold_crash",     "P1",  "bearish"),
    ("原油",     3.0, "oil_surge",      "P1",  "bullish"),
    ("原油",    -3.0, "oil_crash",      "P0",  "bullish"),  # 油价跌利好消费端
    ("铜",       3.0, "copper_surge",   "P1",  "bullish"),
    ("铜",      -3.0, "copper_crash",   "P1",  "bearish"),
    # company 产品
    ("六氟化钨",   5.0, "wf6_surge",     "P0",  "bullish"),
    ("六氟化钨",  -5.0, "wf6_crash",     "P1",  "bearish"),
    ("六氟磷酸锂", 5.0, "lipf6_surge",    "P2",  "bullish"),
    ("六氟磷酸锂",-5.0, "lipf6_crash",    "P1",  "bearish"),
    ("纯碱",     5.0, "soda_surge",     "P2",  "bullish"),
    ("纯碱",    -5.0, "soda_crash",     "P2",  "bearish"),
]

# ─── 商品→stock 映射（从KG加载）───────────────────────────────

def load_kg_mapping():
    """从知识图谱加载 commodity_name → [affected_stocks] 映射"""
    conn = sqlite3.connect(KG_DB)
    conn.row_factory = sqlite3.Row
    
    # 查所有 company_commodities 关联
    rows = conn.execute("""
        SELECT cc.*, s.name as stock_name, s.code
        FROM company_commodities cc
        JOIN stock_picks s ON cc.stock_pick_id=s.id
    """).fetchall()
    conn.close()
    
    mapping = {}
    for r in rows:
        d = dict(r)
        cname = d["commodity_name"]
        if cname not in mapping:
            mapping[cname] = []
        mapping[cname].append({
            "code": d["code"],
            "name": d["stock_name"],
            "relation": d["relation"],
            "note": d["note"]
        })
    return mapping

def get_latest_prices(days=30):
    """从KG commodities 表获取最近N日商品价格"""
    conn = sqlite3.connect(KG_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT date, name, price, change_pct, direction, product_type
           FROM commodities WHERE date >= DATE('now', ? || ' days')
           ORDER BY name, date""",
        (f'-{days}',)).fetchall()
    conn.close()
    
    prices = {}
    for r in rows:
        d = dict(r)
        n = d["name"]
        if n not in prices:
            prices[n] = []
        prices[n].append(d)
    return prices

def detect_signals(prices, mapping):
    """检测商品变动触发的事件信号
    
    Returns:
        [(code, event_type, p_level, direction, reason), ...]
    """
    signals = []
    
    for cname, history in prices.items():
        if len(history) < 1:
            continue
        
        # 优先用 change_pct（直接计算好的涨跌幅）
        latest = history[-1]
        chg = latest.get("change_pct", 0)
        
        # 没有 change_pct 但有2天历史 → 手动算
        if chg == 0 and len(history) >= 2:
            prev = history[-2]
            if latest["price"] and prev["price"]:
                chg = ((latest["price"] - prev["price"]) / prev["price"]) * 100
        
        # 检查是否触发规则
        for rule_name, threshold, ev_type, p_level, direction in SIGNAL_RULES:
            if rule_name != cname:
                continue
            
            triggered = (chg >= threshold) if threshold > 0 else (chg <= threshold)
            if not triggered:
                continue
            
            # 找受影响的股票
            affected = mapping.get(cname, [])
            if not affected:
                # 不知道谁受影响 → 按 affected_sectors 生成泛信号
                signals.append({
                    "code": "__ALL__",
                    "name": cname,
                    "event_type": ev_type,
                    "p_level": p_level,
                    "direction": direction,
                    "change_pct": round(chg, 2),
                    "reason": f"{cname} {chg:+.2f}% 触发{ev_type}"
                })
                continue
            
            for stock in affected:
                rel = stock["relation"]
                # 根据关系调整信号方向
                # produce: 涨价利好，跌价利空
                # consume: 涨价利空(成本升)，跌价利好(成本降)
                sig_dir = direction
                if rel == "produce":
                    # 生产端：价格涨利好，价格跌利空 → 翻转
                    sig_dir = "bearish" if direction == "bullish" and "crash" in ev_type else sig_dir
                elif rel == "consume":
                    # 消费端：价格跌利好，价格涨利空
                    sig_dir = "bearish" if direction == "bullish" and "surge" in ev_type else sig_dir
                
                signals.append({
                    "code": stock["code"],
                    "name": stock["name"],
                    "event_type": ev_type,
                    "p_level": p_level,
                    "direction": sig_dir,
                    "change_pct": round(chg, 2),
                    "relation": rel,
                    "reason": f"{cname} {chg:+.2f}% → {stock['name']}({rel})"
                })
    
    return signals

def write_signals(signals):
    """写入 events_daily/{TODAY8}.parquet + events_clean 格式"""
    if not signals:
        print("  ⚠ 无触发信号")
        return
    
    df = pd.DataFrame(signals)
    df["date"] = TODAY
    
    # 保存当日信号
    os.makedirs(EVENTS_DIR, exist_ok=True)
    path = os.path.join(EVENTS_DIR, f"commodity_signals_{TODAY8}.parquet")
    df.to_parquet(path, index=False)
    print(f"  ✓ 信号已保存: {len(signals)}条 → {path}")
    
    # 转换到 events_clean 格式（供 feature_engine 直接使用）
    clean = df.rename(columns={
        "code": "stock_code",
        "event_type": "event_name",
        "p_level": "p_level",
    })
    clean["direction"] = clean["direction"].map({"bullish": 1, "bearish": -1}).fillna(0)
    clean["impact"] = clean["direction"] * clean["p_level"].map({"P0": 10, "P1": 5, "P2": 2}).fillna(1)
    
    clean_path = os.path.join(DATA_DIR, "raw", "events_daily", f"commodity_clean_{TODAY8}.parquet")
    clean.to_parquet(clean_path, index=False)
    print(f"  ✓ events_clean 格式: → {clean_path}")
    
    return df

def main(dry_run=False):
    print(f"=== 跨资产信号扫描 [{TODAY}] ===")
    
    # 1. 加载 KG 映射
    mapping = load_kg_mapping()
    print(f"  KG映射: {len(mapping)} 种商品关联了公司")
    
    # 2. 获取最新价格
    prices = get_latest_prices(days=30)
    print(f"  商品价格: {sum(len(v) for v in prices.values())} 条记录, {len(prices)} 种商品")
    
    # 3. 检测信号
    signals = detect_signals(prices, mapping)
    
    if not signals:
        print("  ✓ 今日无触发信号")
        return
    
    print(f"\n  ⚡ 触发 {len(signals)} 条信号:")
    for s in signals:
        marker = "🟢" if s["direction"] == "bullish" else "🔴"
        print(f"    {marker} {s['code']} {s['name']} | {s['event_type']} ({s['p_level']}) | {s['reason']}")
    
    if not dry_run:
        write_signals(signals)
    
    return signals

if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
