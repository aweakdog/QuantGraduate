"""
链主监控管道 — 实时追踪链主事件 → 映射到A股受影响标的

工作流:
1. 每日检查链主关键日期(财报/产品发布)
2. 扫描链主相关新闻(7x24快讯/百度热搜)
3. 事件匹配 → 供需映射表 → 受影响A股标的
4. 输出监控告警(JSON格式, 供六维评分引擎消费)

使用: python chain_leader_monitor.py [--date YYYY-MM-DD]
"""
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("chain_leader_monitor")

SUPPLY_CHAIN_PATH = str(settings.SUPPLY_CHAIN_PATH)
WATCHLIST_PATH = str(settings.WATCHLIST_PATH)
ALERTS_PATH = str(settings.PROCESSED_DIR / "chain_leader_alerts.json")

# ─── 数据加载 ───────────────────────────────────────────────

def load_supply_chain() -> dict:
    with open(SUPPLY_CHAIN_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_watchlist() -> dict:
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        return json.load(f)

def save_alerts(alerts: list[dict]) -> None:
    os.makedirs(os.path.dirname(ALERTS_PATH), exist_ok=True)
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "alerts": alerts},
                  f, ensure_ascii=False, indent=2)
    log.info(f"  ✅ Alerts saved: {ALERTS_PATH} ({len(alerts)} alerts)")

# ─── 关键日期检查 ─────────────────────────────────────────────

def check_key_dates(chains: list[dict], today: str) -> list[dict]:
    """检查未来7天内有无链主关键事件"""
    alerts = []
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    window_end = today_dt + timedelta(days=7)

    for c in chains:
        cl = c.get("chain_leader", {})
        next_date = cl.get("next_earnings")
        if not next_date:
            continue
        try:
            nd = datetime.strptime(next_date, "%Y-%m-%d")
        except ValueError:
            continue

        days_until = (nd - today_dt).days

        if days_until < 0:
            continue  # 已过期
        if days_until <= 7:
            # 即将发生或当天发生
            alert = {
                "type": "upcoming_event",
                "severity": "high" if days_until <= 3 else "medium",
                "chain_leader": cl["name"],
                "event": cl.get("watch_events", ["earnings"])[0],
                "date": next_date,
                "days_until": days_until,
                "theme": c["theme"],
                "affected_stocks": _get_all_supplier_codes(c),
                "message": f"{cl['name']} 财报/事件在 {days_until} 天后 ({next_date})"
            }
            alerts.append(alert)

    return alerts

# ─── 新闻事件匹配 ──────────────────────────────────────────────

def match_news_event(title: str, chains: list[dict]) -> list[dict]:
    """根据新闻标题匹配链主, 返回受影响标的"""
    alerts = []
    title_lower = title.lower()

    for c in chains:
        cl = c.get("chain_leader", {})
        name = cl.get("name", "").lower()
        keywords = cl.get("watch_events", [])

        # 链主名称匹配
        if name not in title_lower:
            continue

        # 判断事件类型
        event_type = "general"
        if any(kw in title_lower for kw in ["财报", "earnings", "营收", "净利"]):
            event_type = "earnings"
        elif any(kw in title_lower for kw in ["新品", "发布", "发布", "launch"]):
            event_type = "product_launch"
        elif any(kw in title_lower for kw in ["制裁", "禁令", "出口管制", "sanctions"]):
            event_type = "sanctions"
        elif any(kw in title_lower for kw in ["订单", "合同", "合作", "供货"]):
            event_type = "supply_chain"

        alert = {
            "type": "news_event",
            "severity": "high" if event_type in ("earnings", "sanctions") else "medium",
            "chain_leader": cl["name"],
            "event": event_type,
            "source_title": title[:100],
            "theme": c["theme"],
            "affected_stocks": _get_all_supplier_codes(c),
            "message": f"链主 {cl['name']} 出现{event_type}事件: {title[:60]}"
        }
        alerts.append(alert)

    return alerts

def _get_all_supplier_codes(chain: dict) -> list[dict]:
    """从供需链中提取所有A股供应商"""
    stocks = {}
    for link in chain.get("demand_links", []):
        for s in link.get("a_share_suppliers", []):
            code = s["code"]
            if code not in stocks:
                stocks[code] = {
                    "code": code,
                    "name": s["name"],
                    "role": s["role"],
                    "exposure": s["exposure"]
                }
    return list(stocks.values())

# ─── 7x24新闻快速扫描 ─────────────────────────────────────────

def scan_7x24_news(chains: list[dict], pages: int = 5) -> list[dict]:
    """
    扫描同花顺7x24快讯, 匹配链主事件.
    返回告警列表.
    """
    import requests
    import time

    alerts = []
    leader_names = set()
    for c in chains:
        cl = c.get("chain_leader", {})
        leader_names.add(cl.get("name", "").lower())

    for page in range(1, pages + 1):
        url = "https://news.10jqka.com.cn/tapp/news/push/stock/?page={}&tag=A%E8%82%A1&track=website&pagesize=30".format(page)
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            for item in data.get("data", []):
                title = item.get("title", "") + " " + item.get("digest", "")
                matched = match_news_event(title, chains)
                alerts.extend(matched)
            time.sleep(0.5)
        except Exception as e:
            log.info(f"  ⚠ 7x24 page {page} error: {e}")

    return alerts

# ─── 主流程 ───────────────────────────────────────────────────

def main():
    # 支持 --date YYYY-MM-DD 或 直接传日期
    if len(sys.argv) > 1:
        if sys.argv[1] == "--date" and len(sys.argv) > 2:
            today = sys.argv[2]
        else:
            today = sys.argv[1]
    else:
        today = datetime.now().strftime("%Y-%m-%d")
    log.info(f"[MONITOR] 链主监控 [{today}]")

    sc = load_supply_chain()
    chains = sc.get("chains", [])
    log.info(f"  已加载 {len(chains)} 条供需链")

    all_alerts = []

    # 1. 检查关键日期
    log.info(f"  → 检查关键日期(未来7天)...")
    date_alerts = check_key_dates(chains, today)
    all_alerts.extend(date_alerts)
    for a in date_alerts:
        log.info(f"    [{'HIGH' if a['severity']=='high' else 'MED'}] {a['message']}")

    # 2. 扫描7x24新闻
    log.info(f"  → 扫描7x24快讯(5页)...")
    news_alerts = scan_7x24_news(chains, pages=5)
    all_alerts.extend(news_alerts)
    for a in news_alerts:
        log.info(f"    [{'HIGH' if a['severity']=='high' else 'MED'}] {a['message']}")

    # 3. 去重 (相同链主+事件类型只保留最新)
    seen = set()
    deduped = []
    for a in all_alerts:
        key = (a["chain_leader"], a["event"])
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    # 4. 保存
    if deduped:
        save_alerts(deduped)
        log.info(f"\n[ALERTS] 共 {len(deduped)} 条去重告警")
    else:
        log.info(f"\n✅ 无重大链主事件触发")

if __name__ == "__main__":
    main()
