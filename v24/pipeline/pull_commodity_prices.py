"""
每日商品/产品行情拉取

策略:
  macro商品: 通过 iFinD MCP 接口直拉沪金/沪铜等期货行情
  company产品: 记录最新基准价格 → 写入KG

数据源:
  - iFinD MCP: api-mcp.51ifind.com (配在 mcp.json 的 token)
  - company产品: 已知市场价+生意社确认

运行:
  python pull_commodity_prices.py [all|macro|sync|status|update_price]

自动化: 交易日17:30
"""
import json, os, sys, sqlite3, pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
TODAY = NOW.strftime("%Y-%m-%d")
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
KG_DB = os.path.expanduser("~/AppData/Local/hermes/skills/miaoxiong/knowledge-graph/trade_knowledge.db")
MACRO_DIR = os.path.join(DATA_DIR, "raw", "macro")

# ─── iFinD MCP 配置 ──────────────────────────────────────────
MCP_TOKEN = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjtIFPAyXTksV4T-hKnzLqkW0Q1j-02SSnHyIzSMgGqD74Rj1lZto2oIAynV5gHiZXjTgX8ARvE1NtnBNLbdHMADuomjMNRHAXpPE83sCL4ehLGL6zb5_n8XVzLwr_RuJ4SZiekMR3sEMGNePywrP2flMO_K6R0suTvFTlSWU5WxOYMKqLxUOciZnZqTnxUs6_Lnj6He4XBEgul2VdJX4w6lcPq5ibDx7CDp-8SzW_FW0CBkREtIWBbyuqHaQyWdnUbg6nPoCo3sD3ipTL3ereUqX33GY8mn8dYfIFKZShADp5kGziTtqWRLQ.gQm_IZm7qxG8OKz-.mHYCbUproLUp1qLvMntUQ5rq6e27ORuzqnhXvhkIVFbA5UsTZBq_1UqJuq4XlN5EuI6j2o91dgWFz2vIHhm7482C1vcpwDTlUC48j_UymGR03dX8iiriSA-qE7ZQJLx50YFrG7aFw5sALibKzwDGVETilkI9upyDUu5s7tMg3cIhj0GUWU-8xso-AZf_frGahYyEzZsK4EHKHBxxVmE5IghBnJcTvjvB-Hs46nrhbeQ2wr2aSP82bq8JtXaHvstS6CC_63YS_jB7KWBF1sqkP25138A7y31xzOlmMEW_GIuDOElFXT2SXJ3qbGQmYBg_EwPMyGdy4rs2xk74WmQNyCzrdRIY86zPOlWqyt2EHJi9GHwxOjgAWdJ8eQ0o_kCsw7lyvYoWwQbyeqcs2rOTrhLeMbI.-AlkZXTtGcQkQqGT_JGRzw"
MCP_INDEX_URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-index-mcp"

# ─── 期货查询清单 ────────────────────────────────────────────
FUTURES = [
    ("黄金", "沪金主连AUZL.SHF的最新收盘价、涨跌幅", "元/克"),
    ("白银", "沪银主连AGZL.SHF的最新收盘价、涨跌幅", "元/千克"),
    ("铜", "沪铜主连CUZL.SHF的最新收盘价、涨跌幅", "元/吨"),
    ("铝", "沪铝主连ALZL.SHF的最新收盘价、涨跌幅", "元/吨"),
    ("锌", "沪锌主连ZNZL.SHF的最新收盘价、涨跌幅", "元/吨"),
    ("镍", "沪镍主连NIZL.SHF的最新收盘价、涨跌幅", "元/吨"),
    ("锡", "沪锡主连SNZL.SHF的最新收盘价、涨跌幅", "元/吨"),
]

# ─── 公司产品基准价 ──────────────────────────────────────────
COMPANY_BENCHMARKS = {
    "六氟化钨": {"price": 1750, "unit": "元/kg",  "range": "1670-2500",
                 "source": "百川盈孚/买化塑", "date": "2026-06-09",
                 "note": "5N级99.999%, 同比+232%"},
    "六氟磷酸锂": {"price": 58000, "unit": "元/吨", "range": "50000-70000",
                   "source": "百川盈孚", "date": "2026-06-30",
                   "note": "底部震荡, 产能出清中"},
    "EVA光伏料": {"price": 12000, "unit": "元/吨", "range": "11000-14000",
                  "source": "百川盈孚", "date": "2026-07-01", "note": "底部震荡"},
    "磷矿石": {"price": 1050, "unit": "元/吨", "range": "950-1150",
               "source": "百川盈孚", "date": "2026-07-01", "note": "价格高位"},
    "纯碱": {"price": 1800, "unit": "元/吨", "range": "1600-2000",
             "source": "百川盈孚", "date": "2026-07-01", "note": "下跌通道"},
}

def mcp_query(query):
    body = json.dumps({
        'jsonrpc': '2.0', 'method': 'tools/call',
        'params': {'name': 'index_data', 'arguments': {'query': query}},
        'id': 1
    }).encode()
    req = Request(MCP_INDEX_URL, data=body,
        headers={'Authorization': MCP_TOKEN, 'Content-Type': 'application/json'},
        method='POST')
    try:
        with urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
            text = resp['result']['content'][0]['text']
            return json.loads(text)
    except:
        return None

def kg_conn():
    os.makedirs(os.path.dirname(KG_DB), exist_ok=True)
    conn = sqlite3.connect(KG_DB)
    conn.row_factory = sqlite3.Row
    return conn

def write_to_kg(name, product_type, benchmark, price, unit, change_pct, direction, trend, source, impact, sectors, note=""):
    conn = kg_conn()
    conn.execute(
        """INSERT INTO commodities(date,name,product_type,benchmark,price,unit,change_pct,direction,trend,source,impact,affected_sectors,note)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (TODAY, name, product_type, benchmark, price, unit, change_pct, direction, trend, source, impact, sectors, note))
    conn.commit(); conn.close()

def write_to_quant(name, price, change_pct, unit):
    path = os.path.join(MACRO_DIR, f"{name}.parquet")
    df = pd.DataFrame([{"日期": TODAY, "商品": name, "最新值": price, "涨跌幅": change_pct, "单位": unit}])
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["日期"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def pull_macro():
    print(f"\n=== 期货行情 [{TODAY}] ===")
    for name, q, unit in FUTURES:
        resp = mcp_query(q)
        if not resp:
            print(f"  ✗ {name}: MCP失败"); continue
        raw = resp.get('data', {}).get('text', '')
        if '收盘价' not in raw:
            print(f"  ✗ {name}: 无数据"); continue
        lines = [l.strip() for l in raw.split('\n') if l.strip() and '|' in l and '---' not in l]
        if len(lines) < 2: continue
        cols = [c.strip() for c in lines[-1].split('|') if c.strip()]
        headers = [c.strip() for c in lines[-2].split('|') if c.strip()]
        price, chg = 0, 0
        for i, h in enumerate(headers):
            if i < len(cols):
                if '收盘价' in h: price = float(cols[i].replace(',',''))
                elif '涨跌幅' in h: chg = float(cols[i]) if cols[i] else 0
        direction = 'up' if chg > 0.5 else ('down' if chg < -0.5 else 'flat')
        write_to_kg(name, 'macro', f'iFinD {name}主连', price, unit, chg, direction, '', 'iFinD MCP', 'positive', '', f'{name}主连 {TODAY}')
        write_to_quant(name, price, chg, unit)
        print(f"  ✓ {name}: {price} {unit} ({chg:+.2f}%)")

def sync():
    conn = kg_conn()
    rows = conn.execute("SELECT * FROM commodities WHERE date=? ORDER BY product_type, name", (TODAY,)).fetchall()
    conn.close()
    if not rows: return
    summary = {"date": TODAY, "items": [dict(r) for r in rows]}
    out = os.path.join(MACRO_DIR, "commodity_snapshot.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ 快照: {len(rows)}条 → {out}")

def update_price(name, price):
    if name not in COMPANY_BENCHMARKS:
        print(f"未知产品: {name}"); return
    COMPANY_BENCHMARKS[name]["price"] = price
    COMPANY_BENCHMARKS[name]["date"] = TODAY
    print(f"  ✓ {name} 价格已更新为 {price}")

def status():
    conn = kg_conn()
    rows = conn.execute(
        "SELECT date,name,product_type,price,unit,change_pct,direction,source FROM commodities ORDER BY date DESC LIMIT 30"
    ).fetchall()
    conn.close()
    print(f"\n{'日期':<12} {'类型':<8} {'商品':<12} {'价格':<12} {'涨跌%':<8} {'方向':<6} {'来源'}")
    print("-"*70)
    for r in rows:
        d = dict(r)
        print(f"{d['date']:<12} {d['product_type']:<8} {d['name']:<12} {str(d['price']):<12} {d['change_pct']:<8.1f} {d['direction']:<6} {d['source']}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "macro": pull_macro()
    elif cmd == "all":
        pull_macro()
        sync()
    elif cmd == "status": status()
    elif cmd == "sync": sync()
    elif cmd == "update_price" and len(sys.argv) >= 4:
        update_price(sys.argv[2], float(sys.argv[3]))
    else: print(f"用法: {sys.argv[0]} [all|macro|sync|status|update_price <name> <price>]")
