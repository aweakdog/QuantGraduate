"""
美股隔夜行情 → A股映射信号

数据源:
  - 美股个股: westock-data (腾讯自选股接口)
  - 美股指数: SOX 从已有 macro parquet; SPX/IXIC 从 MCP index_data
  - A股映射: us_sector_mapping.json → concept_stock_map.json

日期对齐:
  US T日收盘 → BJT T+1日04:00 → A股T+1日可用
  K线日期 = US实际交易日, 特征工程 shift(1) 后自动对齐

运行:
  python pull_us_overnight.py

自动化: 交易日 08:00
"""
import json, os, sys, subprocess, csv, io, pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen

BJT = timezone(timedelta(hours=8))
NOW = datetime.now(BJT)
TODAY = NOW.strftime("%Y-%m-%d")
TODAY8 = NOW.strftime("%Y%m%d")
DATA_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data"
EVENTS_DIR = os.path.join(DATA_DIR, "raw", "events_daily")
SECTOR_MAP = os.path.join(DATA_DIR, "universe", "us_sector_mapping.json")
CONCEPT_MAP = os.path.join(DATA_DIR, "universe", "concept_stock_map.json")
MACRO_DIR = os.path.join(DATA_DIR, "raw", "macro")

NODE = "C:/Users/admin/.workbuddy/binaries/node/versions/22.22.2/node.exe"
WESTOCK = "D:/myAI/WorkBuddy/resources/app.asar.unpacked/resources/builtin-skills/westock-data/scripts/index.js"

MCP_TOKEN = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjtIFPAyXTksV4T-hKnzLqkW0Q1j-02SSnHyIzSMgGqD74Rj1lZto2oIAynV5gHiZXjTgX8ARvE1NtnBNLbdHMADuomjMNRHAXpPE83sCL4ehLGL6zb5_n8XVzLwr_RuJ4SZiekMR3sEMGNePywrP2flMO_K6R0suTvFTlSWU5WxOYMKqLxUOciZnZqTnxUs6_Lnj6He4XBEgul2VdJX4w6lcPq5ibDx7CDp-8SzW_FW0CBkREtIWBbyuqHaQyWdnUbg6nPoCo3sD3ipTL3ereUqX33GY8mn8dYfIFKZShADp5kGziTtqWRLQ.gQm_IZm7qxG8OKz-.mHYCbUproLUp1qLvMntUQ5rq6e27ORuzqnhXvhkIVFbA5UsTZBq_1UqJuq4XlN5EuI6j2o91dgWFz2vIHhm7482C1vcpwDTlUC48j_UymGR03dX8iiriSA-qE7ZQJLx50YFrG7aFw5sALibKzwDGVETilkI9upyDUu5s7tMg3cIhj0GUWU-8xso-AZf_frGahYyEzZsK4EHKHBxxVmE5IghBnJcTvjvB-Hs46nrhbeQ2wr2aSP82bq8JtXaHvstS6CC_63YS_jB7KWBF1sqkP25138A7y31xzOlmMEW_GIuDOElFXT2SXJ3qbGQmYBg_EwPMyGdy4rs2xk74WmQNyCzrdRIY86zPOlWqyt2EHJi9GHwxOjgAWdJ8eQ0o_kCsw7lyvYoWwQbyeqcs2rOTrhLeMbI.-AlkZXTtGcQkQqGT_JGRzw"
MCP_IDX = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-index-mcp"

# ─── US标的查询清单 ──────────────────────────────────────────
# (显示名, westock code, MCP备用, 映射类型)
US_TICKERS = [
    ("英伟达", "usNVDA", "NVDA.OQ", "stock"),
    ("特斯拉", "usTSLA", "TSLA.OQ", "stock"),
    ("苹果",   "usAAPL", "AAPL.OQ", "stock"),
    ("台积电", "usTSM",  "TSM.N",   "stock"),
    ("AMD",    "usAMD",  "AMD.OQ",  "stock"),
    ("美光",   "usMU",   "MU.OQ",   "stock"),
    ("微软",   "usMSFT", "MSFT.OQ", "stock"),
    ("谷歌",   "usGOOGL","GOOGL.OQ","stock"),
    ("亚马逊", "usAMZN", "AMZN.OQ", "stock"),
    ("纳斯达克ETF", "usQQQ.OQ", "", "etf"),
]

# US指数: 从macro parquet + MCP补充
US_INDICES = ["SPX", "IXIC", "SOX"]

def westock_cmd(args):
    """执行 westock-data 命令"""
    cmd = [NODE, WESTOCK] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.stdout

def parse_table(md_text):
    """解析westock-data返回的markdown表格 → list[dict]"""
    lines = [l.strip() for l in md_text.split('\n') if l.strip()]
    data_start = -1
    for i, l in enumerate(lines):
        if l.startswith('|') and '---' not in l:
            if data_start == -1:
                data_start = i
            elif i > data_start + 1:
                # 第二行数据开始
                rows = []
                headers = [h.strip() for h in lines[data_start].split('|') if h.strip()]
                for j in range(data_start + 2, len(lines)):
                    if not lines[j].startswith('|'): break
                    cols = [c.strip() for c in lines[j].split('|') if c.strip()]
                    if len(cols) == len(headers):
                        rows.append(dict(zip(headers, cols)))
                return rows
    return []

def quote_us(ticker):
    """westock-data quote 获取美股行情"""
    raw = westock_cmd(["quote", ticker])
    rows = parse_table(raw)
    return rows[0] if rows else None

def kline_us(ticker, limit=200):
    """westock-data kline 获取美股日K线"""
    raw = westock_cmd(["kline", ticker, "--period", "day", "--limit", str(limit)])
    return parse_table(raw)

def mcp_query(query):
    body = json.dumps({'jsonrpc':'2.0','method':'tools/call','params':{'name':'index_data','arguments':{'query':query}},'id':1}).encode()
    req = Request(MCP_IDX, data=body, headers={'Authorization':MCP_TOKEN,'Content-Type':'application/json'}, method='POST')
    try:
        with urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
            text = resp['result']['content'][0]['text']
            return json.loads(text)
    except: return None

def get_us_index_data():
    """获取美股指数行情 (SOX从macro parquet, SPX/IXIC从MCP)"""
    results = {}
    
    # SOX: 从已有 parquet 取最新
    sox_path = os.path.join(MACRO_DIR, "全球半导体SOX.parquet")
    if os.path.exists(sox_path):
        df = pd.read_parquet(sox_path)
        if len(df) >= 2:
            last2 = df.tail(2)
            chg = (last2.iloc[-1]["最新值"] / last2.iloc[-2]["最新值"] - 1) * 100
            results["费城半导体"] = {"chg": round(chg, 2), "price": last2.iloc[-1]["最新值"]}
    
    # SPX/IXIC: 从MCP
    for name, q in [("标普500","标普500指数SPX的最新收盘价涨跌幅"), 
                    ("纳斯达克","纳斯达克指数IXIC的最新收盘价涨跌幅")]:
        resp = mcp_query(q)
        if resp:
            raw = resp.get('data',{}).get('text','')
            lines = [l.strip() for l in raw.split('\n') if l.strip() and '|' in l and '---' not in l]
            if len(lines) >= 2:
                cols = [c.strip() for c in lines[-1].split('|') if c.strip()]
                headers = [c.strip() for c in lines[-2].split('|') if c.strip()]
                price, chg = 0, 0
                for i, h in enumerate(headers):
                    if i < len(cols):
                        if '收盘价' in h: price = float(cols[i].replace(',',''))
                        elif '涨跌幅' in h and cols[i]: chg = float(cols[i])
                results[name] = {"chg": chg, "price": price}
    
    return results

def resolve_stocks(aconcepts, direct_codes):
    """概念名 → A股代码列表"""
    with open(CONCEPT_MAP, encoding='utf-8') as f:
        cmap = json.load(f)
    ctos = cmap.get('concept_to_stocks', {})
    codes = set()
    for ac in aconcepts:
        for code in ctos.get(ac, []):
            codes.add(code[:6])
    for dc in (direct_codes or []):
        codes.add(dc[:6])
    return sorted(codes)

def save_us_kline(ticker_us, data_name):
    """保存美股K线到macro parquet, 日期对齐: US T日→标签为T日 (shift在feature_engine处理)"""
    kl = kline_us(ticker_us, limit=200)
    if not kl:
        return
    records = []
    for r in kl:
        try:
            records.append({
                "日期": r.get("date",""),
                "商品": data_name,
                "最新值": float(r.get("last",0)),
                "涨跌幅": 0,  # 从K线算变化率
                "单位": "美元"
            })
        except: pass
    if not records:
        return
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["日期"]).sort_values("日期")
    # 算涨跌幅
    df["涨跌幅"] = df["最新值"].pct_change() * 100
    
    path = os.path.join(MACRO_DIR, f"{data_name}.parquet")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["日期"]).sort_values("日期")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  ✓ K线已保存: {data_name} {len(df)}条")

def main():
    with open(SECTOR_MAP, encoding='utf-8') as f:
        cfg = json.load(f)
    mappings = cfg["mappings"]
    threshold = cfg.get("threshold", 2.0)
    print(f"US板块映射: {len(mappings)} 组, 阈值 {threshold}%")

    # 1. 获取US指数
    print(f"\n=== US指数 [{TODAY}] ===")
    index_data = get_us_index_data()
    for k, v in index_data.items():
        print(f"  {k}: {v['price']} ({v['chg']:+.2f}%)")

    all_prices = {}  # 合并所有US价格

    # 1b. 获取美股板块ETF
    print(f"\n=== US板块ETF ===")
    us_prices = {}
    etf_list = cfg.get("etf_tickers", [])
    for etf in etf_list:
        q = quote_us(etf["westock"])
        if q:
            chg = float(q.get("change_percent", 0))
            price = float(q.get("price", 0))
            us_prices[etf["name"]] = {"chg": chg, "price": price}
            print(f"  {etf['name']}: ${price} ({chg:+.2f}%)")
            save_us_kline(etf["westock"], f"us_{etf['westock'].replace('us','')}")
        else:
            print(f"  {etf['name']}: ❌")

    # 2. 获取US个股行情 + 保存K线
    print(f"\n=== US个股 ===")
    for name, ticker, mcp_code, typ in US_TICKERS:
        q = quote_us(ticker)
        if q:
            chg = float(q.get("change_percent", 0))
            price = float(q.get("price", 0))
            us_prices[name] = {"chg": chg, "price": price}
            print(f"  {name}: ${price} ({chg:+.2f}%)")
            # 保存K线
            save_us_kline(ticker, f"us_{ticker.replace('us','')}")
        else:
            print(f"  {name}: ❌")
    
    # 合并指数+个股
    all_prices = {**index_data, **us_prices}
    # 过滤异常值
    all_prices = {k: v for k, v in all_prices.items() if abs(v.get("chg", 0)) < 50}
    # 过滤异常值
    all_prices = {k: v for k, v in all_prices.items() if abs(v.get("chg", 0)) < 100}

    # 3. 检测信号
    print(f"\n=== A股映射信号 ===")
    signals = []
    sector_maps = cfg.get("sector_mappings", [])
    
    # 个股映射
    for m in mappings:
        us_name = m["us_name"]
        us_sector = m.get("us_sector", "")
        
        # 找对应价格
        matched_key = next((k for k in all_prices if k in us_name or us_name in k), None)
        if matched_key:
            chg = all_prices[matched_key]["chg"]
        else:
            # 按板块关联近似
            if "半导" in us_sector:
                chg = index_data.get("费城半导体", {}).get("chg", 0)
            elif "科技" in us_sector:
                chg = index_data.get("纳斯达克", {}).get("chg", 0)
            elif "大盘" in us_sector:
                chg = index_data.get("标普500", {}).get("chg", 0)
            else:
                chg = 0
        
        if abs(chg) < threshold:
            continue
        
        direction = "bullish" if chg > 0 else "bearish"
        p_level = m.get("weight", "P1")
        if abs(chg) >= 5: p_level = "P0"
        if abs(chg) < 1: p_level = "P2"
        marker = "🟢" if direction == "bullish" else "🔴"

        codes = resolve_stocks(m.get("a_concepts",[]), m.get("direct_codes",[]))
        if not codes:
            codes = ["__ALL__"]

        print(f"  {marker} {us_name} ({chg:+.2f}%) → {p_level} | {len(codes)}只")
        
        us_code = m.get("us_code", us_name)
        ticker_short = us_code.split('.')[0].lower()
        for code in codes:
            signals.append({
                "code": code, "name": "",
                "event_type": f"us_{ticker_short}_{direction}",
                "p_level": p_level, "direction": direction,
                "change_pct": round(chg, 2),
                "reason": f"{us_name} {chg:+.2f}% → {us_sector}"
            })

    # 板块ETF映射
    for sm in sector_maps:
        ename = sm["etf_name"]
        if ename not in us_prices:
            continue
        chg = us_prices[ename]["chg"]
        if abs(chg) < threshold:
            continue
        direction = "bullish" if chg > 0 else "bearish"
        p_level = sm.get("weight", "P2")
        if abs(chg) >= 5: p_level = "P0"
        if abs(chg) < 1: p_level = "P2"
        codes = resolve_stocks(sm.get("a_concepts",[]), [])
        if not codes: codes = ["__ALL__"]
        print(f"  {'🟢' if direction=='bullish' else '🔴'} {ename} ({chg:+.2f}%) → {p_level} | {len(codes)}只")
        etf_ticker = next((e["ticker"] for e in etf_list if e["name"]==ename), ename)
        for code in codes:
            signals.append({
                "code": code, "name": "",
                "event_type": f"us_{etf_ticker.lower()}_{direction}",
                "p_level": p_level, "direction": direction,
                "change_pct": round(chg, 2),
                "reason": f"{ename} {chg:+.2f}%"
            })

    # 4. 保存信号
    if not signals:
        print("  ⚠ 无触发信号"); return

    df = pd.DataFrame(signals)
    df["date"] = TODAY
    os.makedirs(EVENTS_DIR, exist_ok=True)
    path = os.path.join(EVENTS_DIR, f"us_overnight_{TODAY8}.parquet")
    df.to_parquet(path, index=False)

    clean = df.rename(columns={"code":"stock_code","event_type":"event_name","p_level":"p_level"})
    clean["dir_hard"] = clean["direction"].map({"bullish":1,"bearish":-1}).fillna(0)
    clean["impact"] = clean["dir_hard"] * clean["p_level"].map({"P0":10,"P1":5,"P2":2}).fillna(1)
    cp = os.path.join(EVENTS_DIR, f"us_overnight_clean_{TODAY8}.parquet")
    clean.to_parquet(cp, index=False)
    print(f"\n  ✓ {len(signals)}条 → {path}")

if __name__ == "__main__":
    main()
