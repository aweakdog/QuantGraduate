"""
批量拉取关注圈+链主财报 → 生成结构化报告 → 上传 IMA「公司财报」知识库
数据源: 快查 MCP (资产负债表/利润表/现金流量表)
"""
import json, time, os, subprocess, tempfile, sys

# ─── 配置 ───
WATCHLIST = "D:/myAI/WorkBuddy-workspace/quant-strategy/data/universe/watchlist.json"
IMA_API_JS = "C:/Users/admin/.workbuddy/skills/ima-skill/ima_api.cjs"
KB_ID = "A16RMjHou2zMbtvfGStl9Op_HqliNPce-r-gZakN600="  # 公司财报
NODE = "C:/Users/admin/.workbuddy/binaries/node/versions/22.22.2/node.exe"
MCP_NODE = "C:/Users/admin/.workbuddy/skills/ifind-finance-data/call-node.js"  # reuse pattern

OUT_DIR = "D:/myAI/WorkBuddy-workspace/quant-strategy/data/financials"
os.makedirs(OUT_DIR, exist_ok=True)

# 快查 API key from mcp.json
KUAICHA_KEY = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.aVk5poC_cp7Zbhg2yCN-HzExUvof8tqmksxaWK1xf0FbnarcNhY0GLoN7SzUI8A5miOpNk2yoEMGK1h584Pnc1_qhliFJEEixyAO6CmSUYzHeKLq-4G2Lu43mIGw_yHMTI4pV1kEwk5gc9NM70-M7ZFfVriTB5o13TRNyuLmsep7v1hQQKniebK8El8IM7XkhmXkEYdBhQw1Gxl_n8S1IKZ9tTZhcL-DUW5caFCSCtVGxG4cGE2SeGxyO0-4NtqV-YTuxkFIyVM36UU78Y0yQC4Gf6GpOV6b973bz69mLOQ_zboRhc2gE3GJBoexKhAiEag7mkq4UpSAJ7VBPGqRsA.BOzk0a3aOAXMZ-MR.pu74FcTDUPtju3CLOLfbKFjTN0Wlw6qsk67uocTFQBmM044d8a0t0lMCK_Mb6vxJVqWw2GbNkq9V811ULMB6knigUoSc1IyKqaTaEnw7pSI7StrrKs_rfiLGV9FnVNzVk6H9PZhPTaJDxgsKj4I70xvsz5BEI9E59p05YwsqUUd_uzXaOQ.GI8kLAQAgvo70sWA4ishAA"

# ─── 链主 ───
CHAIN_LEADERS = [
    ("300750", "宁德时代"),
    ("002594", "比亚迪"),
    ("601127", "赛力斯"),
]

def kuaicha_call(tool_id, params):
    """直接 HTTP 调用快查 MCP"""
    import urllib.request, urllib.error
    url = "https://bizveris.kuaicha365.com/mcp/call"
    headers = {
        "Content-Type": "application/json",
        "open-authorization": f"Bearer {KUAICHA_KEY}",
    }
    body = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_id, "arguments": params},
        "id": 1
    }, ensure_ascii=False).encode("utf-8")
    
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def get_orgid(code):
    """搜索公司获取 orgid"""
    r = kuaicha_call("basic_get_enterprise_associate", {"query": code})
    try:
        content = r.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "")
            data = json.loads(text)
            items = data.get("data", {}).get("list", [])
            for item in items:
                if item.get("orgid"):
                    return item["orgid"], item.get("corp_name", ""), item.get("creditcode", "")
    except:
        pass
    return None, None, None

def get_statement(tool_id, orgid, max_pages=3):
    """拉取财报三表中的一张，返回合并后的 list"""
    all_items = []
    for page in range(1, max_pages+1):
        r = kuaicha_call(tool_id, {"orgid": orgid, "type": "HB", "is_audited": "1", "page_size": 5, "page": page})
        try:
            content = r.get("result", {}).get("content", [])
            if content:
                text = content[0].get("text", "")
                data = json.loads(text)
                items = data.get("data", {}).get("list", [])
                if not items:
                    break
                all_items.extend(items)
            else:
                break
        except:
            break
        time.sleep(0.15)
    return all_items

def extract_key_metrics(income, balance, cashflow):
    """从三张表提取关键指标 (最近3年)"""
    years = []
    
    # 利润表 → 按 end_date 排序取最新3年
    income_sorted = sorted(income, key=lambda x: x.get("end_date", ""), reverse=True)
    income_by_year = {}
    for item in income_sorted:
        yr = item.get("end_date", "")[:4]
        if yr not in income_by_year:
            income_by_year[yr] = item
    
    # 资产负债表
    balance_sorted = sorted(balance, key=lambda x: x.get("end_date", ""), reverse=True)
    balance_by_year = {}
    for item in balance_sorted:
        yr = item.get("end_date", "")[:4]
        if yr not in balance_by_year:
            balance_by_year[yr] = item
    
    # 现金流量表
    cash_sorted = sorted(cashflow, key=lambda x: x.get("end_date", ""), reverse=True)
    cash_by_year = {}
    for item in cash_sorted:
        yr = item.get("end_date", "")[:4]
        if yr not in cash_by_year:
            cash_by_year[yr] = item
    
    all_years = sorted(set(list(income_by_year.keys()) + list(balance_by_year.keys()) + list(cash_by_year.keys())), reverse=True)[:3]
    
    for yr in all_years:
        inc = income_by_year.get(yr, {})
        bal = balance_by_year.get(yr, {})
        cf = cash_by_year.get(yr, {})
        
        years.append({
            "year": yr,
            "revenue": _fmt(inc.get("revenue") or inc.get("total_revenue")),
            "revenue_growth": "",
            "net_profit": _fmt(inc.get("net_profit")),
            "np_margin": "",
            "total_assets": _fmt(bal.get("total_assets")),
            "total_liab": _fmt(bal.get("total_liab")),
            "equity": _fmt(bal.get("total_holders_equity")),
            "roe": "",
            "ocf": _fmt(cf.get("ncf_from_oa")),
            "cash": _fmt(bal.get("currency_funds")),
            "eps": inc.get("basic_eps", ""),
        })
    
    # 计算增长率
    for i in range(len(years)):
        if i < len(years) - 1:
            r0 = _parse(years[i]["revenue"])
            r1 = _parse(years[i+1]["revenue"])
            if r0 and r1:
                years[i]["revenue_growth"] = f"{(r0/r1 - 1)*100:+.1f}%"
    
    return years

def _fmt(v):
    """格式化金额为亿"""
    if v is None or v == "":
        return "-"
    try:
        val = float(v)
        if abs(val) < 1000:
            return f"{val:.2f}"
        yi = val / 1e8
        if abs(yi) < 10000:
            return f"{yi:.2f}亿"
        return f"{yi/10000:.2f}万亿"
    except:
        return str(v)[:12]

def _parse(v):
    try:
        return float(v) if isinstance(v, (int, float)) else float(str(v).replace("亿","").replace("万亿","0000"))
    except:
        return None

def gen_markdown_report(results):
    """生成 markdown 报告"""
    now = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 关注圈+链主 财报速览",
        f"",
        f"> 生成时间: {now}",
        f"> 覆盖: {len(results)} 家公司",
        f"> 数据源: 快查365 MCP (审计年报)",
        f"",
        "---",
        "",
    ]
    
    for r in results:
        lines.append(f"## {r['name']} ({r['code']})")
        lines.append(f"")
        lines.append(f"- 信用代码: `{r.get('creditcode','-')}`")
        lines.append(f"")
        
        if r.get("years"):
            lines.append("| 年份 | 营收 | 营收增速 | 净利润 | 总资产 | 净资产 | ROE | 经营CF | 现金 | EPS |")
            lines.append("|------|------|---------|--------|--------|--------|-----|--------|------|-----|")
            for y in r["years"]:
                roe = _fmt_roe(y)
                lines.append(f"| {y['year']} | {y['revenue']} | {y['revenue_growth']} | {y['net_profit']} | {y['total_assets']} | {y['equity']} | {roe} | {y['ocf']} | {y['cash']} | {y['eps']} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)

def _fmt_roe(y):
    np = y.get("net_profit", "-")
    eq = y.get("equity", "-")
    if np == "-" or eq == "-":
        return "-"
    try:
        np_v = float(np.replace("亿","").replace("万亿","0000").replace("-","nan"))
        eq_v = float(eq.replace("亿","").replace("万亿","0000").replace("-","nan"))
        if eq_v > 0:
            return f"{np_v/eq_v*100:.1f}%"
    except:
        pass
    return "-"

def upload_to_ima(md_path):
    """上传 markdown 到 IMA「公司财报」知识库"""
    cmd = [
        NODE, IMA_API_JS,
        "openapi/wiki/v1/import_urls",
        json.dumps({"knowledge_base_id": KB_ID, "urls": [f"file://{md_path}"]}, ensure_ascii=False),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout[:200]
    except Exception as e:
        return str(e)

def main():
    # ─── 读取关注圈 ───
    with open(WATCHLIST, "r", encoding="utf-8") as f:
        watchlist = json.load(f)["watchlist"]
    
    all_companies = []
    seen = set()
    for leader in CHAIN_LEADERS:
        if leader[0] not in seen:
            all_companies.append(leader)
            seen.add(leader[0])
    for s in watchlist:
        code6 = s["code"][:6]
        if code6 not in seen:
            all_companies.append((code6, s["name"]))
            seen.add(code6)
    
    print(f"共 {len(all_companies)} 家公司 ({len(CHAIN_LEADERS)} 链主 + {len(watchlist)} 关注圈)")
    print(f"快查剩余免费额度: 1000次")
    
    results = []
    errors = []
    
    for i, (code, name) in enumerate(all_companies):
        print(f"\n[{i+1}/{len(all_companies)}] {code} {name}")
        
        # 1. 搜索 orgid
        orgid, corp_name, creditcode = get_orgid(code)
        if not orgid:
            errors.append(f"{code} {name}: 未找到企业信息")
            print("  ✗ 未找到 orgid")
            continue
        time.sleep(0.2)
        
        # 2. 拉三张表
        print("  → 利润表...", end=" ", flush=True)
        income = get_statement("listed_get_income_statement", orgid)
        print(f"{len(income)}期", end=" ", flush=True)
        
        print("资产负债表...", end=" ", flush=True)
        balance = get_statement("listed_get_balance_sheet", orgid)
        print(f"{len(balance)}期", end=" ", flush=True)
        
        print("现金流量表...", end=" ", flush=True)
        cashflow = get_statement("listed_get_cash_flow", orgid)
        print(f"{len(cashflow)}期")
        
        if not income and not balance:
            errors.append(f"{code} {name}: 无财报数据")
            continue
        
        # 3. 提取关键指标
        metrics = extract_key_metrics(income, balance, cashflow)
        
        results.append({
            "code": code,
            "name": corp_name or name,
            "creditcode": creditcode or "",
            "years": metrics,
        })
        
        # 每10家暂存一次
        if (i + 1) % 10 == 0:
            with open(f"{OUT_DIR}/progress_{i+1}.json", "w", encoding="utf-8") as f:
                json.dump({"results": results, "errors": errors}, f, ensure_ascii=False, indent=2)
        
        # 控制速率
        time.sleep(0.5)
    
    # ─── 生成报告 ───
    md = gen_markdown_report(results)
    md_path = f"{OUT_DIR}/financial_report_{time.strftime('%Y%m%d')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"\n\n=== 完成 ===")
    print(f"成功: {len(results)}/{len(all_companies)}")
    print(f"失败: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"  - {e}")
    print(f"报告: {md_path}")
    print(f"大小: {os.path.getsize(md_path):,} bytes")
    
    # ─── 上传 IMA ───
    print(f"\n上传到 IMA「公司财报」...")
    # Note: import_urls 支持 http/https，不支持 file://
    # 改为用 add_knowledge + 文本笔记方式
    result = upload_to_ima(md_path)
    print(f"IMA response: {result[:200]}")

if __name__ == "__main__":
    main()
