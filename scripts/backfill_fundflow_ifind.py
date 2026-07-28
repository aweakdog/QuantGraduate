"""
iFinD MCP 增量拉取资金流数据

检查现有 fundflow_history.parquet 的最晚日期 → 逐只拉取缺失日期的 主力净流入额
输出: data/raw/fund_flow_full/fundflow_history.parquet (与现有格式一致)

用法: python scripts/backfill_fundflow_ifind.py [--progress]
"""
import urllib.request, json, os, ssl, time, sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "raw" / "fund_flow_full" / "fundflow_history.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

# iFinD MCP 连接
CFG = json.load(open(os.path.expanduser("~/.workbuddy/mcp.json")))
SRV = CFG["mcpServers"]["ifind-stock"]
URL = SRV["url"]; AUTH = SRV["headers"]["Authorization"]
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

progress = "--progress" in sys.argv

def rpc(payload, sid):
    data = json.dumps(payload).encode()
    h = {"Authorization": AUTH, "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if sid: h["Mcp-Session-Id"] = sid
    req = urllib.request.Request(URL, data=data, headers=h, method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=60, context=CTX)
        return r.read().decode(), None, dict(r.headers)
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:800], dict(e.headers)
    except Exception as e:
        return None, repr(e)[:300], {}

def new_session():
    b, err, h = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "ff", "version": "1.0"}}}, None)
    if not b: return None
    sid = h.get("Mcp-Session-Id") or h.get("mcp-session-id")
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid

def parse_num(s):
    """解析 iFinD 返回的金额字符串, 支持 万/亿 单位"""
    s = str(s).strip()
    if s in ("", "--", "None", "nan", "NaN", "-", "0.0"): return None
    s = s.replace(",", "").replace(" ", "")
    mult = 1.0
    if "亿" in s: mult = 1e8
    elif "万" in s: mult = 1e4
    s = s.replace("亿", "").replace("万", "").replace("%", "").strip()
    try: return float(s) * mult
    except: return None

def call_perf(sid, q):
    b, err, h = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "get_stock_performance",
                   "arguments": {"query": q}}}, sid)
    if not b: return None
    try:
        obj = json.loads(b)
        txt = obj["result"]["content"][0]["text"]
        data = json.loads(txt)
        return data.get("data", {}).get("answer")
    except Exception as e:
        return "PARSE_ERR " + repr(e)

def parse_md_table(answer):
    """解析 Markdown 表格, 返回 [{col: val}, ...]"""
    if not answer or answer.startswith("PARSE_ERR"):
        return []
    lines = [l for l in answer.split("\n") if l.strip().startswith("|")]
    if not lines: return []
    # 找表头行 (含 日期/证券代码 等)
    hdr_idx = None; hdr = None
    for i, l in enumerate(lines):
        cols = [c.strip() for c in l.strip().strip("|").split("|")]
        if any("日期" in c for c in cols) and any("净流入" in c for c in cols):
            hdr_idx = i; hdr = cols; break
    if hdr_idx is None: return []
    # 列索引
    j_date = next((j for j, c in enumerate(hdr) if "日期" in c), None)
    j_main_net = next((j for j, c in enumerate(hdr) if "主力净流入额" in c), None)
    j_main_vol = next((j for j, c in enumerate(hdr) if "主力净流入量" in c), None)
    j_retail_net = next((j for j, c in enumerate(hdr) if "散户净流入额" in c), None)
    if j_date is None or j_main_net is None:
        return []
    rows = []
    for l in lines[hdr_idx + 1:]:
        cols = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cols) < len(hdr): continue
        if not (j_date is not None and cols[j_date].isdigit()): continue
        d_str = cols[j_date]
        try:
            dt = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}")
        except:
            continue
        main_net = parse_num(cols[j_main_net]) if j_main_net is not None else None
        main_vol = parse_num(cols[j_main_vol]) if j_main_vol is not None else None
        retail_net = parse_num(cols[j_retail_net]) if j_retail_net is not None else None
        rows.append({"date": dt, "main_force_net": main_net,
                      "main_force_vol": main_vol, "retail_net": retail_net})
    return rows

def to_ifind(c6):
    return c6 + ".SH" if c6.startswith("6") else c6 + ".SZ"

def get_stock_codes() -> list[str]:
    """从 watchlist_216 获取股票代码列表."""
    wl_path = DATA_DIR / "universe" / "watchlist_216.json"
    if not wl_path.exists():
        wl_path = DATA_DIR / "universe" / "watchlist.json"
    with open(wl_path, encoding="utf-8") as f:
        wl = json.load(f)
    items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
    return [str(x["code"]).split(".")[0] for x in items]

def main():
    codes = get_stock_codes()
    print(f"股票数: {len(codes)}")

    # 检查现有数据的最晚日期
    if OUT.exists():
        old = pd.read_parquet(str(OUT))
        old_max = pd.to_datetime(old["date"]).max()
        existing_codes = set(old["code"].unique())
    else:
        old = pd.DataFrame(columns=["code", "date", "main_force_net", "main_force_pct",
                                     "dde_net", "mtss_balance", "fund_flow"])
        old_max = pd.Timestamp("2020-01-01")
        existing_codes = set()

    end_default = pd.Timestamp.today() - pd.Timedelta(days=1)
    print(f"现有数据最新日: {old_max.date()}, 目标: {end_default.date()}")

    if old_max >= end_default:
        print("已是最新, 无需补数")
        if progress:
            print("PROGRESS:1/1")
        return

    # 需要补数的股票 (已有 + 新增)
    new_stocks = [c for c in codes if c not in existing_codes]
    stale_stocks = [c for c in codes if c in existing_codes]
    print(f"新增股票: {len(new_stocks)}, 需补数据: {len(stale_stocks)}")

    sid = new_session()
    if not sid:
        print("iFinD 会话创建失败"); sys.exit(1)

    done = fail = 0
    to_fetch = new_stocks + stale_stocks
    total = len(to_fetch)
    if progress:
        print(f"PROGRESS:0/{total}", flush=True)

    all_new_rows = []
    for idx, c6 in enumerate(to_fetch):
        # iFinD 查询: 最近5日主力净流入数据
        q = f"{to_ifind(c6)} 最近5个交易日的主力净流入额、主力净流入量、散户净流入额"
        rows = None
        for attempt in range(3):
            ans = call_perf(sid, q)
            if ans is None:
                sid = new_session(); continue
            rows = parse_md_table(ans)
            if rows: break
            time.sleep(1)
        if not rows:
            print(f"  [{idx+1}/{total}] ⚠ {c6}: 无数据")
            fail += 1
        else:
            for r in rows:
                r["code"] = c6
                # 估算 main_force_pct (主力净流入额/成交额... 实际没成交额, 保持None)
                r["main_force_pct"] = None
                r["dde_net"] = None
                r["mtss_balance"] = None
                r["fund_flow"] = None
            all_new_rows.extend(rows)
            done += 1
            if idx < 5 or (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{total}] ✅ {c6}: {len(rows)} 天", flush=True)

        if progress:
            print(f"PROGRESS:{idx+1}/{total}", flush=True)
        time.sleep(0.5)

    if not all_new_rows:
        print(f"无新数据 done={done} fail={fail}")
        return

    new_df = pd.DataFrame(all_new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])
    new_df = new_df.sort_values(["date", "code"]).reset_index(drop=True)

    # 合并到旧数据
    if len(old) > 0:
        combined = pd.concat([old, new_df]).drop_duplicates(
            ["date", "code"], keep="last").sort_values(["date", "code"]).reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(str(OUT), index=False)
    print(f"\n完成: +{len(new_df)} 行 → 总计 {len(combined)} 行, "
          f"date {combined['date'].min().date()} ~ {combined['date'].max().date()}, "
          f"done={done} fail={fail}")

if __name__ == "__main__":
    main()
