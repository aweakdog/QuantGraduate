import urllib.request, json, os, ssl, time, sys
import pandas as pd

ROOT = "D:/myAI/WorkBuddy-workspace/quant-strategy"
KL = os.path.join(ROOT, "data/raw/kline")
CFG = json.load(open(os.path.expanduser("~/.workbuddy/mcp.json")))
SRV = CFG["mcpServers"]["ifind-stock"]
URL = SRV["url"]; AUTH = SRV["headers"]["Authorization"]
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

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
                   "clientInfo": {"name": "bf", "version": "1.0"}}}, None)
    if not b:
        print("INIT FAIL", err); return None
    sid = h.get("Mcp-Session-Id") or h.get("mcp-session-id")
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid

def call_perf(sid, q):
    b, err, h = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "get_stock_performance", "arguments": {"query": q}}}, sid)
    if not b: return None
    try:
        obj = json.loads(b)
        txt = obj["result"]["content"][0]["text"]
        data = json.loads(txt)
        return data.get("data", {}).get("answer")
    except Exception as e:
        return "PARSE_ERR " + repr(e)

def parse_num(s):
    s = str(s).strip()
    if s in ("", "--", "None", "nan", "NaN"): return None
    mult = 1.0
    if "亿" in s: mult = 1e8
    elif "万" in s: mult = 1e4
    s = s.replace("亿", "").replace("万", "").replace(",", "").replace("%", "")
    try: return float(s) * mult
    except: return None

def to_ifind(c6):
    return c6 + ".SH" if c6.startswith("6") else c6 + ".SZ"

def parse_md(answer):
    if not answer or answer.startswith("PARSE_ERR"): return []
    lines = [l for l in answer.split("\n") if l.strip().startswith("|")]
    if not lines: return []
    hidx = None; hdr = None
    for i, l in enumerate(lines):
        cols = [c.strip() for c in l.strip().strip("|").split("|")]
        if any("日期" in c for c in cols) and (any("开盘价" in c for c in cols) or any("前收盘价" in c for c in cols)):
            hidx = i; hdr = cols; break
    if hidx is None: return []
    def col(name):
        for j, c in enumerate(hdr):
            if c == name: return j
        for j, c in enumerate(hdr):
            if name in c:
                if name == "成交量" and "含大宗" in c: continue
                return j
        return None
    jd = col("日期"); jo = col("开盘价"); jh = col("最高价")
    jl = col("最低价"); jc = col("收盘价"); jv = col("成交量")
    jamt = col("成交额"); jpc = col("前收盘价")
    rows = []
    for l in lines[hidx + 1:]:
        cols = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cols) < len(hdr): continue
        if not (jd is not None and cols[jd].isdigit()): continue
        d = cols[jd]
        try: dt = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
        except: continue
        o = parse_num(cols[jo]) if jo is not None else None
        h = parse_num(cols[jh]) if jh is not None else None
        l_ = parse_num(cols[jl]) if jl is not None else None
        c = parse_num(cols[jc]) if jc is not None else None
        v = parse_num(cols[jv]) if jv is not None else None
        amt = parse_num(cols[jamt]) if jamt is not None else None
        pc = parse_num(cols[jpc]) if jpc is not None else None
        if c is None and pc is not None: c = pc
        if o is None and c is not None: o = c
        if c is None and pc is None: continue
        rows.append({"时间": dt, "开盘价": o, "最高价": h, "最低价": l_,
                     "收盘价": c, "成交量": v, "总金额": amt, "preclose": pc})
    for i, r in enumerate(rows):
        if r["收盘价"] is None:
            nxt = rows[i + 1]["preclose"] if i + 1 < len(rows) else None
            r["收盘价"] = nxt if nxt is not None else r["preclose"]
    for r in rows: r.pop("preclose", None)
    return rows

def to_chinese(df):
    rev = {"date": "时间", "close": "收盘价", "open": "开盘价",
           "high": "最高价", "low": "最低价", "volume": "成交量", "amount": "总金额"}
    return df.rename(columns=rev, errors="ignore")

GAP8 = ["301165","301368","688525","688506","688507","688631","603296","688347"]

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    sid = new_session()
    if not sid:
        print("NO SESSION"); return
    for c in GAP8:
        if only and c != only: continue
        q = f"{to_ifind(c)} 2020-01-01至2026-07-09的前复权每日开盘价、最高价、最低价、收盘价、成交量、成交额"
        rows = None
        for attempt in range(3):
            ans = call_perf(sid, q)
            if ans is None:
                sid = new_session(); continue
            rows = parse_md(ans)
            if rows: break
            time.sleep(2)
        if not rows:
            print(f"FAIL {c} ans={str(ans)[:600]}", flush=True); time.sleep(0.5); continue
        new = pd.DataFrame(rows, columns=["时间", "开盘价", "最高价", "最低价",
                                          "收盘价", "成交量", "总金额"])
        new["时间"] = pd.to_datetime(new["时间"])
        for col in ["成交量", "总金额"]:
            new[col] = new[col].fillna(0).round().astype("int64")
        p = os.path.join(KL, c + ".parquet")
        if os.path.exists(p):
            ex = to_chinese(pd.read_parquet(p))
            comb = pd.concat([ex, new]).drop_duplicates("时间", keep="last") \
                      .sort_values("时间").reset_index(drop=True)
        else:
            comb = new.sort_values("时间").reset_index(drop=True)
        comb.to_parquet(p, index=False)
        print(f"OK {c} total={len(comb)} {comb['时间'].min().date()}~{comb['时间'].max().date()}", flush=True)
        time.sleep(0.3)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
