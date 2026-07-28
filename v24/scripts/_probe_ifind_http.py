import urllib.request, json, os, ssl

CFG = json.load(open(os.path.expanduser("~/.workbuddy/mcp.json")))
srv = CFG["mcpServers"]["ifind-stock"]
URL = srv["url"]
AUTH = srv["headers"]["Authorization"]
print("URL:", URL)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

def rpc(payload, sid=None):
    data = json.dumps(payload).encode()
    headers = {"Authorization": AUTH, "Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if sid: headers["Mcp-Session-Id"] = sid
    req = urllib.request.Request(URL, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=60, context=CTX)
        return resp.read().decode(), None, dict(resp.headers)
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:800], dict(e.headers)

body, err, hdrs = rpc({"jsonrpc":"2.0","id":1,"method":"initialize",
    "params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"bf","version":"1.0"}}})
print("INIT ok:", bool(body), "len", len(body) if body else 0)
print("INIT head:", (body or err)[:200])
sid = hdrs.get("Mcp-Session-Id") or hdrs.get("mcp-session-id")
print("SID:", sid)

rpc({"jsonrpc":"2.0","method":"notifications/initialized"}, sid)

q = "600519.SH 2026-06-27至2026-07-09的前复权每日开盘价、最高价、最低价、收盘价、成交量、成交额"
body3, err3, _ = rpc({"jsonrpc":"2.0","id":2,"method":"tools/call",
    "params":{"name":"get_stock_performance","arguments":{"query":q}}}, sid)
print("CALL ok:", bool(body3))
print("RAW:", (body3 or err3)[:1200])
