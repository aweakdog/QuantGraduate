"""
24:00 统一数据拉取 — 并行后台执行
P0: 资金流 (iFinD 7年)
P1: 基本面 (thsdk wencai 修正字段名)
P2: 链主事件 (iFinD)
"""
import subprocess, sys, os, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJ = r"D:\myAI\Hermes-Workspace\quant-strategy"
SKILL = r"C:\Users\admin\AppData\Local\hermes\skills\ifind-finance-data"
EVENTS_DIR = os.path.join(PROJ, "data", "raw", "events")
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(os.path.join(EVENTS_DIR, "chain_leader"), exist_ok=True)

# 加载链主清单
with open(os.path.join(PROJ, "data", "universe", "supply_chain_map.json")) as f:
    sc = json.load(f)

leaders = []
for c in sc["chains"]:
    cl = c["chain_leader"]
    if cl.get("market") in ("US", "Unlisted"):
        leaders.append({"name": cl["name"], "code": cl.get("code", ""), "market": cl["market"]})
    else:
        leaders.append({"name": cl["name"], "code": cl["code"], "market": cl.get("market", "SZ")})

print(f"链主: {len(leaders)} 个")
print(f"开始时间: {time.strftime('%H:%M:%S')}")
t0 = time.time()

# ─── P2: 链主事件拉取 ───────────────────────────────────
os.chdir(SKILL)
sys.path.insert(0, SKILL)
from call import call
import pandas as pd

def fetch_chain_events(leader):
    name = leader["name"]
    out = os.path.join(EVENTS_DIR, "chain_leader", f"{name.replace(' ','_')}.parquet")
    if os.path.exists(out):
        return (name, "skip")
    recs = []
    # 新闻搜索: 每3个月一段
    periods = [("2020-01-01","2022-12-31"), ("2023-01-01","2025-06-30"), ("2025-07-01","2026-06-30")]
    for start, end in periods:
        try:
            # 尝试多种时间粒度
            for yr in range(2020, 2027):
                r = call("news", "search_news", {"query": name, "time_start": f"{yr}-01-01", 
                                                   "time_end": f"{yr}-12-31", "size": 10})
                if r.get("ok") and r.get("data"):
                    d = json.loads(r["data"]["result"]["content"][0]["text"])
                    ans = d.get("data", {}).get("answer", "")
                    if ans:
                        recs.append({"leader": name, "year": yr, "content": ans})
        except: pass
    if recs:
        pd.DataFrame(recs).to_parquet(out, index=False)
        return (name, f"{len(recs)}条")
    return (name, "无数据")

def fetch_fund_flow_single(code6, mkt):
    """拉单只股票资金流（每日主力净流入额）"""
    out = os.path.join(PROJ, "data", "raw", "fund_flow", f"{code6}.parquet")
    if os.path.exists(out):
        return (code6, "skip")
    recs = []
    for yr in range(2020, 2027):
        try:
            # "每日主力净流入额" 返回日频 data: code, name, date, volume(股), amount(元)
            r = call("stock", "get_stock_info", {"query": f"{code6} {yr} 每日主力净流入额"})
            if r.get("ok") and r.get("data"):
                d = json.loads(r["data"]["result"]["content"][0]["text"])
                ans = d.get("data", {}).get("answer", "")
                for line in ans.split("\n"):
                    if f"|{code6}" not in line: continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 5:
                        ds = parts[3]  # date
                        ns = parts[4]  # net amount
                        if ds.isdigit() and len(ds)==8 and ds.startswith("20"):
                            if "亿" in ns: nv = float(ns.replace("亿",""))*1e8
                            elif "万" in ns: nv = float(ns.replace("万",""))*1e4
                            else:
                                try: nv = float(ns.replace(",",""))
                                except: nv = None
                            if nv and abs(nv) > 100:
                                recs.append({"date":ds,"main_force_net":nv})
        except: pass
    if recs:
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        df.to_parquet(out, index=False)
        return (code6, f"{len(df)}行")
    return (code6, "no")

# P0: 资金流 (8线程并行)
print("\n[P0] 资金流 7年日频...")
ff_todo = []
with open(os.path.join(PROJ, "data", "universe", "watchlist.json")) as f:
    for s in json.load(f)["watchlist"]:
        c6 = s["code"][:6]
        if not os.path.exists(os.path.join(PROJ, "data", "raw", "fund_flow", f"{c6}.parquet")):
            mkt = {"SH":"sh","SZ":"sz","BJ":"bj"}.get(s["code"][7:],"sh")
            ff_todo.append((c6, mkt))
print(f"  待拉: {len(ff_todo)} 只")
if ff_todo:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(fetch_fund_flow_single, c, m) for c, m in ff_todo]):
            c, r = f.result()
            print(f"    {c}: {r}")

# P2: 链主事件 (8线程并行)
print(f"\n[P2] 链主事件...")
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(fetch_chain_events, l) for l in leaders]):
        n, r = f.result()
        print(f"    {n}: {r}")

print(f"\n✅ 全部完成! 耗时 {time.time()-t0:.0f}s")
