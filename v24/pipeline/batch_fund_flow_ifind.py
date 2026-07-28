"""iFinD 个人版 5并发 — 补齐 2015-2019 基金流缺口
已有 2020-2026 数据，追加 2015-2019"""
import os, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

os.chdir(r"C:\Users\admin\AppData\Local\hermes\skills\ifind-finance-data")
sys.path.insert(0, r"C:\Users\admin\AppData\Local\hermes\skills\ifind-finance-data")
from call import call

# Config
import importlib
spec = importlib.util.spec_from_file_location("config", r"D:\myAI\Hermes-Workspace\quant-strategy\pipeline\config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
FF_DIR = str(cfg.settings.DATA_DIR / "raw" / "fund_flow")
WATCH_PATH = str(cfg.settings.DATA_DIR.parent / "data" / "universe" / "watchlist.json")
os.makedirs(FF_DIR, exist_ok=True)

with open(WATCH_PATH, encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]
print(f"补齐 {len(stocks)} 只 2015-2019 基金流, 5线程并发")

def query_daily(c6: str, year: int) -> list:
    """查询单只单年每日主力净流入额"""
    try:
        r = call("stock", "get_stock_info", {"query": f"{c6} {year} 每日主力净流入额"})
        if not r.get("ok") or not r.get("data"):
            return []
        d = json.loads(r["data"]["result"]["content"][0]["text"])
        ans = d.get("data", {}).get("answer", "")
        recs = []
        for line in ans.split("\n"):
            if f"|{c6}" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            ds = parts[3]
            if not (ds.isdigit() and len(ds) == 8 and ds.startswith("20")):
                continue
            ns = parts[4]
            ps = parts[5]
            # 解析净额
            if "亿" in ns:
                nv = float(ns.replace("亿", "")) * 1e8
            elif "万" in ns:
                nv = float(ns.replace("万", "")) * 1e4
            else:
                try:
                    nv = float(ns.replace(",", "").replace("\t", ""))
                except (ValueError, TypeError):
                    continue
            # 解析净率
            try:
                pv = float(ps.replace(",", "").replace("\t", ""))
            except (ValueError, TypeError):
                pv = None
            if abs(nv) > 100:  # 过滤噪音
                recs.append({"date": ds, "main_force_net": nv, "main_force_pct": pv})
        return recs
    except (KeyError, json.JSONDecodeError, IndexError, TypeError, ValueError, Exception) as e:
        return []

def fetch(s: dict) -> tuple:
    c6 = s["code"][:6]
    out = os.path.join(FF_DIR, f"{c6}.parquet")
    
    # 检查是否已有 2015-2019 数据
    if os.path.exists(out):
        df = pd.read_parquet(out)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.replace('-',''), format='%Y%m%d', errors='coerce')
        existing_years = set(df['date'].dt.year)
        missing_years = [y for y in range(2015, 2020) if y not in existing_years]
        if not missing_years:
            return (c6, "skip", 0)
    else:
        missing_years = list(range(2015, 2020))
    
    # 拉缺失年份
    all_recs = []
    for yr in missing_years:
        recs = query_daily(c6, yr)
        all_recs.extend(recs)
        time.sleep(0.2)
    
    if not all_recs:
        return (c6, "no", 0)
    
    df_new = pd.DataFrame(all_recs)
    df_new["date"] = pd.to_datetime(df_new["date"], format="%Y%m%d")
    
    if os.path.exists(out):
        df = pd.read_parquet(out)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.replace('-',''), format='%Y%m%d', errors='coerce')
        combined = pd.concat([df, df_new])
        combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    else:
        combined = df_new.sort_values("date").reset_index(drop=True)
    
    combined.to_parquet(out, index=False)
    return (c6, "ok", len(combined))

t0 = time.time()
ok_count = no_count = err_count = 0
with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(fetch, s): s for s in stocks}
    for f in as_completed(futures):
        try:
            c, st, n = f.result()
            if st == "ok":
                ok_count += 1
                print(f"  {c}: {n}行 ({time.time()-t0:.0f}s)")
            else:
                no_count += 1
                if no_count <= 10:
                    print(f"  {c}: 无数据")
        except Exception as e:
            err_count += 1
            print(f"  {futures[f]['code'][:6]}: 异常 {e}")

elapsed = time.time() - t0
print(f"\n=== 完成! {ok_count}成功/{no_count}无数据/{err_count}异常 ({elapsed:.0f}s) ===")
print(f"速率: {ok_count/elapsed*60:.1f}只/分钟")
