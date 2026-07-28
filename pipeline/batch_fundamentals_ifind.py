"""iFinD 全量基本面拉取 — 198只 × 10年 × 5并发"""
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
FUNDA_DIR = str(cfg.settings.DATA_DIR / "raw" / "fundamentals")
WATCH_PATH = str(cfg.settings.DATA_DIR.parent / "data" / "universe" / "watchlist.json")
os.makedirs(FUNDA_DIR, exist_ok=True)

with open(WATCH_PATH, encoding="utf-8") as f:
    stocks = json.load(f)["watchlist"]
print(f"拉取 {len(stocks)} 只基本面, 5线程")

def parse_num(s: str) -> float:
    """解析中文单位数字 → float"""
    if not s or s == '\t' or s == '--':
        return None
    s = s.replace(',', '').replace('\t', '').strip()
    if not s:
        return None
    try:
        if '亿' in s: return float(s.replace('亿','')) * 1e8
        if '万' in s: return float(s.replace('万','')) * 1e4
        if '万亿' in s: return float(s.replace('万亿','')) * 1e12
        return float(s)
    except (ValueError, TypeError):
        return None

def fetch_funda(s: dict) -> tuple:
    c6 = s["code"][:6]
    
    # 1. 基本面（营收、利润、负债率、增长率）
    r = call("stock", "get_stock_info", {"query": f"{c6} 基本面"})
    recs = []
    if r.get("ok") and r.get("data"):
        try:
            d = json.loads(r["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                # 表头固定格式：0=空 1=代码 2=名称 3=日期 4=资产负债率 6=营收 8=净利润
                for line in lines[2:]:
                    if line.startswith('#') or line.startswith('(') or '---' in line:
                        continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 8:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            rec = {'date': ds, 'code': c6}
                            rec['revenue'] = parse_num(parts[6]) if len(parts) > 6 else None
                            rec['profit'] = parse_num(parts[8]) if len(parts) > 8 else None
                            rec['debt_ratio'] = parse_num(parts[4]) if len(parts) > 4 else None
                            recs.append(rec)
        except: pass
    
    # 2. 每股指标（EPS, BPS）
    r2 = call("stock", "get_stock_info", {"query": f"{c6} 每股指标"})
    eps_map, bps_map = {}, {}
    if r2.get("ok") and r2.get("data"):
        try:
            d = json.loads(r2["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                # 0=空 1=代码 2=名称 3=日期 ...
                for line in lines[2:]:
                    if line.startswith('#') or '---' in line: continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 10:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            eps_map[ds] = parse_num(parts[4]) if len(parts) > 4 else None
                            bps_map[ds] = parse_num(parts[8]) if len(parts) > 8 else None
        except: pass
    
    # 3. 净资产收益率
    r3 = call("stock", "get_stock_info", {"query": f"{c6} 净资产收益率"})
    roe_map = {}
    if r3.get("ok") and r3.get("data"):
        try:
            d = json.loads(r3["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                for line in lines[2:]:
                    if line.startswith('#') or '---' in line: continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 4:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            roe_map[ds] = parse_num(parts[4])
        except: pass
    
    # 4. 市盈率
    r4 = call("stock", "get_stock_info", {"query": f"{c6} 市盈率"})
    pe_map = {}
    if r4.get("ok") and r4.get("data"):
        try:
            d = json.loads(r4["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                for line in lines[2:]:
                    if line.startswith('#') or '---' in line: continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 4:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            pe_map[ds] = parse_num(parts[4])
        except: pass
    
    # 5. 总市值
    r5 = call("stock", "get_stock_info", {"query": f"{c6} 总市值"})
    mcap_map = {}
    if r5.get("ok") and r5.get("data"):
        try:
            d = json.loads(r5["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                for line in lines[2:]:
                    if line.startswith('#') or '---' in line: continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 4:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            mcap_map[ds] = parse_num(parts[4])
        except: pass
    
    # 6. 毛利率
    r6 = call("stock", "get_stock_info", {"query": f"{c6} 毛利率"})
    gm_map = {}
    if r6.get("ok") and r6.get("data"):
        try:
            d = json.loads(r6["data"]["result"]["content"][0]["text"])
            ans = d.get("data", {}).get("answer", "")
            lines = [l for l in ans.split('\n') if l.strip()]
            if lines:
                for line in lines[2:]:
                    if line.startswith('#') or '---' in line: continue
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 4:
                        ds = parts[3]
                        if ds.isdigit() and len(ds) == 8:
                            gm_map[ds] = parse_num(parts[4])
        except: pass
    
    # 合并数据
    for rec in recs:
        ds = rec['date']
        if ds in eps_map: rec['eps'] = eps_map[ds]
        if ds in bps_map: rec['bps'] = bps_map[ds]
        if ds in roe_map: rec['roe'] = roe_map[ds]
        if ds in pe_map: rec['pe'] = pe_map[ds]
        if ds in mcap_map: rec['mcap'] = mcap_map[ds]
        if ds in gm_map: rec['gross_margin'] = gm_map[ds]
    
    if recs:
        df = pd.DataFrame(recs)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
        df.to_parquet(os.path.join(FUNDA_DIR, f'{c6}.parquet'), index=False)
        return (c6, 'ok', len(df))
    return (c6, 'no', 0)

t0 = time.time()
ok_count = no_count = 0
total = len(stocks)
with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {pool.submit(fetch_funda, s): s for s in stocks}
    for f in as_completed(futures):
        s = futures[f]
        c6 = s['code'][:6]
        try:
            c, st, n = f.result()
            if st == 'ok':
                ok_count += 1
                if ok_count % 20 == 0:
                    print(f'  {ok_count}/{total}: {c} {n}行 ({int(time.time()-t0)}s)')
            else:
                no_count += 1
        except Exception as e:
            print(f'  {c6}: 异常 {e}')

elapsed = time.time() - t0
print(f'\n=== 完成! {ok_count}成功/{no_count}无数据 ({elapsed:.0f}s) ===')
print(f'速率: {ok_count/elapsed*60:.1f}只/分钟')
