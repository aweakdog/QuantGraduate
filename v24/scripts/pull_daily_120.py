"""
盘后数据拉取 — 120池专用
1. 1分钟K线 (OHLCV+成交额) — via thsdk KQ2026
2. 主力资金净流入额 — via thsdk wencai
用法: python scripts/pull_daily_120.py [--date YYYY-MM-DD]
默认拉取当天日期
"""
import sys, json, time, os, argparse
from datetime import datetime
from pathlib import Path

BASE = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy\data")

def get_120_codes():
    """从训练数据提取120池代码"""
    try:
        import pandas as pd
        df = pd.read_parquet(BASE / "processed" / "training_data_v16.parquet")
        return sorted(df["code"].unique())
    except:
        # Fallback: read from watchlist
        with open(BASE / "universe" / "watchlist_top120.json") as f:
            d = json.load(f)
        codes = []
        for item in d["watchlist"]:
            if isinstance(item, dict) and "code" in item:
                codes.append(item["code"])
        return sorted(set(codes))

def gen_1min_script(codes, date_str):
    """生成thsdk 1分钟K线拉取脚本"""
    code_list_str = json.dumps(codes)
    date8 = date_str.replace("-", "")
    tmp_py = Path(os.environ.get("TEMP", "C:/tmp")) / f"pull_1min_{date8}.py"
    
    script = f'''
import sys, json, time
sys.path.insert(0, r"C:\\Users\\admin\\.workbuddy\\skills\\ths-all-in-one\\scripts")
from thsdk import THS
import pandas as pd

CODES = {code_list_str}
DATE = "{date_str}"
DATE8 = "{date8}"
targets = CODES

KQ = {{"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}}
all_rows = []
t0 = time.time()

with THS(KQ) as ths:
    for i, code in enumerate(targets):
        try:
            sym = ths.search_symbols(code)
            time.sleep(0.05)
            if not sym.success or not sym.data:
                continue
            cand = [d for d in sym.data if d.get("MarketStr","").startswith(("USZA","USHA","UBJA"))]
            if not cand:
                cand = sym.data
            ths_code = cand[0].get("THSCODE", "")
            
            k = ths.klines(ths_code, count=240, interval="1m")
            time.sleep(0.05)
            if k.success and k.data:
                for row in k.data:
                    t = str(row.get("时间",""))
                    all_rows.append({{
                        "code": code,
                        "time": t[11:16] if len(t) > 16 else t,
                        "open": row.get("开盘价"),
                        "close": row.get("收盘价"),
                        "high": row.get("最高价"),
                        "low": row.get("最低价"),
                        "vol": row.get("成交量", 0),
                        "amount": row.get("总金额", 0),
                        "date": DATE,
                    }})
            if (i+1) % 20 == 0:
                done = len(set(r["code"] for r in all_rows))
                elapsed = time.time() - t0
                print(f"  [{{i+1}}/{{len(targets)}}] {{done}} stocks, {{len(all_rows)}} bars, {{elapsed:.0f}}s", flush=True)
        except Exception as e:
            print(f"  SKIP {{code}}: {{e}}", flush=True)
            continue

if all_rows:
    df = pd.DataFrame(all_rows)
    out = r"{BASE}\\raw\\kline_1min\\{{DATE8}}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved: {{out}} ({{len(df)}} bars, {{df.code.nunique()}} stocks)", flush=True)
else:
    print("No data")
'''
    return tmp_py, script

def gen_fundflow_script(codes, date_str):
    """生成主力资金净流入脚本"""
    code_list_str = ",".join(codes[:50])  # 分批50个
    date8 = date_str.replace("-", "")
    tmp_py = Path(os.environ.get("TEMP", "C:/tmp")) / f"fundflow_{date8}.py"
    
    script = f'''
import sys, json, time
sys.path.insert(0, r"C:\\Users\\admin\\.workbuddy\\skills\\ths-all-in-one\\scripts")
from thsdk import THS
import pandas as pd

ALL_CODES = {json.dumps(codes)}
DATE = "{date_str}"
DATE8 = "{date8}"
BATCH = 50

KQ = {{"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}}
FIELDS = "主力资金流向,主力增仓占比,DDE大单净额"
FNAMES = ["main_force_net", "main_force_pct", "dde_net"]
FKEYS = ["主力资金流向", "主力增仓占比", "dde大单净额"]
rows = []

with THS(KQ) as ths:
    for i in range(0, len(ALL_CODES), BATCH):
        batch = ",".join(ALL_CODES[i:i+BATCH])
        try:
            r = ths.wencai_nlp(f"{{batch}} {{DATE}} {{FIELDS}}")
            time.sleep(0.3)
            if r.success and r.data:
                for rd in r.data:
                    c = rd.get("股票代码", "").split(".")[0]
                    if not c:
                        continue
                    rec = {{"date": DATE, "code": c}}
                    for fk, fn in zip(FKEYS, FNAMES):
                        v = rd.get(f"{{fk}}[{{DATE8}}]")
                        if v is not None:
                            try:
                                rec[fn] = float(str(v).replace(",", ""))
                            except:
                                rec[fn] = None
                        else:
                            rec[fn] = None
                    rows.append(rec)
            print(f"  [{{min(i+BATCH, len(ALL_CODES))}}/{{len(ALL_CODES)}}] {{len(rows)}} records", flush=True)
        except Exception as e:
            print(f"  SKIP batch {{i}}: {{e}}", flush=True)
            continue

if rows:
    df = pd.DataFrame(rows)
    out = r"{BASE}\\raw\\fund_flow_daily\\{{DATE8}}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved: {{out}} ({{len(df)}} records)", flush=True)
else:
    print("No data")
'''
    return tmp_py, script


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    
    THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
    
    print(f"=== 盘后数据拉取 120池 — {args.date} ===")
    codes = get_120_codes()
    print(f"股票池: {len(codes)} 只")
    
    # 1. 1-min K线
    print(f"\n--- 1. 1分钟K线 ---")
    tmp_path, script = gen_1min_script(codes, args.date)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(script)
    import subprocess
    r = subprocess.run([THS_PY, str(tmp_path)], capture_output=True, text=True, timeout=600)
    print(r.stdout.strip()[-500:])
    if r.returncode:
        print(f"ERR: {r.stderr[-300:]}")
    os.remove(tmp_path)
    
    # 2. 主力资金
    print(f"\n--- 2. 主力资金净流入 ---")
    tmp_path2, script2 = gen_fundflow_script(codes, args.date)
    with open(tmp_path2, "w", encoding="utf-8") as f:
        f.write(script2)
    r2 = subprocess.run([THS_PY, str(tmp_path2)], capture_output=True, text=True, timeout=300)
    print(r2.stdout.strip()[-500:])
    if r2.returncode:
        print(f"ERR: {r2.stderr[-300:]}")
    os.remove(tmp_path2)
    
    print(f"\nDone: {args.date}")
