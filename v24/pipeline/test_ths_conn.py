"""快速测试: THS连接 + wencai_nlp 单次查询"""
import sys, time, json
sys.stdout.reconfigure(line_buffering=True)

print("=== START ===", flush=True)

from thsdk import THS
print("import OK", flush=True)

KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
print("connecting...", flush=True)

with THS(KQ) as ths:
    print("connected!", flush=True)
    r = ths.wencai_nlp("300308.SZ 2026-04-01 主力资金流向,主力增仓占比,dde大单净额")
    if r.success and r.data:
        print(f"OK: {len(r.data)} rows", flush=True)
        print(f"cols: {r.df.columns.tolist()}", flush=True)
    else:
        print(f"FAIL: {r.error}", flush=True)

print("=== DONE ===", flush=True)
