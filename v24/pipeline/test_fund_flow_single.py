"""测试: 单只股票近3个月资金流 — 验证 wencai_nlp 批量查全部指标"""
import sys, time, json
from datetime import datetime, timedelta
from pathlib import Path

from pipeline.config import settings

THS_PY = settings.THS_PYTHON
DATA_DIR = settings.FUND_FLOW_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 测试股票
STOCK = "300308.SZ"  # 中际旭创
# 近3个月
END = datetime.now()
START = END - timedelta(days=90)

# 生成交易日（简化：周一到周五）
dates = []
d = START
while d <= END:
    if d.weekday() < 5:
        dates.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)
print(f"交易日: {len(dates)} ({dates[0]} ~ {dates[-1]})", flush=True)

# 测试1: 批量查全部指标 vs 逐指标查
code_test = f"""
from thsdk import THS
import time, json

KQ = {{"username": settings.THS_USERNAME, "password": settings.THS_PASSWORD}}

# 先测批量查询
print("=== BATCH QUERY ===", flush=True)
with THS(KQ) as ths:
    # 一次查全部资金流指标
    r = ths.wencai_nlp("{STOCK} 2026-04-01 主力资金流向,主力增仓占比,dde大单净额,融资融券余额,资金流向(万元)")
    if r.success and r.data:
        print(f"BATCH OK: type={{type(r.data).__name__}}", flush=True)
        df = r.df
        print(f"Columns: {{list(df.columns)}}", flush=True)
        print(f"Shape: {{df.shape}}", flush=True)
        print(f"Row 0: {{dict(df.iloc[0])}}", flush=True)
    else:
        print(f"BATCH FAILED: {{r.error}}", flush=True)

    print("", flush=True)
    print("=== PER-METRIC QUERY ===", flush=True)
    # 逐指标查（原有方式）
    for mn in ["主力资金流向", "主力增仓占比", "dde大单净额", "融资融券余额", "资金流向(万元)"]:
        r = ths.wencai_nlp(f"{{STOCK}} 2026-04-01 {{mn}}")
        if r.success and r.data:
            df = r.df
            cols = [c for c in df.columns if c not in ("股票代码","股票简称","最新价","最新涨跌幅")]
            if cols:
                v = str(df[cols[0]].iloc[0])
                print(f"  {{mn}}: {{v}}", flush=True)
        time.sleep(0.3)
    print("=== DONE ===", flush=True)
"""

print("运行测试...", flush=True)
import subprocess
r = subprocess.run([THS_PY, "-c", code_test], capture_output=True, text=True, timeout=30)
print("STDOUT:", r.stdout, flush=True)
if r.stderr:
    print("STDERR:", r.stderr[:500], flush=True)
print("RC:", r.returncode, flush=True)
