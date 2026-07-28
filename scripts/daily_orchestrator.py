"""
每日全流程编排 — morning/evening 两时段
08:30 (盘前): 拉外盘数据 → 训练 → 预测今日T+1
17:30 (盘后): 拉A股1min+资金流 → 训练 → 预测明日T+1 → 准备收评数据
"""
import sys, subprocess, json, time, os
from pathlib import Path
from datetime import datetime

PROJECT = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy")
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
MAIN_PY = r"C:\Users\admin\.workbuddy\binaries\python\versions\3.13.12\python.exe"
ROUTER = PROJECT / "scripts" / "daily_pipeline.py"
PULL_120 = PROJECT / "scripts" / "pull_daily_120.py"

SLOT = sys.argv[1] if len(sys.argv) > 1 else "evening"
today = datetime.now().strftime("%Y-%m-%d")
t0 = datetime.now()

print(f"\n{'='*60}")
print(f"  每日全流程 — {SLOT.upper()} — {today}")
print(f"{'='*60}")

# ── Step 1: 数据拉取 ──
print(f"\n[Step 1] 数据拉取...")

if SLOT == "morning":
    # 08:30: 拉外盘/隔夜数据
    print(f"  盘前时段：拉外盘数据...")
    # Pull US overnight
    r = subprocess.run([MAIN_PY, str(PROJECT / "pipeline" / "pull_us_overnight.py")],
                       capture_output=True, text=True, timeout=300)
    print(f"  US overnight: {'OK' if r.returncode==0 else 'FAIL'}")
    # Pull commodities
    r = subprocess.run([MAIN_PY, str(PROJECT / "pipeline" / "commodity_signal.py")],
                       capture_output=True, text=True, timeout=300)
    print(f"  Commodity: {'OK' if r.returncode==0 else 'FAIL'}")
    # Pull macro (yields, FX)
    r = subprocess.run([MAIN_PY, str(PROJECT / "pipeline" / "pull_akshare_events.py")],
                       capture_output=True, text=True, timeout=120)
    print(f"  Macro: {'OK' if r.returncode==0 else 'FAIL'}")

elif SLOT == "evening":
    # 17:30: 拉A股盘后数据
    print(f"  盘后时段：拉A股数据...")
    r = subprocess.run([MAIN_PY, str(PULL_120), "--date", today],
                       capture_output=True, text=True, timeout=600)
    # Print last 3 lines of output
    lines = [l for l in r.stdout.split("\n") if l.strip()]
    for l in lines[-3:]:
        print(f"  {l}")
    print(f"  {'OK' if r.returncode==0 else 'FAIL'}")

print(f"  数据拉取完成 ({ (datetime.now()-t0).total_seconds():.0f}s)")

# ── Step 2: 训练+预测 ──
print(f"\n[Step 2] 训练+预测...")
r = subprocess.run([MAIN_PY, str(ROUTER)], capture_output=True, text=True, timeout=600)
print(r.stdout[-1000:])  # Last 1000 chars for summary
print(f"  训练完成 ({ (datetime.now()-t0).total_seconds():.0f}s)")

# ── Step 3: 结果检查 ──
daily_out = PROJECT / "data" / "daily_output"
latest = sorted(daily_out.glob("*.json"))[-1] if daily_out.exists() and list(daily_out.glob("*.json")) else None
if latest:
    with open(str(latest)) as f:
        report = json.load(f)
    top3 = report.get("top10", [])[:3]
    print(f"\n  Top3:")
    for i, s in enumerate(top3, 1):
        print(f"    #{i}  {s['code']}  {s['score']:+.4f}")

total = (datetime.now() - t0).total_seconds()
print(f"\n  总耗时: {total:.0f}s")
print(f"  Done: {SLOT}")
