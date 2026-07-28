"""
统一每日管线 — 数据拉取 → 合并 → 训练 → SuperMind上传 → 预测
一条命令全链路，支持 morning/evening/debug 模式
"""
import sys, subprocess, json, time
from pathlib import Path
from datetime import datetime

PROJECT = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy")
PY = r"C:\Users\admin\.workbuddy\binaries\python\versions\3.13.12\python.exe"
THS_PY = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"

SCRIPTS = {
    "pull":     str(PROJECT / "scripts" / "pull_daily_120.py"),
    "merge":    str(PROJECT / "scripts" / "merge_raw_data.py"),
    "supermind": str(PROJECT / "scripts" / "supermind_120pool.py"),
    "sm_upload": str(PROJECT / "scripts" / "sm_120_v2.py"),
    "pipeline": str(PROJECT / "scripts" / "daily_pipeline.py"),
    "wf":       str(PROJECT / "scripts" / "wf_daily_expanding.py"),
}

def run(script, desc, timeout=600):
    t0 = time.time()
    print(f"\n[{desc}]")
    r = subprocess.run([PY, script], capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    # Print last 5 meaningful lines
    for line in r.stdout.split("\n"):
        if line.strip() and ("Top" in line or "Sharpe" in line or "Done" in line or
                             "cum" in line or "LOAD_OK" in line or "PREDICT" in line or
                             "Model" in line or "数据" in line):
            print(f"  {line.strip()}")
    status = "OK" if ok else f"FAIL (rc={r.returncode})"
    print(f"  → {status} ({elapsed:.0f}s)")
    return ok

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "debug"
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"{'='*55}")
    print(f"  Unified Pipeline — {mode.upper()} — {today}")
    print(f"{'='*55}")

    if mode == "morning":
        # 08:30: 拉外盘 → 训练 → 预测
        pass  # External data pull not yet in pull_daily_120
        run(SCRIPTS["pipeline"], "Training+Prediction", timeout=600)

    elif mode == "evening":
        # 17:30: 拉A股 → 合并 → 训练+SuperMind
        run(SCRIPTS["pull"], "Data Pull (120 pool)", timeout=600)
        run(SCRIPTS["merge"], "Merge -> V17", timeout=120)
        run(SCRIPTS["supermind"], "Train+Serialize+SuperMind", timeout=300)

    elif mode == "debug":
        # 仅训练+SuperMind，跳过数据拉取
        run(SCRIPTS["supermind"], "Train+Serialize+SuperMind (no data pull)", timeout=300)

    elif mode == "backtest":
        # 跑WF回测
        run(SCRIPTS["wf"], "Walk-Forward Backtest", timeout=600)

    elif mode == "upload":
        # 仅SuperMind上传+预测（复用上次序列化）
        run(SCRIPTS["sm_upload"], "SuperMind Upload+Predict", timeout=60)

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python unified_pipeline.py [morning|evening|debug|backtest|upload]")

    print(f"\n{'='*55}")
    print(f"  Done: {mode.upper()}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
