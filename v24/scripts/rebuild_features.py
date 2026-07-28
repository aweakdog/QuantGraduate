"""
重建特征矩阵 (v3补数后): v15(build_all) -> v22=v15 -> v23(build_v23)
- feature_engine.build_all 读扩展后的 raw/kline, 产出含最新真实特征的 v15
- v22 原 = v15 + 占位行; 现 build_all 已覆盖真实特征, 故 v22=v15
- build_v23 读 v22 加 vol_ma20 -> v23 (Web 自动选最新 vXX)
用法: python scripts/rebuild_features.py [--progress] [--no-incremental] [--workers N]

--progress: 输出 PROGRESS:N/M 行供 UI 解析
--no-incremental: 禁用增量跳过, 全量重建
--workers N: 并行线程数 (默认 16)
"""
import subprocess, sys, time
from pathlib import Path
import pandas as pd

PROJECT = Path(r"D:\myAI\WorkBuddy-workspace\quant-strategy")
PY = r"C:\Users\admin\.workbuddy\binaries\python\versions\3.13.12\python.exe"


def main():
    progress = "--progress" in sys.argv
    no_inc = "--no-incremental" in sys.argv
    workers = 16
    for i, a in enumerate(sys.argv):
        if a == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])

    t0 = time.time()
    if progress:
        print("PROGRESS:0/3", flush=True)
    print(f"[rebuild] 1/3 feature_engine.build_all -> v15 (incremental={not no_inc}, workers={workers})", flush=True)
    r = subprocess.run(
        [PY, "-m", "pipeline.feature_engine",
         "--incremental" if not no_inc else "--no-incremental",
         "--workers", str(workers)],
        cwd=str(PROJECT))
    if r.returncode != 0:
        print("BUILD_ALL FAILED rc=", r.returncode)
        sys.exit(1)
    v15 = pd.read_parquet(PROJECT / "data" / "processed" / "training_data_v15.parquet")
    print(f"  v15: {len(v15):,} rows, {v15['code'].nunique()} codes, {len(v15.columns)} cols, max={v15['date'].max().date()}")
    if progress:
        print("PROGRESS:1/3", flush=True)

    print("[rebuild] 2/3 v22 = v15 (占位行已由build_all真实覆盖)", flush=True)
    v15.to_parquet(PROJECT / "data" / "processed" / "training_data_v22.parquet", index=False)
    if progress:
        print("PROGRESS:2/3", flush=True)

    print("[rebuild] 3/3 build_v23.py -> v23", flush=True)
    r = subprocess.run([PY, str(PROJECT / "scripts" / "build_v23.py")], cwd=str(PROJECT))
    if r.returncode != 0:
        print("BUILD_V23 FAILED rc=", r.returncode)
        sys.exit(1)
    v23 = pd.read_parquet(PROJECT / "data" / "processed" / "training_data_v23.parquet")
    print(f"  v23: {len(v23):,} rows, {v23['code'].nunique()} codes, {len(v23.columns)} cols, max={v23['date'].max().date()}")
    if progress:
        print("PROGRESS:3/3", flush=True)

    # 健全性: 最新特征日应无全NaN
    last = v23[v23["date"] == v23["date"].max()]
    nan_cols = last.isna().all(axis=0).sum()
    print(f"  最新日({v23['date'].max().date()}) 全NaN列数: {nan_cols}")
    print(f"[rebuild] 完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
