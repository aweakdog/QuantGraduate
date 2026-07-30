"""扫特征数量, 用 IC 而非总收益做判据

直接回答"信息越多越好吗"。用修复后的确定性筛选(gain + colsample=1.0 +
区块重采样 + 稳定排序), 只改 --n-features, 看 IC 随特征数如何变化。

判据说明: 只持3只时总收益 t≈1, 没有区分力, 所以本脚本以 IC 及其 t 值
为主指标, 总收益仅作参考。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PY = str(ROOT / ".venv" / "bin" / "python")

ap = argparse.ArgumentParser()
ap.add_argument("--n-features", default="20,40,60,80,120")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

NS = [int(x) for x in args.n_features.split(",")]

BASE = [
    "--test-start", "2022-09-01", "--test-end", "2026-07-27",
    "--initial-capital", "20000", "--label", "5d",
    "--hold-days", "10", "--tranche-n", "3",
    "--portfolio-mode", "periodic", "--exec-mode", "t1close",
    "--slippage", "0.001",
    "--regime-filter", "breadth", "--regime-breadth", "0.35",
    "--regime-confirm", "1",
    "--train-file", "training_data_pit_v24.parquet",
    "--pit-universe", "universe_pit.parquet",
]

rows = []
for i, n in enumerate(NS, 1):
    tag = f"NF{n}"
    out = PROC / f"wf_daily_{tag}_ts2022-09-01_te2026-07-27_cap20000.json"
    cmd = [PY, str(ROOT / "scripts" / "wf_v35_breadth_alpha.py"), *BASE,
           "--n-features", str(n), "--tag", tag]
    print(f"[{i}/{len(NS)}] n_features={n} ...", flush=True)
    if args.dry_run:
        print("   " + " ".join(cmd))
        continue
    if not out.exists():
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        if r.returncode != 0:
            print(f"   失败: {r.stderr[-600:]}")
            continue
    d = json.loads(out.read_text(encoding="utf-8"))
    s = d["summary"]
    daily = pd.DataFrame(d["daily"])
    ic = daily["ic"].astype(float).dropna() if "ic" in daily else pd.Series(dtype=float)
    rows.append({
        "n_feat_req": n,
        "n_feat_act": d.get("features"),
        "IC": s["ic_mean"], "IC_t": s["ic_tstat"],
        "IC>0占比": round((ic > 0).mean() * 100, 1) if len(ic) else np.nan,
        "IR": s["information_ratio"],
        "ret%": s["total_return_pct"],
        "sharpe": s["sharpe"],
        "maxDD%": s["max_dd_pct"],
    })
    print(f"   IC {s['ic_mean']:+.4f} (t={s['ic_tstat']:.2f}) | "
          f"IR {s['information_ratio']:.2f} | 收益 {s['total_return_pct']:.1f}%")

if args.dry_run or not rows:
    sys.exit(0)

df = pd.DataFrame(rows)
print("\n" + "=" * 92)
print("特征数量 vs IC  (判据是 IC 及其 t 值, 不是总收益)")
print("=" * 92)
print(df.to_string(index=False))
df.to_csv(PROC / "n_features_sweep.csv", index=False)
print(f"\n已保存: {PROC / 'n_features_sweep.csv'}")

best = df.loc[df["IC_t"].idxmax()]
print(f"\nIC t值最高: n_features={int(best['n_feat_req'])} "
      f"-> IC {best['IC']:+.4f} (t={best['IC_t']:.2f})")
print("\n注: IC 的标准误约 0.008, 因此 IC 差异小于 0.016 (2SE) 不足以区分。")
