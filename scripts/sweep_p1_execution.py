"""P1: 用缓存预测扫执行层参数, 验证"提高广度能否让微弱信号稳定表达"

背景: IC≈0.029 但只持3只时 IR 理论上限仅 0.23, 且实测收益由 3~5 笔交易主导。
本脚本在【同一份缓存预测】上只改执行层参数, 因此:
  - 模型/特征/训练完全不变, 排除建模噪声
  - 单次运行几十秒, 可以密集扫参

关注指标不只是收益, 更重要的是【稳健性】:
  - top5_pnl_share: 最赚5笔占总盈亏的比例, 越低越健康
  - ex_top5_return: 剔除最赚5笔后的收益, 这才是"可重复"的部分
  - IR 是否向理论上限靠拢
"""
import argparse
import itertools
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PY = str(ROOT / ".venv" / "bin" / "python")

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="preds_P1BASE_oldfeats.pkl",
                help="缓存预测 pickle (data/processed/ 下)")
ap.add_argument("--features-from",
                default="wf_daily_em_t1close_s001_ts2022-09-01_te2026-07-27_cap20000.json")
ap.add_argument("--positions", default="3,5,8,10,15",
                help="要扫的持仓只数 (periodic 模式下即 tranche_n)")
ap.add_argument("--hold-days", default="10",
                help="要扫的持有天数, 逗号分隔")
ap.add_argument("--regimes", default="breadth,off",
                help="要扫的择时开关")
ap.add_argument("--capitals", default="20000,100000",
                help="要扫的本金, 逗号分隔。缓存 meta 不校验本金, 因此可用同一份"
                     "预测对比不同资金规模 —— 用于隔离'信号弱'与'买不起一手'")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

POSITIONS = [int(x) for x in args.positions.split(",")]
HOLDS = [int(x) for x in args.hold_days.split(",")]
REGIMES = args.regimes.split(",")
CAPITALS = [float(x) for x in args.capitals.split(",")]

BASE = [
    "--test-start", "2022-09-01", "--test-end", "2026-07-27",
    "--label", "5d", "--portfolio-mode", "periodic",
    "--exec-mode", "t1close", "--slippage", "0.001",
    "--train-file", "training_data_pit_v24.parquet",
    "--pit-universe", "universe_pit.parquet",
    "--features-from", args.features_from,
    "--load-preds", args.preds,
]


def round_trips(trades):
    books = defaultdict(deque)
    rts = []
    for t in sorted(trades, key=lambda x: x["date"]):
        code, sh = t["code"], t["shares"]
        if t["action"] == "buy":
            books[code].append({"shares": sh, "cost": -t["net"]})
        else:
            proceeds, remain = t["net"], sh
            while remain > 0 and books[code]:
                lot = books[code][0]
                take = min(remain, lot["shares"])
                fc = lot["cost"] * take / lot["shares"]
                fp = proceeds * take / sh
                rts.append(fp - fc)
                lot["shares"] -= take
                lot["cost"] -= fc
                remain -= take
                if lot["shares"] <= 0:
                    books[code].popleft()
    return np.array(sorted(rts, reverse=True))


rows = []
combos = list(itertools.product(CAPITALS, HOLDS, POSITIONS, REGIMES))
print(f"共 {len(combos)} 个组合待跑\n")

for i, (cap, hold, pos, reg) in enumerate(combos, 1):
    tag = f"P1SW_h{hold}_n{pos}_{reg}"
    out = PROC / f"wf_daily_{tag}_ts2022-09-01_te2026-07-27_cap{int(cap)}.json"
    cmd = [PY, str(ROOT / "scripts" / "wf_v35_breadth_alpha.py"), *BASE,
           "--initial-capital", str(int(cap)),
           "--hold-days", str(hold), "--tranche-n", str(pos),
           "--regime-filter", reg, "--tag", tag]
    if reg == "breadth":
        cmd += ["--regime-breadth", "0.35", "--regime-confirm", "1"]

    print(f"[{i}/{len(combos)}] cap={int(cap)} hold={hold} pos={pos} regime={reg} ...",
          flush=True)
    if args.dry_run:
        print("   " + " ".join(cmd))
        continue
    if not out.exists():
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        if r.returncode != 0:
            print(f"   失败: {r.stderr[-500:]}")
            continue
    d = json.loads(out.read_text(encoding="utf-8"))
    s = d["summary"]
    rt = round_trips(d["trades"])
    tot = rt.sum() if len(rt) else np.nan
    top5 = rt[:5].sum() if len(rt) >= 5 else np.nan
    rej = s.get("reject_breakdown", {}) or {}
    rows.append({
        "cap": int(cap), "hold": hold, "posTgt": pos, "regime": reg,
        "avgHold": s["avg_holdings"],
        "rejLot": rej.get("buy_lot_too_big", 0),
        "ret%": s["total_return_pct"],
        "excess%": s["excess_annual_pct"],
        "IR": s["information_ratio"],
        "sharpe": s["sharpe"],
        "maxDD%": s["max_dd_pct"],
        "cost%": s["total_cost_pct"],
        "trades": s["n_trades"],
        "winRate%": round((rt > 0).mean() * 100, 1) if len(rt) else np.nan,
        "top5share%": round(top5 / tot * 100, 1) if tot and not np.isnan(top5) else np.nan,
        "exTop5%": round((tot - top5) / cap * 100, 1) if tot and not np.isnan(top5) else np.nan,
        "deployed%": s["avg_deployed_pct"],
    })
    print(f"   收益 {s['total_return_pct']:>7.1f}% | IR {s['information_ratio']:>5.2f} | "
          f"实际持仓 {s['avg_holdings']:>4.1f} | 买不起一手拒单 {rows[-1]['rejLot']:>5} | "
          f"成本 {s['total_cost_pct']:>4.1f}% | top5占比 {rows[-1]['top5share%']}%")

if args.dry_run or not rows:
    sys.exit(0)

df = pd.DataFrame(rows)
print("\n" + "=" * 110)
print("P1 执行层扫描结果 (同一份缓存预测, 模型完全一致)")
print("=" * 110)
print(df.to_string(index=False))

csv = PROC / "p1_execution_sweep.csv"
df.to_csv(csv, index=False)
print(f"\n已保存: {csv}")

print("\n--- 解读要点 ---")
print("1. top5share% 下降 = 收益不再靠少数几笔, 策略变稳健")
print("2. exTop5% 上升 = 可重复的收益部分变多 (这比 ret% 更可信)")
print("3. cost% 随持仓数上升 = ¥5最低佣金的代价")
print("4. 若 IR 随 pos 上升并接近 IC*sqrt(N) 上限, 则广度假设成立")
