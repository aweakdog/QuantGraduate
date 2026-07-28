"""v35 配置扫描: 自动依次跑多组配置, 汇总成排行榜

选优标准 (按优先级):
  1. 两个半段都跑赢基准 (稳健性优先, 避免在测试集上过拟合挑参)
  2. 年化 alpha (剔除 beta 后的真实超额)
  3. 信息比率 IR

用法: python scripts/run_v35_sweep.py
"""
import json, subprocess, sys, os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python3")
SCRIPT = str(ROOT / "scripts" / "wf_v35_breadth_alpha.py")
PROC = ROOT / "data" / "processed"
LOG = ROOT / "data" / "processed" / "v35_sweep_log.txt"
BOARD = ROOT / "data" / "processed" / "v35_sweep_leaderboard.json"

TS, TE = "2022-09-01", "2026-07-16"

# (tag, 说明, 额外参数)
CONFIGS = [
    ("c1_base",        "基线 5d/持有5/10只 L2回归",
     ["--hold-days", "5", "--tranche-n", "2"]),
    ("c2_style",       "+ C风格中性化",
     ["--hold-days", "5", "--tranche-n", "2", "--neutralize-style"]),
    ("c3_rank",        "+ B排序目标",
     ["--hold-days", "5", "--tranche-n", "2", "--objective", "rank"]),
    ("c4_rank_style",  "+ B+C",
     ["--hold-days", "5", "--tranche-n", "2", "--objective", "rank", "--neutralize-style"]),
    # B/C 已被证伪(c3/c4 显著变差), 以下扫 c1 家族的 换手 x 广度 前沿
    ("d1_hold3_n9",    "持有3/9只 (换手更高)",
     ["--hold-days", "3", "--tranche-n", "3"]),
    ("d2_hold5_n15",   "持有5/15只 (加广度)",
     ["--hold-days", "5", "--tranche-n", "3"]),
    ("d3_hold10_n20",  "持有10/20只 (换手减半)",
     ["--hold-days", "10", "--tranche-n", "2"]),
    ("d4_hold10_n10",  "持有10/10只 (换手减半且不挤仓位)",
     ["--hold-days", "10", "--tranche-n", "1"]),
    ("d5_hold20_n20",  "持有20/20只 (最低换手)",
     ["--hold-days", "20", "--tranche-n", "1"]),
    ("d6_base_voltgt", "基线 + 波动率目标仓位 (控回撤)",
     ["--hold-days", "5", "--tranche-n", "2", "--vol-target"]),
]


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def out_path(tag):
    return PROC / f"wf_daily_{tag}_ts{TS}_te{TE}_cap100000.json"


def run(tag, desc, extra):
    p = out_path(tag)
    if p.exists():
        log(f"跳过 {tag} ({desc}) — 已存在")
        return json.loads(p.read_text())
    log(f"开始 {tag}: {desc}")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    t0 = datetime.now()
    r = subprocess.run([PY, SCRIPT, "--test-start", TS, "--test-end", TE,
                        "--tag", tag] + extra,
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    dt = (datetime.now() - t0).total_seconds()
    if r.returncode != 0:
        log(f"  失败 {tag} ({dt:.0f}s): {r.stderr[-500:]}")
        return None
    if not p.exists():
        log(f"  异常 {tag}: 未生成结果文件")
        return None
    res = json.loads(p.read_text())
    s = res["summary"]
    log(f"  完成 {tag} ({dt/60:.1f}min) | 总收益 {s['total_return_pct']:+.1f}% "
        f"| 年化超额 {s['excess_annual_pct']:+.1f}% | alpha {s['alpha_annual_pct']:+.1f}% "
        f"| IR {s['information_ratio']} | 两段都赢 {s.get('beat_both_halves')}")
    return res


def leaderboard(rows):
    rows = sorted(rows, key=lambda x: (x["两段都赢"] == "是", x["年化alpha"], x["IR"]), reverse=True)
    w = [16, 28, 9, 9, 9, 7, 7, 8, 9]
    hdr = ["配置", "说明", "总收益%", "基准%", "年化超额", "alpha", "IR", "两段都赢", "成本%"]
    line = "  ".join(h.ljust(x) for h, x in zip(hdr, w))
    print("\n" + "=" * len(line))
    print("  v35 配置排行榜 (按 两段稳健 > alpha > IR 排序)")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for r in rows:
        vals = [r["配置"], r["说明"], f"{r['总收益']:+.1f}", f"{r['基准']:+.1f}",
                f"{r['年化超额']:+.1f}", f"{r['年化alpha']:+.1f}", f"{r['IR']:.2f}",
                r["两段都赢"], f"{r['成本']:.1f}"]
        print("  ".join(str(v).ljust(x) for v, x in zip(vals, w)))
    print("=" * len(line))
    return rows


def main():
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== v35 sweep 第二轮 {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    log(f"共 {len(CONFIGS)} 组配置, 预计 {len(CONFIGS)*20} 分钟")
    rows = []
    for tag, desc, extra in CONFIGS:
        res = run(tag, desc, extra)
        if res is None:
            continue
        s = res["summary"]
        rows.append({
            "配置": tag, "说明": desc,
            "总收益": s["total_return_pct"], "基准": s["benchmark_total_pct"],
            "年化超额": s["excess_annual_pct"], "年化alpha": s["alpha_annual_pct"],
            "IR": s["information_ratio"], "beta": s["beta"],
            "两段都赢": "是" if s.get("beat_both_halves") else "否",
            "成本": s["total_cost_pct"], "IC_t": s["ic_tstat"],
            "夏普": s["sharpe"], "最大回撤": s["max_dd_pct"],
            "持仓": s["avg_holdings"], "成交笔数": s["n_trades"],
            "stability": res.get("stability", []),
            "file": str(out_path(tag)),
        })
        leaderboard(rows)
        BOARD.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    if not rows:
        log("没有任何配置成功")
        return
    best = leaderboard(rows)[0]
    log(f"\n最优配置: {best['配置']} ({best['说明']})")
    log(f"  年化alpha {best['年化alpha']:+.1f}% | IR {best['IR']} | 两段都赢 {best['两段都赢']}")
    log(f"  结果文件 {best['file']}")

    # 给最优配置导出 Excel
    log("导出最优配置的操作 Excel...")
    r = subprocess.run([PY, str(ROOT / "scripts" / "export_v35_excel.py"), best["file"]],
                       capture_output=True, text=True, cwd=str(ROOT))
    log(r.stdout.strip() or r.stderr[-300:])
    BOARD.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    log("全部完成")


if __name__ == "__main__":
    main()
