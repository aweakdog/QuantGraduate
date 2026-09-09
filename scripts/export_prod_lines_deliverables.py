"""生产线回测面板 -> 桌面交付物 (与 08-23 "生产全配置" 批次同一套式样)

每条线:
  <批次>_<显示名>_<只数>只..._中位种子s<seed>_<起>至<止>.xlsx   操作 Excel (export_v35_excel, 含空仓走势图页)
  <批次>_<显示名>_..._空仓走势图.png                             上面那张图的独立文件
  <批次>_<显示名>(<只数>只)_20种子对比图.png                      20 个单种子净值 + 中位种子加粗 + 生产 5 种子集成虚线
外加 export_prod_lines_excel.py 的六线汇总表 (同目录).

前置: 040 的 wf_daily_PL_*_te<te>_*.json / wf_daily_SEK_*_k5_0_*_te<te>_*.json 已 rsync 到 data/processed/,
      基准用服务器当前矩阵的瘦身副本 data/processed/training_data_pit_v24_tick1__bench_slim.parquet
      (ssh eez041 导出: read_parquet(columns=[date,code,fwd_1d_ret]).to_parquet(...)).

用法:
  python scripts/export_prod_lines_deliverables.py                      # 最新 te, 输出 ~/Desktop/l量化操作表/<今天>/
  python scripts/export_prod_lines_deliverables.py --te 2026-09-04 --outdir /路径
"""
import argparse
import glob
import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DESKTOP_ROOT = Path.home() / "Desktop" / "l量化操作表"
BENCH_SLIM = "training_data_pit_v24_tick1__bench_slim.parquet"
sys.path.insert(0, str(ROOT / "scripts"))
from export_v35_excel import export as export_ops_excel  # noqa: E402
from export_prod_lines_excel import build as build_summary_excel  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BATCH = "生产线最新"
# (线, 显示名, 文件名里的配置短语, 图标题里的只数短语)
LINES = [("aggr5w", "激进5万", "3只行业限2仅主板T1A", "3只·行业限2"),
         ("aggr10w", "激进10万", "3只行业限2仅主板T1A", "3只·行业限2"),
         ("steady5w", "稳妥5万", "5只仅主板T1B", "5只"),
         ("fyf100w", "FYF实盘100万", "8只仅主板T1B", "8只"),
         ("aggr2w", "激进2万", "2只仅主板基线80", "2只"),
         ("steady2w", "稳妥2万", "3只行业限2全市场基线80", "3只·行业限2·全市场")]
CFG_LINE = "tick1矩阵 · 广度择时 · min-pred 0.002 · FDRR8 · T1A/T1B 分点 · 20bp 滑点 · 2022-09 起 expanding"


def latest_te():
    tes = {m.group(1) for p in PROC.glob("wf_daily_PL_*_te*.json")
           if (m := re.search(r"_te(\d{4}-\d{2}-\d{2})_", p.name))}
    if not tes:
        raise SystemExit(f"{PROC} 下没有 wf_daily_PL_*.json, 先从 040 同步")
    return max(tes)


def nav(d):
    daily = d["daily"]
    return [r["date"] for r in daily], [r["portfolio_value"] / d["initial_capital"] for r in daily]


def seed_chart(line, disp, npos, runs, med_seed, prod, out: Path, te: str):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=130)
    dates = None
    for s, d in runs.items():
        dates, eq = nav(d)
        if s == med_seed:
            ax.plot(dates, eq, lw=2.4, color="#d62728", zorder=5,
                    label=f"中位种子 s{s} ({d['summary']['total_return_pct']:+.1f}%)")
        else:
            ax.plot(dates, eq, lw=0.8, alpha=0.45)
    if prod is not None:
        pdates, peq = nav(prod)
        ax.plot(pdates, peq, lw=1.8, color="black", ls="--", zorder=6,
                label=f"生产 5 种子集成 ({prod['summary']['total_return_pct']:+.1f}%)")
    rets = [d["summary"]["total_return_pct"] for d in runs.values()]
    neg = sum(r <= 0 for r in rets)
    ax.set_title(f"{disp}({npos}) 生产线到最新日 20种子净值对比 2023-09 ~ {te[:7]}   "
                 f"区间 {min(rets):+.0f}% ~ {max(rets):+.0f}%, {'无一亏损' if neg == 0 else f'{neg} 个亏损'}\n"
                 f"({CFG_LINE})", fontsize=12)
    ax.set_ylabel("净值 (期初=1)")
    step = max(1, len(dates) // 10)
    ax.set_xticks(dates[::step])
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.axhline(1.0, color="#999", lw=0.6, ls="--")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--bench-file", default=BENCH_SLIM,
                    help="算基准的矿(data/processed 下), 缺省用服务器当前矩阵的瘦身副本")
    a = ap.parse_args()
    te = a.te or latest_te()
    outdir = a.outdir or DESKTOP_ROOT / datetime.now().strftime("%Y-%m-%d")
    outdir.mkdir(parents=True, exist_ok=True)
    if not (PROC / a.bench_file).exists():
        raise SystemExit(f"缺 {PROC / a.bench_file}: 先从 eez041 导一份 date/code/fwd_1d_ret 瘦身矩阵")

    for line, disp, cfg_words, npos in LINES:
        files = sorted(PROC.glob(f"wf_daily_PL_{line}_s*_te{te}_*.json"))
        if not files:
            print(f"!! {line}: 没有 PL 结果, 跳过"); continue
        runs = {int(re.search(r"_s(\d+)_ts", f.name).group(1)): json.loads(f.read_text(encoding="utf-8")) for f in files}
        paths = {int(re.search(r"_s(\d+)_ts", f.name).group(1)): f for f in files}
        rets = {s: d["summary"]["total_return_pct"] for s, d in runs.items()}
        med_val = st.median(rets.values())
        med_seed = min(rets, key=lambda s: abs(rets[s] - med_val))   # 离面板中位最近的那个种子
        pf = sorted(PROC.glob(f"wf_daily_SEK_{line}_k5_0_s42_*_te{te}_*.json"))
        prod = json.loads(pf[0].read_text(encoding="utf-8")) if pf else None
        sm = runs[med_seed]["summary"]
        print(f"[{line}] {len(runs)} 种子, 面板中位 {med_val:+.1f}% -> 中位种子 s{med_seed} "
              f"{sm['total_return_pct']:+.1f}% 夏普 {sm['sharpe']:.2f} 回撤 {sm['max_dd_pct']:.1f}%"
              + (f" | 生产5种子 {prod['summary']['total_return_pct']:+.1f}%" if prod else ""))
        span = f"2023-09至{te[:7]}"
        xlsx = outdir / f"{BATCH}_{disp}_{cfg_words}20bp_中位种子s{med_seed}_{span}.xlsx"
        export_ops_excel(paths[med_seed], xlsx, a.bench_file)
        png = outdir / f"{BATCH}_{disp}({npos})_20种子对比图.png"
        seed_chart(line, disp, npos, runs, med_seed, prod, png, te)
        print(f"  20种子对比图: {png.name}")

    summary_xlsx = outdir / f"生产线回测_te{te}.xlsx"
    build_summary_excel(te, summary_xlsx)
    print(f"  六线汇总: {summary_xlsx.name}")
    print(f"\n交付目录 {outdir}:")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
