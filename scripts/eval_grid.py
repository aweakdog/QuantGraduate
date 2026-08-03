"""多种子 x 双窗口 评估框架 —— 让"这个配置好"变成可检验的陈述

为什么需要这个
──────────────
1) 单次回测的收益是随机数。3 只持仓下, 20 个随机种子的总收益从 -31% 到 +190%,
   而它们的 IC 全部落在 0.0133~0.0170 这个极窄区间。也就是说同样的预测能力,
   收益能差 220 个百分点。任何"某配置 +292%"的结论都可能只是抽中了右尾。
   => 所以本框架的任何指标都取【多种子分布】, 报中位数 / 最差 / 分位数,
      而不是单次值。

2) data/processed/ 下已有 90+ 个同窗口(2022-09~2026-07)的回测结果, 这段数据
   被用于选参数上百次。在同一段数据上反复挑最优, 挑出来的是噪声。
   => 所以配置必须在【两个独立窗口】上同时达标才采纳。

窗口定义与"无未来函数"的前提
────────────────────────────
    窗口A  2020-07-01 ~ 2022-08-31 (530 交易日)  从未用于任何调参
    窗口B  2022-09-01 ~ 2026-07-27 (943 交易日)  历史上被反复使用

    关键: 80 个入选特征必须【按窗口各自现场筛选】。现行线上锁定的特征集
    (wf_daily_REGRESS_CHK_*.json) 是用 2022-09 之前的数据筛的, 而这段数据
    正好覆盖窗口A的整个测试期 —— 直接拿来测窗口A就是未来函数, 结果会虚高。
    因此本框架对每个窗口各跑一次"筛特征"运行(只用该窗口起点之前的数据),
    再把结果锁给该窗口的所有种子。两个窗口因此用不同的 80 个特征, 这是正确的:
    我们检验的是"同一套流程按时间点重复应用", 不是"同一组特征".

三个阶段 (都可断点续跑, 已有产物自动跳过)
──────────────────────────────────────
    features : 每窗口 1 次, 现场筛特征并落盘, 供该窗口后续所有运行复用
    caches   : 每窗口 x 每种子 1 次模型运行, 落盘预测缓存 (贵, 一次性)
    eval     : 每配置 x 每窗口 x 每种子 1 次, --load-preds 跳过训练 (便宜)

因为 regime / min-pred / tranche-n / roll-rank / lot-flex / 成本 全是执行层参数,
换这些不需要重训 —— 建好缓存后, 每个新想法只需跑 eval 阶段。

用法
────
    python scripts/eval_grid.py features                # 先筛两个窗口的特征
    python scripts/eval_grid.py caches --seeds 20       # 建 2x20 预测缓存
    python scripts/eval_grid.py eval --configs base,n5  # 评估配置
    python scripts/eval_grid.py report --configs base,n5

注意: 本框架只读 data/processed 下的实验产物, 不碰线上任何文件。
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOGDIR = ROOT / "data" / "processed" / "eval_logs"
PY = ".venv/bin/python"

# 实验数据源: 2019 起的矩阵 + 修复了退市股缺口的 PIT 池
TRAIN_FILE = "training_data_pit_2019.parquet"
PIT_UNIVERSE = "universe_pit_2019.parquet"

# ── 两个独立窗口 ────────────────────────────────────────────
# feat_tag: 该窗口"筛特征"运行的 tag, 其产物 json 即该窗口的锁定特征来源
WINDOWS = {
    "A": {"test_start": "2020-07-01", "test_end": "2022-08-31",
          "feat_tag": "EVALFEAT_A", "desc": "2021抱团+2022熊市, 未用于调参"},
    "B": {"test_start": "2022-09-01", "test_end": "2026-07-27",
          "feat_tag": "EVALFEAT_B", "desc": "历史上被反复用于选参数"},
}

DEFAULT_SEEDS = [42, 7, 123, 2024, 31337, 1, 2, 3, 5, 11,
                 17, 23, 55, 77, 99, 202, 314, 512, 888, 1234]

# 所有运行共享的模型层参数。改这里等于换模型, 缓存必须重建。
MODEL_ARGS = ["--train-file", TRAIN_FILE, "--pit-universe", PIT_UNIVERSE,
              "--label", "5d", "--objective", "l2"]

# 所有运行共享的执行层默认值 (与线上 BASE_PARAMS 对齐)
EXEC_BASE = ["--hold-days", "5", "--portfolio-mode", "periodic",
             "--exec-mode", "t1close", "--slippage", "0.002"]

# ── 待评估的配置 ────────────────────────────────────────────
# 只允许放【执行层】参数, 否则 --load-preds 会因缓存不匹配而拒绝运行
# (这正是我们想要的保护: 模型层参数变了就必须重建缓存)。
CONFIGS = {
    "base3":  {"desc": "3只持仓, 不择时 (对应线上 aggr5w 的持仓数)",
               "args": ["--tranche-n", "3", "--initial-capital", "50000",
                        "--regime-filter", "off"]},
    "base5":  {"desc": "5只持仓, 不择时 (对应线上 steady5w)",
               "args": ["--tranche-n", "5", "--initial-capital", "50000",
                        "--regime-filter", "off"]},
}

# ── 验收门槛 ────────────────────────────────────────────────
# 必须【两个窗口同时】满足。中位数看"典型情况", 最差种子看"运气不好时"——
# 后者才是"不是偶发现象"的真正检验: 一个配置如果只有中位数好看但最差种子
# 亏 30%, 那你实盘抽到那个种子的概率并不低。
#
# 门槛怎么定出来的 (全部实测于窗口B, 943天, 5万本金):
#     配置              夏普中位  夏普最差  回撤中位  回撤最差  收益中位
#     3只 (20种子)        0.45    -0.06    -45.1%   -63.8%    +46.6%
#     5只 (5种子)         0.33     0.02    -49.7%   -53.8%    +22.2%
#     10只 (5种子)        0.16     0.13    -39.8%   -48.6%     +1.0%
#     20只/100万 (天花板)  0.61     0.54    -37.5%   -39.1%    +70.0%
#
# 注意最后一行: 即便完全分散 + 本金放大 20 倍, 夏普中位数也只有 0.61。
# 这就是这套 alpha 的真实水位, 不存在"夏普 1.4"那种东西 —— 历史上记下的
# 1.39 是单次跑抽中右尾的结果。因此门槛必须贴着 0.5 附近定, 定在 0.8/1.0
# 等于要求一个不存在的东西, 只会逼着自己去过拟合。
#
# 重点放在【最差种子】而非中位数: 中位数从 0.45 抬到 0.50 是锦上添花,
# 把最差种子从 -0.06/-63.8% 收到 0.15/-50% 才是"能不能睡着觉"的区别。
THRESHOLDS = {
    "sharpe_median": 0.50,   # 略高于现状 0.45, 低于天花板 0.61
    "sharpe_worst": 0.15,    # 稳健性核心: 20 个种子全部为正
    "maxdd_median": -42.0,   # 不得比这更深
    "maxdd_worst": -50.0,    # 稳健性核心: 从 -63.8% 收进来
    "max_loss_seeds_pct": 10.0,  # 亏损种子占比上限 (现状 4/20 = 20%)
}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def out_path(tag, win):
    w = WINDOWS[win]
    return PROC / f"wf_daily_{tag}_ts{w['test_start']}_te{w['test_end']}_cap50000.json"


def out_path_cap(tag, win, cap):
    w = WINDOWS[win]
    return PROC / f"wf_daily_{tag}_ts{w['test_start']}_te{w['test_end']}_cap{int(cap)}.json"


def cache_name(win, seed):
    return f"preds_eval_{win}_s{seed}.pkl"


def features_json(win):
    """该窗口锁定特征的来源文件名 (features 阶段的产物)"""
    return out_path(WINDOWS[win]["feat_tag"], win).name


def win_args(win):
    w = WINDOWS[win]
    return ["--test-start", w["test_start"], "--test-end", w["test_end"]]


def run_parallel(jobs, jobs_cap, phase):
    """jobs: [(name, cmd_list)] —— 并发上限内跑完, 返回失败列表

    每个进程都要把 800MB 矩阵读进来再建特征(约 4 分钟), 所以并发数受内存约束,
    不是 CPU。默认 12 是在 503GB 机器上实测安全的值。
    """
    LOGDIR.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(jobs), [], []
    total = len(pending)
    while pending or running:
        while pending and len(running) < jobs_cap:
            name, cmd = pending.pop(0)
            lf = open(LOGDIR / f"{phase}_{name}.log", "w")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
            running.append((name, p, lf))
            log(f"  启动 {name} ({total - len(pending) - len(running) + 1}/{total} 已排定)")
        time.sleep(5)
        for item in running[:]:
            name, p, lf = item
            if p.poll() is None:
                continue
            running.remove(item)
            lf.close()
            if p.returncode != 0:
                failed.append(name)
                log(f"  ✗ {name} rc={p.returncode} (见 {LOGDIR / f'{phase}_{name}.log'})")
            else:
                log(f"  ✓ {name}")
    return failed


# ── 阶段1: 每窗口现场筛特征 ─────────────────────────────────
def phase_features(args):
    jobs = []
    for win in args.windows:
        dst = out_path(WINDOWS[win]["feat_tag"], win)
        if dst.exists() and not args.force:
            log(f"窗口{win}: 特征已存在, 跳过 ({dst.name})")
            continue
        # 不传 --features-from = 现场筛选, 且脚本内部只用首个出信号日之前的
        # 数据做筛选(见 wf_v35 里 select_features(..., FIRST_PRED)), 无未来函数
        cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *MODEL_ARGS,
               *win_args(win), *EXEC_BASE,
               "--n-features", "80", "--tranche-n", "3",
               "--initial-capital", "50000", "--regime-filter", "off",
               "--lgb-seed", "42", "--tag", WINDOWS[win]["feat_tag"]]
        jobs.append((f"win{win}", cmd))
    if not jobs:
        return
    log(f"阶段 features: {len(jobs)} 个运行 (每窗口一次现场筛选)")
    failed = run_parallel(jobs, args.jobs, "features")
    if failed:
        raise SystemExit(f"features 阶段失败: {failed}")
    for win in args.windows:
        sel = json.loads(out_path(WINDOWS[win]["feat_tag"], win).read_text())
        log(f"窗口{win} 锁定 {len(sel['selected_features'])} 个特征 -> {features_json(win)}")


# ── 阶段2: 建预测缓存 (贵, 一次性) ──────────────────────────
def phase_caches(args):
    jobs = []
    for win in args.windows:
        fj = out_path(WINDOWS[win]["feat_tag"], win)
        if not fj.exists():
            raise SystemExit(f"窗口{win} 还没筛特征, 先跑: eval_grid.py features")
        for seed in args.seeds:
            if (PROC / cache_name(win, seed)).exists() and not args.force:
                continue
            cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *MODEL_ARGS,
                   *win_args(win), *EXEC_BASE,
                   "--features-from", fj.name,
                   "--tranche-n", "3", "--initial-capital", "50000",
                   "--regime-filter", "off", "--lgb-seed", str(seed),
                   "--save-preds", cache_name(win, seed),
                   "--tag", f"EVALCACHE_{win}_s{seed}"]
            jobs.append((f"{win}_s{seed}", cmd))
    if not jobs:
        log("阶段 caches: 全部已存在, 跳过")
        return
    log(f"阶段 caches: {len(jobs)} 个模型运行待跑 (这一步最耗时)")
    failed = run_parallel(jobs, args.jobs, "caches")
    if failed:
        raise SystemExit(f"caches 阶段失败: {failed}")


# ── 阶段3: 评估配置 (便宜, 复用缓存) ────────────────────────
def phase_eval(args):
    jobs = []
    for cname in args.configs:
        cfg = CONFIGS[cname]
        cap = _cap_of(cfg["args"])
        for win in args.windows:
            for seed in args.seeds:
                cache = PROC / cache_name(win, seed)
                if not cache.exists():
                    raise SystemExit(f"缺预测缓存 {cache.name}, 先跑 caches 阶段")
                tag = f"EV_{cname}_{win}_s{seed}"
                if out_path_cap(tag, win, cap).exists() and not args.force:
                    continue
                cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *MODEL_ARGS,
                       *win_args(win), *EXEC_BASE,
                       "--features-from", features_json(win),
                       "--load-preds", cache.name, *cfg["args"], "--tag", tag]
                jobs.append((f"{cname}_{win}_s{seed}", cmd))
    if not jobs:
        log("阶段 eval: 全部已存在, 跳过")
        return
    log(f"阶段 eval: {len(jobs)} 个执行层运行待跑")
    failed = run_parallel(jobs, args.jobs, "eval")
    if failed:
        raise SystemExit(f"eval 阶段失败: {failed}")


def _cap_of(arglist):
    if "--initial-capital" in arglist:
        return float(arglist[arglist.index("--initial-capital") + 1])
    return 100000.0


# ── 汇总 ────────────────────────────────────────────────────
def collect(cname, win, seeds):
    cap = _cap_of(CONFIGS[cname]["args"])
    rows = []
    for seed in seeds:
        p = out_path_cap(f"EV_{cname}_{win}_s{seed}", win, cap)
        if not p.exists():
            continue
        s = json.loads(p.read_text())["summary"]
        rows.append(s)
    return rows


def summarize(rows):
    if not rows:
        return None
    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)
    ret, shp, dd = arr("total_return_pct"), arr("sharpe"), arr("max_dd_pct")
    return {
        "n_seeds": len(rows),
        "ret_median": float(np.median(ret)), "ret_worst": float(ret.min()),
        "ret_best": float(ret.max()),
        "ret_p25": float(np.percentile(ret, 25)), "ret_p75": float(np.percentile(ret, 75)),
        "n_loss": int((ret < 0).sum()),
        "sharpe_median": float(np.median(shp)), "sharpe_worst": float(shp.min()),
        "maxdd_median": float(np.median(dd)), "maxdd_worst": float(dd.min()),
        "fee_median": float(np.median(arr("total_cost_pct"))),
        "ic_median": float(np.median(arr("ic_mean"))),
        "bench": float(np.median(arr("benchmark_total_pct"))),
        "n_below_bench": int((ret < np.median(arr("benchmark_total_pct"))).sum()),
    }


def verdict(sa, sb):
    """两窗口同时达标才算过"""
    if not sa or not sb:
        return "数据不全", []
    fails = []
    for win, s in (("A", sa), ("B", sb)):
        if s["sharpe_median"] < THRESHOLDS["sharpe_median"]:
            fails.append(f"{win}:夏普中位{s['sharpe_median']:.2f}<{THRESHOLDS['sharpe_median']}")
        if s["sharpe_worst"] < THRESHOLDS["sharpe_worst"]:
            fails.append(f"{win}:夏普最差{s['sharpe_worst']:.2f}<{THRESHOLDS['sharpe_worst']}")
        if s["maxdd_median"] < THRESHOLDS["maxdd_median"]:
            fails.append(f"{win}:回撤中位{s['maxdd_median']:.1f}%<{THRESHOLDS['maxdd_median']}%")
        if s["maxdd_worst"] < THRESHOLDS["maxdd_worst"]:
            fails.append(f"{win}:回撤最差{s['maxdd_worst']:.1f}%<{THRESHOLDS['maxdd_worst']}%")
        loss_pct = s["n_loss"] / s["n_seeds"] * 100
        if loss_pct > THRESHOLDS["max_loss_seeds_pct"]:
            fails.append(f"{win}:亏损种子{loss_pct:.0f}%>{THRESHOLDS['max_loss_seeds_pct']:.0f}%")
    return ("通过" if not fails else "不通过"), fails


def phase_report(args):
    print()
    print("=" * 108)
    print("多种子 x 双窗口评估报告")
    print(f"  窗口A {WINDOWS['A']['test_start']} ~ {WINDOWS['A']['test_end']}  ({WINDOWS['A']['desc']})")
    print(f"  窗口B {WINDOWS['B']['test_start']} ~ {WINDOWS['B']['test_end']}  ({WINDOWS['B']['desc']})")
    print(f"  门槛(两窗口同时): 夏普中位>={THRESHOLDS['sharpe_median']} 最差>={THRESHOLDS['sharpe_worst']}"
          f" | 回撤中位>={THRESHOLDS['maxdd_median']}% 最差>={THRESHOLDS['maxdd_worst']}%"
          f" | 亏损种子<={THRESHOLDS['max_loss_seeds_pct']:.0f}%")
    print("=" * 108)
    out = {}
    for cname in args.configs:
        sa = summarize(collect(cname, "A", args.seeds))
        sb = summarize(collect(cname, "B", args.seeds))
        v, fails = verdict(sa, sb)
        out[cname] = {"A": sa, "B": sb, "verdict": v, "fails": fails}
        print()
        print(f"### {cname} —— {CONFIGS[cname]['desc']}")
        print("%-4s %6s %9s %9s %9s %8s %8s %9s %9s %7s %7s" % (
            "窗口", "种子", "收益中位", "收益最差", "收益最好", "夏普中位",
            "夏普最差", "回撤中位", "回撤最差", "亏损数", "费用%"))
        for win, s in (("A", sa), ("B", sb)):
            if not s:
                print(f"{win:<4} (无数据)")
                continue
            print("%-4s %6d %9.1f %9.1f %9.1f %8.2f %8.2f %9.1f %9.1f %7s %7.1f" % (
                win, s["n_seeds"], s["ret_median"], s["ret_worst"], s["ret_best"],
                s["sharpe_median"], s["sharpe_worst"], s["maxdd_median"],
                s["maxdd_worst"], f"{s['n_loss']}/{s['n_seeds']}", s["fee_median"]))
        print(f"  结论: {v}" + (f"  |  未达标项: {'; '.join(fails)}" if fails else ""))
    dst = PROC / "eval_grid_report.json"
    dst.write_text(json.dumps({"thresholds": THRESHOLDS, "windows": WINDOWS,
                               "seeds": args.seeds, "results": out},
                              ensure_ascii=False, indent=2))
    print()
    print(f"报告已写入 {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["features", "caches", "eval", "report", "all"])
    ap.add_argument("--windows", default="A,B", help="逗号分隔, 默认 A,B")
    ap.add_argument("--seeds", default="20",
                    help="种子个数(取内置列表前 N 个), 或逗号分隔的具体种子")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help=f"逗号分隔, 可选: {','.join(CONFIGS)}")
    ap.add_argument("--jobs", type=int, default=12, help="并发进程上限(受内存约束)")
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    args = ap.parse_args()

    args.windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    for w in args.windows:
        if w not in WINDOWS:
            raise SystemExit(f"未知窗口 {w}, 可选 {list(WINDOWS)}")
    if args.seeds.isdigit():
        n = int(args.seeds)
        if n > len(DEFAULT_SEEDS):
            raise SystemExit(f"内置种子只有 {len(DEFAULT_SEEDS)} 个")
        args.seeds = DEFAULT_SEEDS[:n]
    else:
        args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in args.configs:
        if c not in CONFIGS:
            raise SystemExit(f"未知配置 {c}, 可选 {list(CONFIGS)}")

    log(f"窗口={args.windows} 种子={len(args.seeds)}个 配置={args.configs} 并发={args.jobs}")
    if args.phase in ("features", "all"):
        phase_features(args)
    if args.phase in ("caches", "all"):
        phase_caches(args)
    if args.phase in ("eval", "all"):
        phase_eval(args)
    if args.phase in ("report", "all"):
        phase_report(args)


if __name__ == "__main__":
    main()
