"""把预测缓存构建分摊到三台机器 (在 eez041 上跑)

为什么值得分布式
────────────────
建缓存是整条评估链路里最贵的一步: 每个 (窗口, 种子) 要跑一遍完整的走进式
训练。80 个缓存单机约 9 小时。

单机加并发没用: LOCKED_PARAMS 里 n_jobs=10, 12 个进程就是 120 线程,
128 核已经饱和。所以唯一的杠杆是横向加机器 —— 3 x 12 = 36 并发, 约 3 小时。

做法
────
按【种子】分片(不是按窗口), 因为不同窗口耗时差一倍(A 530 天 / B 943 天),
按窗口分会让一台闲着。种子数除以机器数, 余数摊给前几台。

每台跑完把 .pkl 缓存 rsync 回主节点。缺任何一个缓存都会在汇总时报错 ——
不允许"少几个种子也凑合出个分布"。

用法
────
    python scripts/dist_caches.py --variant mb_dmw --seeds 20
    python scripts/dist_caches.py --variant mb_dmw --seeds 20 --status
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_grid import (  # noqa: E402
    DEFAULT_SEEDS, VARIANTS, WINDOWS, cache_name,
)

SELF = "eez041"
WORKERS = ["eez040", "eez042"]
ALL_HOSTS = [SELF] + WORKERS
REMOTE = "~/quant-strategy"
PROC = ROOT / "data" / "processed"

# 工人跑 wf_v35 需要的数据 (合计约 1.4G)。kline 最大但必须有 —— 走进式回测
# 每天都要取行情。
DATA_PATHS = [
    "data/processed/training_data_pit_2019.parquet",
    "data/universe/universe_pit_2019.parquet",
    "data/raw/kline/",
]


def log(m):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


def sh(host, cmd, timeout=1800):
    full = (["bash", "-lc", cmd] if host == SELF else
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             host, cmd])
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def launch_bg(host, cmd, logfile):
    """后台起长任务并立刻返回。三个坑见 expand_2015_overnight.launch_bg 的注释:
    ssh 会等 fd 关闭而挂死; cd 不能被 nohup 包装; 启动前要清空旧日志。"""
    esc = cmd.replace("'", "'\\''")
    return sh(host, f": > {logfile}; setsid nohup bash -c '{esc}' "
                    f"< /dev/null > {logfile} 2>&1 & echo launched", timeout=60)


def alive(host):
    """只认真正的 python 进程 —— 挂死的 ssh 命令行里也含关键词, 会误判在跑"""
    rc, _ = sh(host, 'ps -eo args | grep -E "[p]ython3?.*wf_v35_breadth_alpha" '
                     '| grep -vq "ssh "', timeout=60)
    return rc == 0


def split_seeds(seeds, n):
    """余数摊给前几台, 保证最大最小片只差 1"""
    out = [[] for _ in range(n)]
    for i, s in enumerate(seeds):
        out[i % n].append(s)
    return out


def missing_caches(variant, windows, seeds):
    v = VARIANTS[variant]
    miss = []
    for w in windows:
        for s in seeds:
            p = PROC / cache_name(w, s, variant)
            if not p.exists():
                miss.append((w, s))
    _ = v
    return miss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--windows", default="A,B")
    ap.add_argument("--jobs", type=int, default=12,
                    help="每台的并发上限。128核/n_jobs=10 -> 12 已饱和, 别调高")
    ap.add_argument("--skip-sync", action="store_true", help="跳过数据同步")
    ap.add_argument("--status", action="store_true", help="只看进度")
    a = ap.parse_args()

    seeds = DEFAULT_SEEDS[:a.seeds]
    wins = [w.strip() for w in a.windows.split(",") if w.strip()]
    for w in wins:
        if w not in WINDOWS:
            raise SystemExit(f"未知窗口 {w}")

    if a.status:
        miss = missing_caches(a.variant, wins, seeds)
        log(f"{a.variant}: 需要 {len(wins)*len(seeds)} 个缓存, 还缺 {len(miss)}")
        for h in ALL_HOSTS:
            log(f"  {h}: {'在跑' if alive(h) else '空闲'}")
        return

    # ── 1. 同步数据与代码 ──
    if not a.skip_sync:
        for h in WORKERS:
            log(f"同步数据到 {h} (约1.4G, 首次较慢)")
            paths = " ".join(DATA_PATHS)
            rc, out = sh(SELF, f"cd {ROOT} && rsync -a --relative {paths} {h}:{REMOTE}/",
                         timeout=3600)
            if rc != 0:
                raise SystemExit(f"同步数据到 {h} 失败: {out[-500:]}")
            # 特征 json 与脚本
            rc, out = sh(SELF,
                         f"cd {ROOT} && rsync -a scripts/ {h}:{REMOTE}/scripts/ "
                         f"&& rsync -a pipeline/ {h}:{REMOTE}/pipeline/ "
                         f"&& rsync -a data/processed/wf_daily_EVALFEAT_*.json "
                         f"{h}:{REMOTE}/data/processed/", timeout=600)
            if rc != 0:
                raise SystemExit(f"同步代码到 {h} 失败: {out[-500:]}")
            log(f"  {h} 就绪")

    # ── 2. 按种子分片启动 ──
    shards = split_seeds(seeds, len(ALL_HOSTS))
    for h, sd in zip(ALL_HOSTS, shards, strict=True):
        if not sd:
            continue
        if alive(h):
            log(f"{h}: 已有 wf 进程在跑, 跳过启动")
            continue
        pybin = ".venv/bin/python" if h == SELF else "python3"
        cmd = (f"cd {REMOTE} && {pybin} -u scripts/eval_grid.py caches "
               f"--variant {a.variant} --windows {','.join(wins)} "
               f"--seeds {','.join(map(str, sd))} --jobs {a.jobs}")
        rc, out = launch_bg(h, cmd, f"/tmp/dcache_{a.variant}.log")
        log(f"{h}: 种子 {sd} -> {'已启动' if rc == 0 else '失败 ' + out[-200:]}")
        if rc != 0:
            raise SystemExit(f"{h} 启动失败")

    time.sleep(20)
    for h in ALL_HOSTS:
        if not alive(h):
            _, tail = sh(h, f"tail -n 20 /tmp/dcache_{a.variant}.log")
            raise SystemExit(f"{h} 启动后立刻退出:\n{tail[-1200:]}")
        log(f"{h} 确认在跑")

    # ── 3. 等全部结束 ──
    t0 = time.time()
    while True:
        run = [h for h in ALL_HOSTS if alive(h)]
        if not run:
            break
        if time.time() - t0 > 10 * 3600:
            raise SystemExit(f"超过 10 小时仍在跑: {run}")
        if int(time.time() - t0) % 900 < 65:
            done = len(wins) * len(seeds) - len(missing_caches(a.variant, wins, seeds))
            log(f"  [{(time.time()-t0)/60:.0f}min] 已完成 {done}/{len(wins)*len(seeds)} "
                f"| 在跑 {run}")
        time.sleep(60)
    log(f"三台全部结束, 耗时 {(time.time()-t0)/60:.0f} 分钟")

    # ── 4. 收集缓存 ──
    for h in WORKERS:
        rc, out = sh(SELF, f"rsync -a {h}:{REMOTE}/data/processed/"
                           f"{VARIANTS[a.variant]['cache']}_*.pkl {PROC}/",
                     timeout=1800)
        log(f"  从 {h} 收回缓存: {'ok' if rc == 0 else out[-300:]}")
    miss = missing_caches(a.variant, wins, seeds)
    if miss:
        raise SystemExit(
            f"仍缺 {len(miss)} 个缓存: {miss[:8]}\n"
            "  不允许少几个种子就凑一个分布 —— 那会让最差种子/亏损比例失真。\n"
            "  重跑本脚本会自动只补缺的那些。")
    log(f"全部 {len(wins)*len(seeds)} 个缓存就位")


if __name__ == "__main__":
    main()
