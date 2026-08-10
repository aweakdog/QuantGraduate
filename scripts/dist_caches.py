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
    "data/universe/universe_pit_2019.parquet",
    "data/raw/kline/",
]


def data_paths(variant):
    """变体用哪个训练矩阵就同步哪个 —— 写死会让 _rq 这类变体在工人机上
    因为文件不存在而直接起不来。"""
    tf = VARIANTS[variant].get("train_file", "training_data_pit_2019.parquet")
    return [f"data/processed/{tf}"] + DATA_PATHS


def log(m):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


def sh(host, cmd, timeout=1800):
    full = (["bash", "-lc", cmd] if host == SELF else
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             host, cmd])
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def alive_robust(host, tries=3):
    """存活探测必须对【瞬时 ssh 超时】容错。

    实测踩过: 三台机器都在满负荷跑(各 12 个 wf 进程, CPU 打满), 一次
    `ps` 探测超过 60 秒没返回, subprocess 抛 TimeoutExpired 直接把主控打死 ——
    而那 33 个缓存明明在正常构建。机器忙的时候探测本来就会慢, 不能把它当故障。

    所以: 重试几次, 全部超时才认定异常; 且"超时"倾向于判定【还在跑】,
    因为误判成"已退出"会触发失败检查, 把一个正常任务当成失败停掉。
    """
    for i in range(tries):
        try:
            rc, _ = sh(host,
                       'ps -eo args | grep -E "[p]ython3?.*wf_v35_breadth_alpha" '
                       '| grep -vq "ssh "', timeout=90)
            return rc == 0
        except subprocess.TimeoutExpired:
            if i == tries - 1:
                log(f"  !! {host} 探测连续 {tries} 次超时, 保守当作【还在跑】")
                return True
            time.sleep(20)
    return True


def launch_bg(host, cmd, logfile):
    """后台起长任务并立刻返回。三个坑见 expand_2015_overnight.launch_bg 的注释:
    ssh 会等 fd 关闭而挂死; cd 不能被 nohup 包装; 启动前要清空旧日志。"""
    esc = cmd.replace("'", "'\\''")
    return sh(host, f": > {logfile}; setsid nohup bash -c '{esc}' "
                    f"< /dev/null > {logfile} 2>&1 & echo launched", timeout=60)


def split_seeds(seeds, n):
    """余数摊给前几台, 保证最大最小片只差 1"""
    out = [[] for _ in range(n)]
    for i, s in enumerate(seeds):
        out[i % n].append(s)
    return out


def collect(host, variant):
    """把某台工人机的缓存 rsync 回主节点 (幂等, 可重复调用)"""
    if host == SELF:
        return
    rc, out = sh(SELF, f"rsync -a {host}:{REMOTE}/data/processed/"
                       f"{VARIANTS[variant]['cache']}_*.pkl {PROC}/", timeout=1800)
    log(f"  从 {host} 收回缓存: {'ok' if rc == 0 else out[-300:]}")


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
            log(f"  {h}: {'在跑' if alive_robust(h) else '空闲'}")
        return

    # ── 0. 前置检查: 工人机依赖与版本 ──
    # 实测踩过两次: (a) 工人机缺 scikit-learn(lightgbm.sklearn 依赖它), 全部
    # 任务开跑几分钟就失败; (b) pandas/numpy 版本不一致。后者更危险 —— 版本
    # 不同可能让同一种子产出不同预测, 那样跨机拼出来的 20 种子分布就是假的。
    # (已实测同版本下 s42 跨机逐位一致: 518 天排名零差异, 预测值最大差 0.0)
    probe = ('python3 -c "import lightgbm,sklearn,pandas,numpy,scipy,pyarrow;'
             'print(lightgbm.__version__,pandas.__version__,numpy.__version__)"')
    vers = {}
    for h in ALL_HOSTS:
        cmd = probe if h != SELF else probe.replace("python3", ".venv/bin/python")
        rc, out = sh(h, f"cd {REMOTE} && {cmd}", timeout=120)
        if rc != 0:
            raise SystemExit(f"{h} 依赖不全, 会在开跑几分钟后集体失败:\n{out[-600:]}\n"
                             f"  补装: ssh {h} 'python3 -m pip install --user "
                             f"lightgbm scikit-learn pandas numpy scipy pyarrow'")
        vers[h] = out.strip().splitlines()[-1]
    if len(set(vers.values())) > 1:
        raise SystemExit(
            "各机 lightgbm/pandas/numpy 版本不一致, 同一种子可能产出不同预测, "
            "跨机拼出来的分布就是假的:\n"
            + "\n".join(f"  {h}: {v}" for h, v in vers.items()))
    log(f"依赖与版本一致: {next(iter(vers.values()))}")

    # ── 1. 同步数据与代码 ──
    if not a.skip_sync:
        for h in WORKERS:
            log(f"同步数据到 {h} (约1.4G, 首次较慢)")
            paths = " ".join(data_paths(a.variant))
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
    # 重跑必须可续: 主控挂掉时工人还在正常跑(实测踩过 —— 一次 ps 探测超时就
    # 把主控打死, 而缓存在照常构建)。所以启动前先把【已空闲】机器的缓存收回来,
    # 再据此判断谁还有活 —— 否则会给一台已完工的机器重发任务, 它秒退, 紧接着被
    # "启动后立刻退出" 误判成故障。
    shards = split_seeds(seeds, len(ALL_HOSTS))
    for h in WORKERS:
        if not alive_robust(h):
            collect(h, a.variant)

    if not missing_caches(a.variant, wins, seeds):
        log(f"全部 {len(wins)*len(seeds)} 个缓存已就位, 无需构建")
        return

    launched = []
    for h, sd in zip(ALL_HOSTS, shards, strict=True):
        if not sd:
            continue
        if alive_robust(h):
            log(f"{h}: 已有 wf 进程在跑, 跳过启动")
            launched.append(h)
            continue
        todo = [(w, s) for w in wins for s in sd
                if not (PROC / cache_name(w, s, a.variant)).exists()]
        if not todo:
            log(f"{h}: 分片 {sd} 已全部完成, 跳过")
            continue
        pybin = ".venv/bin/python" if h == SELF else "python3"
        cmd = (f"cd {REMOTE} && {pybin} -u scripts/eval_grid.py caches "
               f"--variant {a.variant} --windows {','.join(wins)} "
               f"--seeds {','.join(map(str, sd))} --jobs {a.jobs}")
        rc, out = launch_bg(h, cmd, f"/tmp/dcache_{a.variant}.log")
        log(f"{h}: 种子 {sd} (缺 {len(todo)}) -> "
            f"{'已启动' if rc == 0 else '失败 ' + out[-200:]}")
        if rc != 0:
            raise SystemExit(f"{h} 启动失败")
        launched.append(h)

    time.sleep(20)
    for h in launched:
        if not alive_robust(h):
            _, tail = sh(h, f"tail -n 20 /tmp/dcache_{a.variant}.log")
            raise SystemExit(f"{h} 启动后立刻退出:\n{tail[-1200:]}")
        log(f"{h} 确认在跑")

    # ── 3. 等全部结束 ──
    # 关键: 不能只看"还有没有进程在跑"就一直等 —— 实测踩过, 两台工人机因为缺
    # scikit-learn 在开头几分钟就全部失败退出, 而主节点自己那片还在慢慢跑,
    # 于是等待循环看到"有机器在跑"就一直等下去, 白等了 57 分钟才发现。
    # 所以每轮都检查【已退出的机器是否真的完成了它那片】, 一发现失败立刻停,
    # 并把子进程日志带回来 —— 失败要立刻可见, 而不是等到最后收集时才暴露。
    t0 = time.time()
    assigned = {h: sd for h, sd in zip(ALL_HOSTS, shards, strict=True)
                if sd and h in launched}
    while True:
        run = [h for h in ALL_HOSTS if alive_robust(h)]
        for h, sd in assigned.items():
            if h in run:
                continue
            # 工人的缓存写在它自己的盘上, 判断"它那片是否完成"之前必须先收回来,
            # 否则一台先完工就会被误判成失败, 把还在跑的另一台一起停掉。
            collect(h, a.variant)
            left = [(w, s) for w in wins for s in sd
                    if not (PROC / cache_name(w, s, a.variant)).exists()]
            if left:
                _, tail = sh(h, f"tail -n 25 /tmp/dcache_{a.variant}.log")
                raise SystemExit(
                    f"{h} 已退出但它那片还缺 {len(left)} 个缓存: {left[:6]}\n"
                    f"  子进程日志:\n{tail[-1500:]}")
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
        collect(h, a.variant)
    miss = missing_caches(a.variant, wins, seeds)
    if miss:
        raise SystemExit(
            f"仍缺 {len(miss)} 个缓存: {miss[:8]}\n"
            "  不允许少几个种子就凑一个分布 —— 那会让最差种子/亏损比例失真。\n"
            "  重跑本脚本会自动只补缺的那些。")
    log(f"全部 {len(wins)*len(seeds)} 个缓存就位")


if __name__ == "__main__":
    main()
