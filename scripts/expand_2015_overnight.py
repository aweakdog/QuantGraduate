"""把历史扩到 2015 的无人值守编排 (在 eez041 上跑, 自动调度三机)

为什么要做这件事
────────────────
到 2026-08-04 为止, 所有结论都建立在 2019 起的数据上, 而那段时间里只有
2~3 次真正的 regime 切换 (信号 IC 由正转负再转正)。用约 11 个独立的 126 天
区间去验证一个择时规则, 统计效力太弱 —— 这是当前的**主要瓶颈**, 不是模型
容量不够。延到 2015 能多拿 2015 股灾、2016 熔断、2018 熊市三个独立 episode,
把 episode 数从 2~3 提到 5~6, 统计效力接近翻倍。

阶段 (每步写状态, 重跑自动跳过已完成的)
──────────────────────────────────────
  1. 前置检查: K线深度/幸存者偏差缺口/实盘定时任务是否在跑
  2. 建 2015 PIT 池 -> universe_pit_2015.parquet + watchlist_pit_2015.json
     参数与 2019 版【逐字一致】(--top-n 300 --freq semiannual --rank-by mcap),
     只改起止日期与输出名, 否则两版不可比
  3. 三机并行回灌资金流到 2015 (按 code 哈希分 3 片, 三个出口 IP 各拉一片)
  4. 收集分片 -> 合并进 fundflow_history.parquet (缺片拒绝合并)
  5. 重建特征矩阵 -> training_data_pit_2015.parquet
  6. 汇总报告 -> data/processed/expand_2015_report.json

全程不碰 v24 与线上任何文件: 输出都是 *_2015 命名, live_config 指向的
training_data_pit_v24.parquet / universe_pit.parquet 一个字节都不动。

用法
────
    nohup python scripts/expand_2015_overnight.py > /tmp/exp2015.log 2>&1 &
    # 看进度:
    python scripts/expand_2015_overnight.py --status
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PY = sys.executable
DATA = ROOT / "data"
KLINE = DATA / "raw" / "kline"
STATUS = DATA / "processed" / "expand_2015_status.json"
REPORT = DATA / "processed" / "expand_2015_report.json"
FF_DIR = DATA / "raw" / "fund_flow_full"
FF_CONS = FF_DIR / "fundflow_history.parquet"

UNI_OUT = DATA / "universe" / "universe_pit_2015.parquet"
WL_OUT = DATA / "universe" / "watchlist_pit_2015.json"
CODES_FILE = DATA / "processed" / "ff_codes_2015.txt"
TRAIN_OUT = "training_data_pit_2015.parquet"
FEATURES_DIR = "features_2015"
# 特征起算日: 比 PIT_START 早半年, 给滚动窗口(ma20/mom60等)留预热期,
# 否则 2015-07 那批样本的滚动特征全是 NaN
FEATURE_CUTOFF = "2015-01-01"

# 与 2019 版逐字一致的池子参数 —— 只改日期
PIT_START = "2015-07-01"
PIT_END = "2026-07-27"
TOP_N = "300"
FREQ = "semiannual"
RANK_BY = "mcap"
EXCLUDED_PREFIXES = ("200", "900")

HOSTS = ["eez040", "eez041", "eez042"]      # 分片 0/1/2, 顺序即片号
SELF = "eez041"
REMOTE_ROOT = "~/quant-strategy"

state = json.loads(STATUS.read_text()) if STATUS.exists() else {"stages": {}}


def log(msg):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def mark(stage, **kv):
    state["stages"][stage] = {"at": datetime.now().isoformat(timespec="seconds"), **kv}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))


def done(stage):
    return state["stages"].get(stage, {}).get("ok", False)


def run(cmd, name, timeout):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout,
                       capture_output=True, text=True)
    tail = ((r.stdout or "") + (r.stderr or ""))[-3000:]
    if r.returncode != 0:
        mark(name, ok=False, log=tail)
        raise RuntimeError(f"{name} rc={r.returncode}\n{tail}")
    return tail


def sh(host, cmd, timeout=600):
    """在指定机器上执行。本机直接跑, 避免依赖自连(短主机名不在 known_hosts)"""
    if host == SELF:
        full = ["bash", "-lc", cmd]
    else:
        full = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                host, cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def launch_bg(host, cmd, logfile):
    """在远端/本机后台起一个长任务, 并【立刻返回】

    坑 (2026-08-05 实测): 直接 `ssh host "nohup ... &"` 会挂死不返回。
    ssh 要等远端命令的 stdout/stderr 全部关闭才退出, 而后台进程 —— 以及它
    fork 出的 curl 子进程 —— 继承了 ssh 通道的 fd, 通道就一直不关。表现是
    编排脚本停在第一台的启动调用上, 后面两台永远起不来, 而第一台其实在正常跑,
    非常难看出来。

    所以: setsid 让它脱离会话, stdin 接 /dev/null, stdout/stderr 全部重定向
    到日志文件, 三个 fd 一个都不留给 ssh。

    坑二: `setsid nohup cd DIR && python ...` 跑不起来 —— cd 是 shell 内建,
    不能被 setsid/nohup 包装, nohup 直接报 "failed to run command 'cd'", 于是
    && 后面的 python 永远不执行; 而末尾的 `echo launched` 返回 0, 编排就误判
    成"已启动"。所以整条命令必须塞进 bash -c 里由子 shell 执行。

    坑三: 启动前先清空日志。否则上一次失败留下的旧日志会被诊断代码读到,
    给出"参数不认识"之类早已修好的误导信息(实测被它骗过一次)。
    """
    esc = cmd.replace("'", "'\\''")
    wrapped = (f": > {logfile}; setsid nohup bash -c '{esc}' "
               f"< /dev/null > {logfile} 2>&1 & echo launched")
    return sh(host, wrapped, timeout=60)


def pull_alive(host):
    """某台机器上资金流拉取是否真的在跑

    不能直接 grep 进程名: 挂死的 `ssh ... pull_fundflow_shard ...` 命令行里也含
    这个词, 会被当成"任务在跑"从而跳过启动 (实测踩过)。所以只认真正的
    python 进程。
    """
    rc, _ = sh(host, 'ps -eo args | grep -E "[p]ython3?.*pull_fundflow_shard" '
                     '| grep -vq "ssh "')
    return rc == 0


def wait_for_live_pipeline():
    """实盘定时任务 17:30/19:30/21:30 会写 kline, 与我们抢 CPU 也抢文件。
    它在跑就等 —— 我们这条链路不着急, 但绝不能影响实盘。"""
    waited = 0
    # grep 里用 [b] 是为了不匹配到 grep 自己那行 —— 否则永远认为流水线在跑
    probe = ["bash", "-lc", 'ps -eo args | grep -q "[b]in/python.*daily_rebuild"']
    while True:
        running = subprocess.run(probe, capture_output=True,
                                 text=True).returncode == 0
        if not running:
            if waited:
                log(f"实盘流水线已结束 (等了 {waited//60} 分钟), 继续")
            return
        if waited % 600 == 0:
            log("实盘定时流水线正在跑, 等它结束...")
        time.sleep(60)
        waited += 60


# ── 1. 前置检查 ───────────────────────────────────────────────────
def stage_precheck():
    if done("precheck"):
        return
    log("=== 1. 前置检查 ===")
    files = list(KLINE.glob("*.parquet"))
    n_2015 = 0
    for f in files[:400]:                      # 抽样, 全量读太慢
        try:
            d = pd.read_parquet(f, columns=["date"])
            if len(d) and pd.to_datetime(d["date"]).min().year <= 2015:
                n_2015 += 1
        except Exception:
            continue
    # 幸存者偏差缺口: 2015-07 后才退市却没有K线的, 剔B股
    meta = pd.read_parquet(DATA / "universe" / "pit_metadata.parquet")
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    for c in ("list_date", "delist_date"):
        meta[c] = pd.to_datetime(meta[c], errors="coerce")
    have = {f.stem[:6] for f in files}
    cand = meta[meta["list_date"] <= PIT_START]
    miss = cand[~cand["code"].isin(have)]
    gap = miss[(miss["delist_date"] >= PIT_START)
               & ~miss["code"].str.startswith(EXCLUDED_PREFIXES)]
    if len(gap):
        mark("precheck", ok=False, survivorship_gap=list(gap["code"]))
        raise RuntimeError(
            f"幸存者偏差缺口 {len(gap)} 只: {list(gap['code'])[:10]}\n"
            "  这些股票 2015-07 后才退市, 当时可投, 缺了会让回测虚高。\n"
            "  补: python scripts/update_kline_akshare.py --codes <清单> --start 20150101")
    log(f"  K线 {len(files)} 只 | 400 只抽样中 {n_2015} 只回到2015 | 幸存者偏差缺口 0")
    mark("precheck", ok=True, n_kline=len(files), sample_2015=n_2015)


# ── 2. PIT 池 ─────────────────────────────────────────────────────
def stage_pit():
    if done("pit"):
        return
    log("=== 2. 建 2015 PIT 池 ===")
    wait_for_live_pipeline()
    run([PY, "-u", "scripts/build_pit_universe.py",
         "--top-n", TOP_N, "--freq", FREQ, "--rank-by", RANK_BY,
         "--start", PIT_START, "--end", PIT_END,
         "--exclude-prefixes", ",".join(EXCLUDED_PREFIXES),
         "--out", str(UNI_OUT), "--jobs", "12"],
        "pit_build", timeout=7200)
    u = pd.read_parquet(UNI_OUT)
    u["code"] = u["code"].astype(str).str.zfill(6)
    codes = sorted(u["code"].unique())
    meta = pd.read_parquet(DATA / "universe" / "pit_metadata.parquet")
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    nm = dict(zip(meta["code"], meta["name"], strict=False))
    WL_OUT.write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"),
         "note": f"PIT {PIT_START}~{PIT_END} top{TOP_N} {FREQ} by {RANK_BY} 的并集",
         "watchlist": [{"code": c, "name": nm.get(c, "")} for c in codes]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    # 三机共用的代码清单: 必须同一份, 否则分片不一致会漏拉
    CODES_FILE.write_text("\n".join(codes) + "\n", encoding="utf-8")
    n_periods = u["effective_date"].nunique()
    log(f"  {len(codes)} 只并集 | {n_periods} 期 | 清单 -> {CODES_FILE}")
    mark("pit", ok=True, n_codes=len(codes), n_periods=int(n_periods))


# ── 3. 三机并行拉资金流 ────────────────────────────────────────────
def stage_fundflow_pull():
    if done("ff_pull"):
        return
    log("=== 3. 三机并行回灌资金流到 2015 ===")
    # 分发代码清单与脚本 (免密已配, eez041 可直推另两台)
    for h in HOSTS:
        if h == SELF:
            continue
        rc, out = sh(SELF, f"rsync -a {CODES_FILE} {h}:/tmp/ff_codes_2015.txt "
                           f"&& rsync -a {ROOT}/scripts/pull_fundflow_shard.py "
                           f"{h}:{REMOTE_ROOT}/scripts/ "
                           f"&& rsync -a {ROOT}/pipeline/pull_fundflow_sina.py "
                           f"{h}:{REMOTE_ROOT}/pipeline/")
        if rc != 0:
            mark("ff_pull", ok=False, error=f"分发到 {h} 失败: {out[-500:]}")
            raise RuntimeError(f"分发到 {h} 失败: {out[-500:]}")
        log(f"  已分发到 {h}")

    for i, h in enumerate(HOSTS):
        # 已经在跑就别重复起 (续跑场景: 上次编排挂了但某台的拉取还活着)
        if pull_alive(h):
            log(f"  {h} 第{i}片: 已在跑, 跳过启动")
            continue
        pybin = ".venv/bin/python" if h == SELF else "python3"
        cf = str(CODES_FILE) if h == SELF else "/tmp/ff_codes_2015.txt"
        cmd = (f"cd {REMOTE_ROOT} && {pybin} -u scripts/pull_fundflow_shard.py "
               f"--shard {i} --of 3 --since 2015-01-01 --codes-file {cf}")
        rc, out = launch_bg(h, cmd, f"/tmp/ff_shard{i}.log")
        log(f"  {h} 第{i}片: {'已启动' if rc == 0 else '启动失败 ' + out[-200:]}")
        if rc != 0:
            mark("ff_pull", ok=False, error=f"{h} 启动失败")
            raise RuntimeError(f"{h} 启动失败: {out[-300:]}")

    # 启动确认: nohup 起进程后立刻返回 0, 所以 rc==0 只代表"命令发出去了",
    # 不代表进程活着。参数不对/依赖缺失时子进程会秒退, 而下面的"等待结束"
    # 循环会立刻认为三片都跑完了 —— 实测踩过: 三台的脚本是旧版没有
    # --codes-file, 秒退后编排却报告"三片全部结束, 耗时 0 分钟"。
    # 所以必须停一下再确认进程真的在跑, 并把子进程日志带回来。
    time.sleep(15)
    for i, h in enumerate(HOSTS):
        if not pull_alive(h):
            _, tail = sh(h, f"tail -n 15 /tmp/ff_shard{i}.log 2>/dev/null")
            mark("ff_pull", ok=False, error=f"{h} 第{i}片启动后立刻退出",
                 child_log=tail[-1500:])
            raise RuntimeError(
                f"{h} 第{i}片启动后立刻退出, 子进程日志:\n{tail[-1500:]}")
        log(f"  {h} 第{i}片确认在跑")

    # 等三片全部结束 (最长 8 小时)
    t0 = time.time()
    while True:
        alive = [h for h in HOSTS if pull_alive(h)]
        if not alive:
            break
        if time.time() - t0 > 8 * 3600:
            mark("ff_pull", ok=False, error=f"超过 8 小时仍在跑: {alive}")
            raise RuntimeError(f"资金流拉取超时, 仍在跑: {alive}")
        if int(time.time() - t0) % 900 < 65:
            prog = []
            for i, h in enumerate(HOSTS):
                _, o = sh(h, f"tail -n 2 /tmp/ff_shard{i}.log 2>/dev/null | tr '\\n' ' '")
                prog.append(f"{h}:{o.strip()[-90:]}")
            log(f"  [{(time.time()-t0)/60:.0f}min] " + " | ".join(prog))
        time.sleep(60)
    log(f"  三片全部结束, 耗时 {(time.time()-t0)/60:.0f} 分钟")
    mark("ff_pull", ok=True, minutes=round((time.time() - t0) / 60, 1))


# ── 4. 收集 + 合并 ────────────────────────────────────────────────
def stage_fundflow_merge():
    if done("ff_merge"):
        return
    log("=== 4. 收集分片并合并 ===")
    for i, h in enumerate(HOSTS):
        if h == SELF:
            continue
        rc, out = sh(SELF, f"rsync -a {h}:{REMOTE_ROOT}/data/raw/fund_flow_full/"
                           f"shard_{i}of3.parquet {FF_DIR}/")
        if rc != 0:
            mark("ff_merge", ok=False, error=f"从 {h} 取分片失败: {out[-400:]}")
            raise RuntimeError(f"从 {h} 取分片失败: {out[-400:]}")
        log(f"  已取回 {h} 的第{i}片")
    out = run([PY, "-u", "scripts/pull_fundflow_shard.py", "--merge", "--of", "3"],
              "ff_merge_run", timeout=1800)
    ff = pd.read_parquet(FF_CONS, columns=["date", "code"])
    ff["date"] = pd.to_datetime(ff["date"])
    log(f"  合并后 {len(ff):,} 行 / {ff['code'].nunique()} 只 / "
        f"{ff['date'].min():%F} ~ {ff['date'].max():%F}")
    mark("ff_merge", ok=True, rows=len(ff), codes=int(ff["code"].nunique()),
         span=f"{ff['date'].min():%F} ~ {ff['date'].max():%F}", log=out[-800:])


# ── 5. 特征矩阵 ───────────────────────────────────────────────────
def stage_features():
    if done("features"):
        return
    log("=== 5. 重建特征矩阵 (2015 起) ===")
    wait_for_live_pipeline()
    # 调用方式与 2019 版逐字一致 (pipeline.feature_engine 作为模块跑, 不是
    # scripts/ 下的脚本)。--no-incremental 是必须的: 增量模式会复用旧的
    # 2019 起的特征文件, 2015-2018 那段就永远补不上。
    run([PY, "-u", "-m", "pipeline.feature_engine", "--no-incremental",
         "--procs", "8", "--watchlist", WL_OUT.name, "--out", TRAIN_OUT,
         "--cutoff", FEATURE_CUTOFF, "--features-dir", FEATURES_DIR],
        "features_build", timeout=8 * 3600)
    tp = DATA / "processed" / TRAIN_OUT
    df = pd.read_parquet(tp, columns=["date", "code"])
    df["date"] = pd.to_datetime(df["date"])
    first = df["date"].min()
    # 结果校验: 没拿到 2015 历史就当失败, 而不是静默通过 —— 整条链路的目的
    # 就是这段历史, 少了它后面所有窗口都白跑 (照抄 2019 版的这道检查)
    if first >= pd.Timestamp("2017-01-01"):
        mark("features", ok=False, rows=len(df),
             span=f"{first:%F} ~ {df['date'].max():%F}",
             error=f"矩阵最早只到 {first:%F}, 没有 2015-2016 训练历史")
        raise RuntimeError(f"特征矩阵没有 2015 历史 (最早 {first:%F})")
    log(f"  {len(df):,} 行 / {df['code'].nunique()} 只 / "
        f"{first:%F} ~ {df['date'].max():%F}")
    mark("features", ok=True, rows=len(df), codes=int(df["code"].nunique()),
         span=f"{first:%F} ~ {df['date'].max():%F}")


# ── 6. 报告 ───────────────────────────────────────────────────────
def stage_report():
    log("=== 6. 汇总报告 ===")
    rep = {"at": datetime.now().isoformat(timespec="seconds"),
           "stages": state["stages"]}
    try:
        ff = pd.read_parquet(FF_CONS, columns=["date", "code"])
        ff["date"] = pd.to_datetime(ff["date"])
        rep["fundflow_by_year"] = {
            str(y): int(g["code"].nunique())
            for y, g in ff.groupby(ff["date"].dt.year)}
    except Exception as e:
        rep["fundflow_by_year"] = f"读失败 {type(e).__name__}"
    try:
        mani = json.loads((DATA / "raw" / "kline_source_manifest.json")
                          .read_text(encoding="utf-8"))
        from collections import Counter
        rep["kline_sources"] = dict(Counter(mani["source_of"].values()))
    except Exception:
        pass
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    log(f"  报告 -> {REPORT}")
    log("下一步(人工): 用 2015 矩阵给 eval_grid 加窗口 C/D, 重验 breadth 择时")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="只看进度")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return
    t0 = time.time()
    try:
        stage_precheck()
        stage_pit()
        stage_fundflow_pull()
        stage_fundflow_merge()
        stage_features()
        stage_report()
        log(f"全部完成, 共 {(time.time()-t0)/3600:.1f} 小时")
    except Exception as e:
        log(f"!! 中断: {type(e).__name__}: {e}")
        log(f"   已完成的阶段不会重跑, 修好后直接再执行一次本脚本即可续跑。状态: {STATUS}")
        raise


if __name__ == "__main__":
    main()
