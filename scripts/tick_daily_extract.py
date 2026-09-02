# -*- coding: utf-8 -*-
"""逐笔特征日更链 (跑在 eez040, 逐笔仓库所在机): 同步 -> 挂链 -> 抽取 -> 推送

为什么存在
──────────
V24PUT 的逐笔 alpha 随 staleness 减半 (lag1 IR 0.52 -> lag5 0.19, 见
docs/findings_2026-08-16 §7), 所以"每天把 d-1 的逐笔变成特征并送到线上机"
是生产依赖, 不是加分项。链条:

    123云盘(供应商, T+1早晨) --sync_tick_123--> ~/tickdata123/<YYYYMM>/<d>.7z
      --symlink--> 百度树 <TICK_DIR>/<年>/<YYYYMM>/<d>.7z   (抽取器只认这个布局)
      --tick_micro_features--> data/processed/tick_micro/<d>.parquet  (幂等, 已有即跳)
      --t1a_order_features--> data/processed/t1a_daily/<d>.parquet    (同上)
      --rsync--> eez041:~/quant-strategy/data/processed/{tick_micro,t1a_daily}/
                 (17:30 重建前就位)

两个抽取器共一个质检闸与一份 7z: T1A(订单结构)与 tk_*(OFI/TED/委托族)读同一
包、同一天, 分两条链就要么重复解压、要么漏质检 —— 7/9~7/21 静默坏包那次栽的
就是质检不到位, 不该再给自己开第二个没闸的口子。

设计取舍
────────
* 自己先调一遍 sync(幂等+文件锁), 不依赖 cron 时序凑巧 —— 16:40 那班是
  17:30 重建前的最后补漏机会, 必须现场拉一次而不是吃 12:17 的剩饭。
* 抽取器 run_day 对已有输出直接 skip, 所以这里只算出"有包没特征"的日期区间
  丢给它, 不需要自己做增量记账。
* 单日抽取约 4 分钟(630 只), NW 限 4 —— 共享机器, 平时只有 1 天缺口用不上并发,
  补历史时也不许吃满。
* rsync 失败要响(退出码非0), 但抽取成功本身有价值(下一班会重推), 所以推送
  失败不回滚不重试, 只留非零退出码给 cron 邮件/日志。

用法
────
    python3 scripts/tick_daily_extract.py            # 全链 (本脚本只用标准库)
    python3 scripts/tick_daily_extract.py --no-sync  # 跳过云盘同步
cron (eez040):
    40 7,12,16 * * * cd ~/quant-strategy && /usr/bin/python3 scripts/tick_daily_extract.py >> ~/logs/tickfeat_$(date +\%Y\%m).log 2>&1
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICK = Path(os.environ.get(
    "TICK_DIR", "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
TREE123 = Path(os.environ.get("TICK123_DIR", os.path.expanduser("~/tickdata123")))
OUT = ROOT / "data/processed/tick_micro"
OUT_T1A = ROOT / "data/processed/t1a_daily"
REMOTE = os.environ.get(
    "TICK_PUSH_TO", "eez041.ece.ust.hk:~/quant-strategy/data/processed/tick_micro/")
REMOTE_T1A = os.environ.get(
    "T1A_PUSH_TO", "eez041.ece.ust.hk:~/quant-strategy/data/processed/t1a_daily/")
LOCK_FILE = "/tmp/tick_daily_extract.lock"
# 两个虚拟环境, 不能混: sync 要 123 云盘 SDK (只装在 venv123),
# 抽取器要 py7zr/pandas/pyarrow (装在仓库 .venv)。本脚本只用标准库, 谁跑都行。
PY_SYNC = os.path.expanduser("~/venv123/bin/python")
PY_EXTRACT = str(ROOT / ".venv/bin/python")


def log(*a):
    print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def run(cmd, name, timeout, env=None):
    log(f"== {name} ==", " ".join(map(str, cmd)))
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(list(map(str, cmd)), cwd=str(ROOT), timeout=timeout, env=e)
    if r.returncode != 0:
        raise RuntimeError(f"{name} 退出码 {r.returncode}")


def link_123_into_tree():
    """把 123 树 (<YYYYMM>/<d>.7z) 缺的文件符号链接进百度树 (<年>/<YYYYMM>/)。

    抽取器 resolve() 只认百度树两种布局; 123 日更文件另起一树是为了和百度全量
    历史隔离。symlink 是零拷贝的粘合层, 也保留了"这个日期来自哪个渠道"的痕迹
    (ls -la 一眼可见)。
    """
    made = []
    for f in sorted(TREE123.glob("2*/*.7z")):
        mon = f.parent.name                       # YYYYMM
        dst_dir = TICK / mon[:4] / mon
        dst = dst_dir / f.name
        if dst.exists() or dst.is_symlink():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(f)
        made.append(f.name)
    if made:
        log(f"挂链 {len(made)} 个: {', '.join(made[-5:])}")
    return made


def missing_days(out_dir):
    """有 .7z 没 parquet 的日期。只看 2022-09 之后 —— 训练矩阵起点之前的不欠账。"""
    have = {p.stem for p in out_dir.glob("*.parquet")}
    days = sorted(p.stem for p in TICK.glob("*/*/*.7z")
                  if p.stem >= "20220901" and p.stem not in have)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true", help="跳过 123 云盘同步")
    ap.add_argument("--no-push", action="store_true", help="跳过 rsync 推送")
    a = ap.parse_args()

    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("已有一个在跑, 退出")
        return 0

    # 1. 云盘同步 (失败不挡抽取: 本地已有的包照样能出特征)
    if not a.no_sync:
        try:
            run([PY_SYNC, "scripts/sync_tick_123.py"], "sync123", timeout=3600)
        except Exception as e:
            log(f"WARN 同步失败(继续, 用已有的包): {e}")

    # 2. 挂链 + 3. 逐日质检 -> 抽取
    # 质检前置: 123 渠道出过"静默缺早盘委托"(7/9~7/21, 体积小 6~18%,
    # 无任何报错)。坏包抽出来的 ord_*/cxl_* 全是错的, 进了矩阵就是毒特征,
    # 所以宁可跳过抽取让矩阵端 --require-fresh 兜 staleness, 也不产毒。
    # 坏包不落 parquet, 下一班会重试(供应商重传后需手工删本地 7z 再 sync)。
    link_123_into_tree()
    need_tm = set(missing_days(OUT))
    need_t1a = set(missing_days(OUT_T1A))
    days = sorted(need_tm | need_t1a)
    if not days:
        log("无待抽取日期")
    for d in days:
        rc = subprocess.run(
            [PY_EXTRACT, "scripts/tick_qc_early.py", "--day", d],
            cwd=str(ROOT), timeout=1800).returncode
        if rc == 3:
            log(f"⚠⚠ {d} 质检判坏(缺早盘委托), 跳过抽取 —— 去催供应商重传!")
            continue
        if rc not in (0, 3):
            log(f"⚠ {d} 质检异常(rc={rc}), 跳过抽取")
            continue
        if d in need_tm:
            run([PY_EXTRACT, "scripts/tick_micro_features.py", d, d],
                f"extract_{d}", timeout=7200, env={"NW": "1", "CHUNK": "50"})
        # T1A 失败不能拖死 tk_* —— 后者是全线在用的硬依赖, T1A 目前只是
        # "列建而不用"(分点分配尚未切线)。真缺值的后果由 041 侧
        # build_t1_augmented 的末日覆盖闸接着管, 那里才知道谁在用这些列。
        if d in need_t1a:
            try:
                run([PY_EXTRACT, "scripts/t1a_order_features.py", d, d],
                    f"t1a_{d}", timeout=3600, env={"NW": "1", "CHUNK": "50"})
            except Exception as e:
                log(f"⚠ {d} T1A 抽取失败(不中断): {e}")

    # 4. 推送 (只推 parquet; --ignore-existing 不覆盖 —— 特征文件写出后不可变,
    #    真要返工用 --force 场景手工删除远端再推)
    if not a.no_push:
        run(["rsync", "-a", "--ignore-existing", "--include=*.parquet",
             "--exclude=*", f"{OUT}/", REMOTE], "push", timeout=1800)
        run(["rsync", "-a", "--ignore-existing", "--include=*.parquet",
             "--exclude=*", f"{OUT_T1A}/", REMOTE_T1A], "push_t1a", timeout=1800)

    for tag, d in (("tk_*", OUT), ("T1A", OUT_T1A)):
        ps = list(d.glob("*.parquet"))
        log(f"完成: {tag} 本地 {len(ps)} 天, 最新 "
            f"{max((p.stem for p in ps), default='-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
