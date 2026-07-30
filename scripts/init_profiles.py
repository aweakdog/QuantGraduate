"""建立(或重置)实盘并行线的状态文件

用途
────
把 live_config.PROFILES 里定义的每条线各建一份 state_<id>.json, 现金 = 该线本金、
持仓为空。之后 daily_rebuild.py 每天会给每条线各出一份计划。

为什么要单独一个脚本: live_signal.py 的 --init 会清空持仓记录, 参数又多且必须与
指纹严格一致, 手敲极易出错。这里从 live_config 取参数, 保证不会写出和定时任务
不一致的状态。

安全设计
────────
  * 默认不覆盖已存在的状态 —— 重置会丢失持仓和历史, 必须显式 --force
  * --force 时先把旧状态备份到 data/live/archive/, 不直接删
  * 默认 --dry-run 关闭, 但会先打印将要执行的动作

用法
────
    python scripts/init_profiles.py                 # 建立所有还不存在的线
    python scripts/init_profiles.py --profile aggr2w
    python scripts/init_profiles.py --force         # 全部归零重建(会备份旧的)
    python scripts/init_profiles.py --list          # 只看当前状态
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_config import PROFILES, init_args, state_file  # noqa: E402

PY = sys.executable
LIVE = ROOT / "data" / "live"
ARCHIVE = LIVE / "archive"


def current(pid):
    """读该线现状, 不存在返回 None"""
    p = LIVE / state_file(pid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_broken": str(e)}


def show():
    print(f"{'profile':10s} {'名称':<10s} {'本金':>8s} {'只':>3s} {'状态':<10s} {'持仓':>4s} {'现金':>10s}")
    for pid, p in PROFILES.items():
        st = current(pid)
        if st is None:
            print(f"{pid:10s} {p['name']:<10s} {p['capital']:>8,.0f} {p['tranche-n']:>3d} "
                  f"{'未建立':<10s} {'-':>4s} {'-':>10s}")
            continue
        if st.get("_broken"):
            print(f"{pid:10s} {p['name']:<10s} {'':>8s} {'':>3s} 损坏: {st['_broken'][:40]}")
            continue
        n = len(st.get("lots") or [])
        print(f"{pid:10s} {p['name']:<10s} {p['capital']:>8,.0f} {p['tranche-n']:>3d} "
              f"{'已建立':<10s} {n:>4d} {st.get('cash', 0):>10,.0f}")


def archive(pid):
    """备份旧状态和该线的历史计划, 便于出问题时回溯"""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = LIVE / state_file(pid)
    if src.exists():
        dst = ARCHIVE / f"{src.stem}_{ts}.json"
        shutil.copy2(src, dst)
        print(f"    旧状态已备份 -> archive/{dst.name}")
    moved = 0
    for pp in LIVE.glob(f"plan_{pid}_*.json"):
        shutil.move(str(pp), str(ARCHIVE / f"{pp.stem}_{ts}.json"))
        moved += 1
    if moved:
        print(f"    旧计划 {moved} 份已移入 archive/")


def init_one(pid, force):
    st = current(pid)
    if st is not None and not force:
        n = len(st.get("lots") or [])
        print(f"[{pid}] 已存在 (持仓 {n} 只, 现金 {st.get('cash', 0):,.0f}) — 跳过。"
              f"要归零请加 --force")
        return None
    print(f"[{pid}] {PROFILES[pid]['name']}: 本金 {PROFILES[pid]['capital']:,.0f} / "
          f"{PROFILES[pid]['tranche-n']} 只")
    if st is not None:
        archive(pid)
    # 必须跑完整流程而不能加 --status —— --status 是个只读快通道, 会在
    # save_state 之前就 sys.exit(0), 根本不会把新状态落盘。
    # 跑完整流程的好处是初始化完就直接得到第一份建仓计划。
    cmd = [PY, "-u", str(ROOT / "scripts" / "live_signal.py")] + init_args(pid)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    if r.returncode != 0:
        print(f"    失败:\n{(r.stderr or r.stdout)[-1500:]}")
        return False
    for line in r.stdout.splitlines():
        if "计划已保存" in line or "买入" in line and "合计" in line:
            print(f"    {line.strip()}")
    print("    已建立")
    return True


def main():
    ap = argparse.ArgumentParser(description="建立/重置实盘并行线")
    ap.add_argument("--profile", help="只处理这一条线")
    ap.add_argument("--force", action="store_true",
                    help="已存在也重建 (会清空持仓, 旧状态先备份到 archive/)")
    ap.add_argument("--list", action="store_true", help="只打印当前状态")
    a = ap.parse_args()

    if a.list:
        show()
        return 0

    pids = [a.profile] if a.profile else list(PROFILES)
    for pid in pids:
        if pid not in PROFILES:
            print(f"未知 profile: {pid} (可选: {', '.join(PROFILES)})")
            return 2

    if a.force:
        print("!! --force: 下列线的持仓与历史将被清空 (旧文件备份到 data/live/archive/)")
        print("   " + ", ".join(pids) + "\n")

    fails = []
    for pid in pids:
        if init_one(pid, a.force) is False:
            fails.append(pid)
    print()
    show()
    if fails:
        print(f"\n失败: {', '.join(fails)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
