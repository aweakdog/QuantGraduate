#!/usr/bin/env python3
"""把基准线对齐到它所参照的真实条线。

为什么不直接 init_profiles.py --profile base5w_steady
──────────────────────────────────────────────
那样基准线会从"满现金空仓"起步, 比真实条线晚一个换仓周期建仓, 入场价也不同。
前期的差额于是全是"起点不同"造成的噪声, 而不是我们想测量的人为干预代价。

所以改成: 直接复制真实条线当前的状态与挂单计划。两者参数逐字相同(只有
tranche_n 决定建仓, 而基准线的 tranche_n 与被参照线一致), 所以状态可以直接
搬。搬完之后两条线从今天起完全同步, 此后任何分歧都只可能来自人为干预。

前提: 被参照的那条线到目前为止没被人动过。脚本会自己检查, 发现有实质干预
(现金校准/出入金/删持仓 且金额非 0) 就拒绝执行 —— 否则等于把污染搬进基准线。

用法:
  python scripts/init_baseline.py --list          # 只看
  python scripts/init_baseline.py                 # 建立(已存在则跳过)
  python scripts/init_baseline.py --force         # 已存在也重建
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_config import PROFILES, state_file  # noqa: E402

LIVE = ROOT / "data" / "live"
ARCHIVE = LIVE / "archive"

# 基准线 -> 它参照的真实条线。两边的 capital 与 tranche-n 必须一致,
# 下面会断言检查, 不一致就说明有人改了配置而没同步改这里。
MIRROR = {
    "base5w_steady": "steady5w",
    "base5w_aggr": "aggr5w",
}

# history 里带 type 的是人为操作记录(set_cash/cash_flow/drop_lot),
# 不带 type 的是自动记账的成交回报。基准线只该保留后者。
def _is_intervention(h):
    return isinstance(h, dict) and "type" in h


def _material(h):
    """这条人为操作有没有真的改动账目 (delta 非 0)"""
    try:
        return abs(float(h.get("delta") or 0)) > 1e-9
    except (TypeError, ValueError):
        return True          # 读不出来就当它有影响, 宁可拦住


def load_state(pid):
    p = LIVE / state_file(pid)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def show():
    print(f"{'基准线':16s} {'参照':10s} {'状态':8s} {'持仓':>4s} {'现金':>10s}  对齐情况")
    for base, src in MIRROR.items():
        b, s = load_state(base), load_state(src)
        if s is None:
            print(f"{base:16s} {src:10s} 参照线还没建立")
            continue
        if b is None:
            print(f"{base:16s} {src:10s} {'未建立':8s} {'-':>4s} {'-':>10s}")
            continue
        same = (abs(b["cash"] - s["cash"]) < 0.01
                and [(l["code"], l["shares"]) for l in (b.get("lots") or [])]
                == [(l["code"], l["shares"]) for l in (s.get("lots") or [])])
        print(f"{base:16s} {src:10s} {'已建立':8s} {len(b.get('lots') or []):>4d} "
              f"{b['cash']:>10,.2f}  {'与参照线一致' if same else '已分歧(正常, 说明参照线被动过)'}")


def check_clean(src, st):
    """参照线是否干净 —— 有实质人为干预就不许复制"""
    bad = [h for h in (st.get("history") or [])
           if _is_intervention(h) and _material(h)]
    if bad:
        print(f"    拒绝: {src} 已有 {len(bad)} 条实质人为干预记录, "
              f"复制过来就等于把污染搬进基准线。")
        for h in bad[:3]:
            print(f"      {h.get('type')} at {h.get('at')} delta={h.get('delta')}")
        print("    要么改用 init_profiles.py 让基准线从满现金另起, "
              "要么接受它与参照线起点不同。")
        return False
    noop = [h for h in (st.get("history") or [])
            if _is_intervention(h) and not _material(h)]
    if noop:
        print(f"    {src} 有 {len(noop)} 条空操作记录(delta=0, 不影响账目), "
              f"复制时会剔除")
    return True


def build_one(base, force):
    src = MIRROR[base]
    bp, sp = PROFILES[base], PROFILES[src]
    if bp["capital"] != sp["capital"] or bp["tranche-n"] != sp["tranche-n"]:
        print(f"[{base}] 拒绝: 与 {src} 的参数不一致 "
              f"({bp['capital']:.0f}/{bp['tranche-n']}只 vs "
              f"{sp['capital']:.0f}/{sp['tranche-n']}只)。"
              f"基准线必须逐字相同才是对照。")
        return False

    st_src = load_state(src)
    if st_src is None:
        print(f"[{base}] 参照线 {src} 还没建立, 跳过")
        return None
    st_old = load_state(base)
    if st_old is not None and not force:
        print(f"[{base}] 已存在 (持仓 {len(st_old.get('lots') or [])} 只, "
              f"现金 {st_old.get('cash', 0):,.2f}) — 跳过。要重建请加 --force")
        return None

    print(f"[{base}] {bp['name']}  <-  复制 {src} ({sp['name']}) 的当前状态")
    if not check_clean(src, st_src):
        return False

    if st_old is not None:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = ARCHIVE / f"state_{base}_{ts}.json"
        shutil.copy2(LIVE / state_file(base), dst)
        print(f"    旧状态已备份 -> archive/{dst.name}")

    new = json.loads(json.dumps(st_src))          # 深拷贝
    # 只留自动记账的成交回报, 剔掉人为操作记录(此处都是 delta=0 的空操作)
    new["history"] = [h for h in (new.get("history") or [])
                      if not _is_intervention(h)]
    new["_mirrored_from"] = {
        "profile": src, "at": datetime.now().isoformat(timespec="seconds"),
        "note": "基准线建立时复制该条线状态, 使两者起点完全一致",
    }
    (LIVE / state_file(base)).write_text(
        json.dumps(new, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"    状态已写入 {state_file(base)} "
          f"(持仓 {len(new.get('lots') or [])} 只, 现金 {new['cash']:,.2f})")

    # 挂单计划也要搬 —— 否则网页上这条线看不到"明天该买什么"
    n = 0
    for pp in sorted(LIVE.glob(f"plan_{src}_*.json")):
        date = pp.name[len(f"plan_{src}_"):]
        tgt = LIVE / f"plan_{base}_{date}"
        plan = json.loads(pp.read_text(encoding="utf-8"))
        plan["profile"] = base
        tgt.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        n += 1
    print(f"    已搬 {n} 份计划文件")
    return True


def main():
    ap = argparse.ArgumentParser(description="把基准线对齐到它参照的真实条线")
    ap.add_argument("--list", action="store_true", help="只打印当前情况")
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    ap.add_argument("--profile", help="只处理这一条基准线")
    a = ap.parse_args()

    if a.list:
        show()
        return 0

    bases = [a.profile] if a.profile else list(MIRROR)
    for b in bases:
        if b not in MIRROR:
            print(f"未知基准线: {b} (可选: {', '.join(MIRROR)})")
            return 2

    fails = [b for b in bases if build_one(b, a.force) is False]
    print()
    show()
    if fails:
        print(f"\n失败: {', '.join(fails)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
