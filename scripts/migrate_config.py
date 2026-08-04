"""换策略参数而不清空账户 —— 替代"改代码 + --init 重置"

为什么需要这个
──────────────
regime_filter 等参数在 FINGERPRINT_KEYS 里, 改了就必须 --init 重置, 而重置会
清空持仓、现金回到本金、历史归零。真金白银在跑的线因此没法调参 —— 每次调参
都要销毁一次真实业绩记录。

但指纹检查是【过度保守】的。看 live_signal.settle() 实际读了什么:

  冻结在 pending 里 (改了不影响在途挂单的结算):
      pending["in_cash"]  <- regime_filter / regime_ma / regime_breadth /
                             regime_confirm 决定
      pending["blocked"]  <- reversal_guard 决定
      pending["ranked"]   <- train_file / pit_universe / label 决定
      pending["is_rebal"]

  settle() 实时读当前参数 (改了会让"实际入账"偏离"你照着执行的计划"):
      HOLD_DAYS      -> 第 636 行判到期
      TRANCHE_N      -> 建续持集与每只分配额
      EXEC_FIELD     -> 取开盘价还是收盘价
      SLIPPAGE / 手续费
      PORTFOLIO_MODE -> lots 的结构含义(分档 vs 整体)

所以只改前一类时, 持仓/现金/历史全都继续有效, 原地改写指纹即可; 只有后一类
需要一个干净的边界。而且后一类的危险窗口只是【挂单在途的那一个交易日】,
不是一整个换仓周期 —— 挂单结算完之后改 hold_days, 只是让存量持仓按新时钟
到期, 那是行为变化不是账目错乱。

必须配套的一件事: 分段记账
──────────────────────────
不重置就保留了历史, 于是这条线的"累计收益"会把两个策略混进一条曲线, 那个
数字就对应不上任何东西了 —— 这正是本项目一直在吃的苦。所以迁移时强制往
state["strategy_epochs"] 追加一段记录, 页面据此分段展示并标出切换点。
【不做分段的话, 宁可重置。】

用法
────
    # 1. 照常改 scripts/live_config.py
    # 2. 看会发生什么(不写任何东西)
    python scripts/migrate_config.py --profile steady5w --dry-run
    # 3. 执行
    python scripts/migrate_config.py --profile steady5w
    # 4. 若提示需要排空(改了 settle 实时读的参数且有持仓)
    python scripts/migrate_config.py --profile steady5w --drain
    #    下一个信号日会只卖不买; 空仓后再跑一次第 3 步完成切换
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from live_config import (FINGERPRINT_KEYS, PROFILES, is_locked,  # noqa: E402
                         signal_args, state_file)

LIVE = ROOT / "data" / "live"
PY = ROOT / ".venv" / "bin" / "python"

# ── 两档参数 ────────────────────────────────────────────────
# 关键事实: pending 在每次成功运行的末尾都会被重新赋值, 所以"没有在途挂单"
# 这个状态在稳态下【几乎不存在】。拿它当迁移前提会让迁移永远没有窗口。
# 因此判据必须落在"settle() 会不会因为这次改动而做出与计划不同的动作"上。

# 第一档: settle() 从不读, 效果已完全冻结进 pending —— 任何时候都能切。
#   regime_* -> 冻结成 pending["in_cash"]
#   reversal_guard -> 冻结成 pending["blocked"]
FREE_KEYS = frozenset({
    "regime_filter", "regime_ma", "regime_breadth", "regime_confirm",
    "reversal_guard",
})

# 第二档: settle() 会实时读, 或会改变 all_dates / LABEL_HORIZON 从而影响
# 到期判定 —— 必须等到"结算不会动任何东西"的时刻才能切。
#   hold_days      -> settle 第 636 行判到期
#   tranche_n      -> 建续持集 + 每只分配额
#   exec_mode      -> 取开盘价还是收盘价
#   slippage       -> 成交价
#   portfolio_mode -> lots 的结构含义(分档 vs 整体)
#   train_file / pit_universe / label -> 换了训练集会改变 all_dates 与
#       LABEL_HORIZON, 而 settle 用 all_dates 算 held_days
BOUNDED_KEYS = frozenset({
    "hold_days", "tranche_n", "exec_mode", "slippage", "portfolio_mode",
    "train_file", "pit_universe", "label",
})

# 漏归类会导致第二档参数被当第一档放过去, 所以在导入时就断言死
_uncovered = set(FINGERPRINT_KEYS) - FREE_KEYS - BOUNDED_KEYS
if _uncovered:
    raise RuntimeError(
        f"指纹里有未归类的参数: {sorted(_uncovered)}。"
        "必须明确它属于 FREE_KEYS 还是 BOUNDED_KEYS —— 判据是 "
        "live_signal.settle() 会不会因它而做出与计划不同的动作。")


def settlement_would_act(st):
    """下一次结算会不会真的动账 —— 第二档参数的迁移窗口判据

    会动账的两种情形:
      1. 手上有持仓 -> settle 要判它们到期没有(读 HOLD_DAYS), 可能卖出
      2. 在途挂单会买入 -> settle 要按 TRANCHE_N 分配额买入
    两者都不成立时(空仓 + 挂单是"继续空仓"), 结算是个空操作, 参数怎么改都
    不会让入账偏离计划。排空流程正是把状态推到这里。
    """
    if st.get("lots"):
        return True, f"有 {len(st['lots'])} 笔持仓, 结算时要判到期"
    p = st.get("pending")
    if p and not p.get("in_cash") and p.get("is_rebal"):
        return True, f"在途挂单({p.get('signal_date')})会按计划买入"
    return False, ""


def current_fingerprint(pid):
    """指纹由 live_signal 自己算, 不在这里重算

    重算一遍的话, 一旦 live_signal 的 argparse 默认值漂移(历史上 hold-days
    和 regime-breadth 都发生过), 这里会静默算出一个错的指纹, 然后"迁移成功"
    却和线上实际跑的参数不一致。
    """
    cmd = [str(PY), str(ROOT / "scripts" / "live_signal.py"),
           *signal_args(pid), "--print-fingerprint"]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"取指纹失败 rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def load_state(pid):
    p = LIVE / state_file(pid)
    if not p.exists():
        raise SystemExit(f"{pid} 还没有状态文件 ({p.name}), 先用 init_profiles.py 建立")
    return p, json.loads(p.read_text(encoding="utf-8"))


def diff_of(old, new):
    keys = set(old) | set(new)
    return {k: (old.get(k), new.get(k)) for k in sorted(keys)
            if old.get(k) != new.get(k)}


def describe(diff):
    return "; ".join(f"{k}: {o!r} -> {n!r}" for k, (o, n) in diff.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--dry-run", action="store_true", help="只报告, 不写任何文件")
    ap.add_argument("--drain", action="store_true",
                    help="请求排空: 下一个信号日只卖不买, 空仓后再跑本工具完成切换")
    ap.add_argument("--note", default="", help="记进 epoch 的备注, 说明为什么改")
    ap.add_argument("--allow-locked", action="store_true",
                    help="基准线默认不许改(它存在的意义就是不动), 确实要改才加")
    args = ap.parse_args()
    pid = args.profile

    if is_locked(pid) and not args.allow_locked:
        raise SystemExit(f"{pid} 是锁定的基准线, 改它就失去参照意义。"
                         f"确实要改请加 --allow-locked")

    path, st = load_state(pid)
    old = st.get("config") or {}
    new = current_fingerprint(pid)
    diff = diff_of(old, new)

    lots = st.get("lots") or []
    pending = st.get("pending")
    would_act, act_why = settlement_would_act(st)
    drain_flag = bool(st.get("drain_requested"))

    print(f"条线      : {pid} ({PROFILES[pid]['name']})")
    print(f"持仓      : {len(lots)} 笔 | 现金 ¥{st.get('cash', 0):,.2f}")
    print(f"在途挂单  : {'信号日 ' + str(pending.get('signal_date')) if pending else '无'}")
    print(f"下次结算  : {'会动账 —— ' + act_why if would_act else '空操作(不动账)'}")
    print(f"排空标记  : {'已置位' if drain_flag else '未置位'}")
    print(f"历史成交  : {len(st.get('history') or [])} 批")
    print(f"已有 epoch: {len(st.get('strategy_epochs') or [])} 段")

    if not diff:
        print("\n参数与状态指纹一致, 无需迁移。")
        if drain_flag and not args.dry_run:
            st["drain_requested"] = False
            _write(path, st, args.dry_run)
            print("顺带清掉了残留的排空标记。")
        return

    bounded = {k: v for k, v in diff.items() if k in BOUNDED_KEYS}
    print(f"\n差异 {len(diff)} 项:")
    for k, (o, n) in diff.items():
        kind = "第二档(需干净边界)" if k in BOUNDED_KEYS else "第一档(已冻结进pending)"
        print(f"  [{kind}] {k}: {o!r} -> {n!r}")

    # ── 判定能不能直接切 ──
    # 只涉及第一档 -> settle 从不读它们, 任何时候都安全, 不看持仓也不看挂单
    if bounded and would_act:
        if args.drain:
            st["drain_requested"] = True
            st["drain_reason"] = {"requested_at": _now(), "diff": describe(bounded)}
            _write(path, st, args.dry_run)
            print(f"\n已置排空标记。下一个信号日这条线只卖不买({len(lots)} 笔持仓), "
                  f"排空后再跑一次本工具完成切换。")
            print("注意: 排空要多付一次往返成本(佣金+滑点), 而手续费是这套系统里"
                  "最大的确定性损失。若这些改动可以等, 不如等持仓自然到期。")
            return
        raise SystemExit(
            f"\n拒绝: 改了 {sorted(bounded)} 这类 settle() 会实时读的参数, 而"
            f"下次结算会动账 ({act_why})。\n"
            "  此刻切换会让实际入账偏离你照着执行的计划。两个选择:\n"
            "    a) 等持仓自然到期清空后再跑本工具 (不多付成本)\n"
            "    b) 加 --drain 让下一个信号日只卖不买, 排空后再切 (多付一次往返成本)")

    # 到这里: 要么只涉及第一档, 要么第二档但结算是空操作 —— 都可以原地切
    epochs = st.get("strategy_epochs") or []
    if not epochs:
        # 迁移工具上线前没有分段记录, 补一条把既有历史归给旧配置, 否则
        # 页面会把切换前的收益也算进新策略名下
        first = _first_history_date(st)
        epochs.append({"since": first, "config": old,
                       "note": "迁移工具上线前的原始配置(起始日按最早成交推断)"})
    epochs.append({
        "since": _next_epoch_since(st),
        "config": new,
        "changed": {k: [o, n] for k, (o, n) in diff.items()},
        "at": _now(),
        "note": args.note or None,
        "had_lots": len(lots),
    })
    st["strategy_epochs"] = epochs
    st["config"] = new
    st["drain_requested"] = False
    _write(path, st, args.dry_run)

    print(f"\n{'[dry-run] 将会' if args.dry_run else '已'}原地切换配置:")
    print(f"  {describe(diff)}")
    print(f"  持仓/现金/历史全部保留 (持仓 {len(lots)} 笔, 现金 ¥{st.get('cash', 0):,.2f})")
    print(f"  strategy_epochs 现有 {len(epochs)} 段, 最新一段自 {epochs[-1]['since']} 起")
    print("\n下一个信号日就会按新参数出计划。注意: 这条线的累计收益从此横跨"
          "多段配置, 看业绩要按 epoch 分段看。")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _first_history_date(st):
    hist = st.get("history") or []
    for h in hist:
        if h.get("signal_date"):
            return h["signal_date"]
    return st.get("last_signal_date") or "unknown"


def _next_epoch_since(st):
    """新配置从下一个信号日起生效; 这里记"已知的最后一个信号日之后"。

    用 last_signal_date 而不是今天: 今天可能不是交易日, 而且切换真正生效的
    时点是下一次出信号, 不是跑本工具的时刻。
    """
    return st.get("last_signal_date") or _now()[:10]


def _write(path, st, dry):
    if dry:
        return
    bak = path.with_name(f"{path.stem}_premigrate_{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy(path, bak)
    path.write_text(json.dumps(st, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"(原状态已备份到 {bak.name})")


if __name__ == "__main__":
    main()
