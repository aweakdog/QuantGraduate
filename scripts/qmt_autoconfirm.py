"""日更之后, 用 QMT 真实成交替 QMT 线完成「确认成交」(041, systemd 定时 18:50 / 20:50)。

流程 (每条 QMT 线):
  1. 线状态不在「待确认」-> 无事可做
  2. 取执行日收盘后的最终 QMT 快照, qmt_sync.build_autoconfirm 转成回报
  3. 有任何对不上 (线外代码/持仓不一致/没有快照) -> 不确认, 停在待确认并告警
  4. 否则写 confirm_<pid>_<时间>_qmt.json, 调 live_signal --confirm (与网页「确认成交」同一路径)
写账仍只经 live_signal; 本脚本自己不改 state。日志 data/live/qmt/autoconfirm.jsonl。
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qmt_sync  # noqa: E402
from live_config import display_name, signal_args, state_file  # noqa: E402

LIVE = ROOT / "data" / "live"


def daily_running():
    """日更链在跑时不动 (它会写同一份 state)。查不到 systemd 时当作没在跑。"""
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "quant-daily.service"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.stdout.strip() in ("active", "activating")


def _log(rec):
    p = LIVE / "qmt" / "autoconfirm.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def run_one(pid, live=LIVE, dry_run=False, runner=subprocess.run, send=None):
    st_p = live / state_file(pid)
    if not st_p.exists():
        return {"pid": pid, "status": "no_state"}
    st = json.loads(st_p.read_text(encoding="utf-8"))
    aw = st.get("awaiting_confirm")
    if not aw:
        return {"pid": pid, "status": "nothing_to_confirm"}
    exec_date = aw["exec_date"]
    plan_p = live / f"plan_{pid}_{aw['signal_date']}.json"
    plan = json.loads(plan_p.read_text(encoding="utf-8")) if plan_p.exists() else None
    snap = qmt_sync.final_snapshot_for(live, exec_date)
    fills, probs, notes = qmt_sync.build_autoconfirm(snap, st, plan, exec_date)
    if plan is None:
        probs = [f"找不到计划文件 {plan_p.name}"] + probs
    rec = {"at": datetime.now().isoformat(timespec="seconds"), "pid": pid, "exec_date": exec_date,
           "signal_date": aw["signal_date"], "snapshot_at": (snap or {}).get("probe_at"),
           "fills": fills, "problems": probs, "notes": notes}
    name = display_name(pid)
    if probs:
        rec["status"] = "blocked"
        qmt_sync.alert_once(live, f"{exec_date}:autoconfirm_blocked:{pid}",
                            f"【QMT】{name} {exec_date} 没有自动确认, 原因: " + "; ".join(probs)
                            + "。请人工核对后在网页上确认成交。", send=send)
        return rec
    if dry_run:
        rec["status"] = "dry_run"
        return rec
    cf = live / f"confirm_{pid}_{datetime.now():%Y%m%d_%H%M%S}_qmt.json"
    cf.write_text(json.dumps(fills, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "live_signal.py")] + signal_args(pid) + ["--confirm", str(cf)]
    r = runner(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    after = json.loads(st_p.read_text(encoding="utf-8"))
    ok = r.returncode == 0 and not after.get("awaiting_confirm")
    rec.update(status="confirmed" if ok else "failed", confirm_file=cf.name, returncode=r.returncode,
               output_tail=(r.stdout or "")[-1500:] + (r.stderr or "")[-800:])
    if ok:
        desc = "无成交" if not fills else "、".join(
            f"{'买' if f['action'] == 'buy' else '卖'}{f['code']}×{f['shares']}@{f['price']}" for f in fills)
        qmt_sync.alert_once(live, f"{exec_date}:autoconfirm_ok:{pid}",
                            f"【QMT】{name} {exec_date} 已按券商真实成交自动确认: {desc}。", send=send)
    else:
        qmt_sync.alert_once(live, f"{exec_date}:autoconfirm_failed:{pid}",
                            f"【QMT】{name} {exec_date} 自动确认执行失败(rc={r.returncode}), 线仍在待确认, 请查看。",
                            send=send)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="只判定能否确认, 不写回报、不调 live_signal")
    args = ap.parse_args(argv)
    if daily_running():
        print("日更链正在运行, 本次跳过")
        return 0
    rc = 0
    for pid in qmt_sync.QMT_LINES:
        rec = run_one(pid, dry_run=args.dry_run)
        if rec.get("status") not in ("nothing_to_confirm", "no_state"):
            _log(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "output_tail"}, ensure_ascii=False, default=str))
        rc |= rec.get("status") in ("blocked", "failed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
