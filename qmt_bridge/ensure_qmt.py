"""确保 QMT 交易账户在线 (Windows 计划任务: 周一至五 09:10 --force 每日重启重登; 13:00 只在不健康时重启)。

  .venv\\Scripts\\python.exe -m qmt_bridge.ensure_qmt --force

为什么要每天重启: 10-22 实测 QMT 隔一段时间交易连接会掉成「等待登录」(查询照常、下单无回执), 且不会自己恢复;
而 QMT 存了密码, 重新打开就自动登录。所以开盘前强制重启一次 = 每天自动重新登录。
健康 = 能连上 miniQMT 且交易账户状态 0。探测放在子进程里做 (xtquant 线程可能卡住退出)。
重启后不管成败都推一份 manual 快照给 041: 失败会触发离线告警 (邮件 + 企业微信, 同日一次)。
必须由计划任务在登录用户的桌面会话里运行 (SSH 会话启动的 GUI 看不见)。
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROBE = r"""
import json, os, sys, time
sys.path.insert(0, r"D:\qmtcode")
out = {"connect": None, "status": None}
try:
    from xtquant.xttrader import XtQuantTrader
    from qmt_bridge.readonly_probe import DEFAULT_USERDATA, ReadOnlyTrader, account_status, connect_with_timeout
    t = ReadOnlyTrader(XtQuantTrader(DEFAULT_USERDATA, int(time.time()) % 100000 + 900000))
    t.start()
    out["connect"] = connect_with_timeout(t, 15)
    if out["connect"] == 0:
        infos = [i for i in (t.query_account_infos() or []) if getattr(i, "account_type", None) == 2]
        if infos:
            out["status"] = account_status(t, infos[0].account_id)
except Exception as e:
    out["error"] = repr(e)[:200]
print(json.dumps(out)); sys.stdout.flush(); os._exit(0)
"""


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def qmt_exe():
    for p in Path("D:/").iterdir():
        exe = p / "bin.x64" / "XtItClient.exe"
        if exe.exists():
            return exe
    raise SystemExit("找不到 QMT 安装目录 (D:\\*\\bin.x64\\XtItClient.exe)")


def probe():
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", PROBE], capture_output=True, text=True, timeout=60)
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {"connect": None, "status": None, "error": repr(e)[:200]}


def healthy(p):
    return p.get("connect") == 0 and p.get("status") == 0


def kill_qmt():
    for name in ("XtMiniQmt.exe", "XtItClient.exe"):
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)


def launch_qmt():
    exe = qmt_exe()
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([str(exe)], cwd=str(exe.parent), creationflags=flags, close_fds=True)


def ensure(force, probe_fn=probe, kill_fn=kill_qmt, launch_fn=launch_qmt, sleep_fn=time.sleep, wait_s=180, every_s=15):
    """返回 (是否健康, 是否重启过, 最后一次探测)。"""
    before = probe_fn()
    if healthy(before) and not force:
        return True, False, before
    kill_fn()
    sleep_fn(5)
    launch_fn()
    waited, last = 0, before
    while waited < wait_s:
        sleep_fn(every_s)
        waited += every_s
        last = probe_fn()
        if healthy(last):
            return True, True, last
    return False, True, last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="不管当前是否健康都重启 (每日开盘前)")
    a = ap.parse_args(argv)
    ok, relaunched, last = ensure(a.force)
    log(f"force={a.force} relaunched={relaunched} healthy={ok} probe={last}")
    if relaunched:
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "qmt_bridge.push_snapshot", "--kind", "manual"],
                           capture_output=True, text=True, timeout=120, cwd=str(Path(__file__).resolve().parents[1]))
        log(f"pushed snapshot: {(r.stdout or r.stderr).strip()[:200]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
