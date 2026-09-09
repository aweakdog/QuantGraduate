#!/usr/bin/env python3
"""quant-web 无 VPN 应急维护面 (VPN fallback admin plane).

背景: eez041 的 sshd 有来源白名单 (AllowUsers *@143.89.*.* 等), 校 VPN 一失效就
SSH 不上; 但 8xxx 段的 HTTP(S) 端口公网可达。本进程在 8738 上提供一条
**能力有限、可审计、可撤销** 的维护通道, 只做几个编译在代码里的固定动作:

    status   GET  /api/admin/status   版本/进程/磁盘/日更链状态
    logs     GET  /api/admin/logs     web_server.log 与 daily_rebuild.log 末 100 行 (脱敏)
    health   POST /api/admin/action   quant-web 是否在响应、依赖是否齐
    backup   POST /api/admin/action   打包 data/live 的 state/plan/confirm json
    update   POST /api/admin/action   仅从固定仓库 main 快进更新代码目录 (admin_update.sh)
    restart  POST /api/admin/action   systemctl --user restart quant-web.service
    daily    POST /api/admin/action   systemctl --user start quant-daily.service (重跑日更链)

不是 web shell: 不接受命令/路径/仓库/分支/任何参数; 请求体必须恰好是 {"action": ...}。

设计要点 (照 English/VPN_FALLBACK_MAINTENANCE_PLAYBOOK.md):
- 独立进程 + 独立 systemd unit: quant-web 崩了 (哪怕 web_server.py 语法错) 也能用它 update+restart 自救;
  更新器把 admin_* 三个文件排除在外, 一次仓库更新改不了应急边界, 改这里必须走 SSH。
- 鉴权: 独立 256 位 token 只存服务器 ~/.config/quant-admin.env (0600) 与本机受限文件, 不上网线。
  每个请求 HMAC-SHA256(token, ts\\nnonce\\nMETHOD\\npath\\nsha256(body)); ts ±60s; nonce 120s 内不可重放;
  常量时间比较; 单 IP 10 分钟 5 次错误签名即 429; 带浏览器 Origin 一律 401; 响应 no-store。
- TLS 自签, 客户端固定证书 SHA-256 指纹 (不是 curl -k)。
- 审计: ~/.local/state/quant-admin/audit.log (0600) 每次请求一行 JSON, 不记 token/签名/请求头。

运行: 由 quant-admin.service 拉起 (见 scripts/admin_bootstrap.sh); 环境变量:
  QA_ADMIN_TOKEN (必填, ≥32 字符, 否则接口 503 "admin interface disabled")
  QA_CERT / QA_KEY   证书与私钥路径 (默认 ~/.config/quant-admin/{cert,key}.pem)
  QA_LIVE_DIR        运行目录 (默认 ~/quant-strategy)
  QA_STATE_DIR       状态目录 (默认 ~/.local/state/quant-admin)
  QA_WEB_PORT        quant-web 端口 (默认 8737)
测试用 --insecure-http 只允许绑 127.0.0.1。
"""
import argparse
import hashlib
import hmac
import http.client
import http.server
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
LIVE = Path(os.environ.get("QA_LIVE_DIR") or HOME / "quant-strategy")
STATE_DIR = Path(os.environ.get("QA_STATE_DIR") or HOME / ".local" / "state" / "quant-admin")
ADMIN_TOKEN = os.environ.get("QA_ADMIN_TOKEN", "")
WEB_PORT = int(os.environ.get("QA_WEB_PORT", "8737"))
WEB_UNIT = "quant-web.service"
DAILY_UNIT = "quant-daily.service"
UPDATE_SCRIPT = LIVE / "scripts" / "admin_update.sh"
AUDIT_PATH = STATE_DIR / "audit.log"
MARKER_PATH = STATE_DIR / "deployed-commit"
BACKUP_DIR = STATE_DIR / "backups"
WEB_LOG = LIVE / "data" / "live" / "web_server.log"
DAILY_LOG = LIVE / "data" / "live" / "daily_rebuild.log"
WEB_ENV = HOME / ".config" / "quant-web.env"
HDR_TS, HDR_NONCE, HDR_SIG = "X-QA-Timestamp", "X-QA-Nonce", "X-QA-Signature"
ACTIONS = ("health", "backup", "update", "restart", "daily")
STARTED_AT = time.time()

_LOCK = threading.Lock()          # 保护 _FAILURES / _NONCES
_FAILURES: dict[str, list[float]] = {}
_NONCES: dict[str, float] = {}
_ACTION_LOCK = threading.Lock()   # 同一时刻只跑一个改状态的动作


# ---------- 审计 ----------
def audit(ip, action, result, detail=""):
    row = {"ts": datetime.now().isoformat(timespec="seconds"), "ip": ip,
           "action": action, "result": result, "detail": str(detail)[:300]}
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.chmod(AUDIT_PATH, 0o600)
    except OSError:
        pass


# ---------- 系统信息 ----------
def _run(cmd, timeout=20):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def unit_info(unit):
    """systemctl --user show 的几个字段; systemd 不在时返回 unknown 而不是抛。"""
    try:
        rc, out, _ = _run(["systemctl", "--user", "show", unit, "-p",
                           "ActiveState,SubState,MainPID,ExecMainStartTimestamp,Result"])
    except (OSError, subprocess.TimeoutExpired):
        return {"active_state": "unknown"}
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return {"active_state": kv.get("ActiveState", "unknown"), "sub_state": kv.get("SubState"),
            "main_pid": int(kv.get("MainPID") or 0) or None,
            "since": kv.get("ExecMainStartTimestamp") or None, "result": kv.get("Result")}


def deployed_commit():
    try:
        v = MARKER_PATH.read_text(encoding="ascii").strip()
        return v if re.fullmatch(r"[0-9a-f]{40}", v) else None
    except OSError:
        return None


def web_probe():
    """本机 quant-web 是否在应答。首页要登录会回 401, 那也是"活着"; >=500 或连不上才算坏。"""
    try:
        c = http.client.HTTPConnection("127.0.0.1", WEB_PORT, timeout=5)
        c.request("GET", "/", headers={"User-Agent": "quant-admin-probe"})
        code = c.getresponse().status
        c.close()
        return {"reachable": True, "http_status": code, "ok": code < 500}
    except OSError as e:
        return {"reachable": False, "error": type(e).__name__, "ok": False}


def status_data():
    live = LIVE / "data" / "live"
    states = sorted(live.glob("state_*.json")) if live.is_dir() else []
    latest = max((p.stat().st_mtime for p in states), default=None)
    disk = shutil.disk_usage(str(HOME))
    return {
        "ok": True, "service": "quant-admin", "server_time": datetime.now().isoformat(timespec="seconds"),
        "admin_pid": os.getpid(), "admin_uptime_seconds": int(time.time() - STARTED_AT),
        "deployed_commit": deployed_commit(),
        "web": {**unit_info(WEB_UNIT), **web_probe()},
        "daily": unit_info(DAILY_UNIT),
        "live_state_files": len(states),
        "live_state_latest": datetime.fromtimestamp(latest).isoformat(timespec="seconds") if latest else None,
        "disk_free_gb": round(disk.free / 1e9, 1),
    }


def health_data():
    web = web_probe()
    unit = unit_info(WEB_UNIT)
    checks = {
        "web_http": web, "web_unit_active": unit.get("active_state") == "active",
        "daily_unit_state": unit_info(DAILY_UNIT).get("active_state"),
        "update_script": UPDATE_SCRIPT.is_file(),
        "venv_python": (LIVE / ".venv" / "bin" / "python").exists(),
        "deployed_marker": deployed_commit() is not None,
        "admin_token_configured": len(ADMIN_TOKEN) >= 32,
        "disk_free_gb": round(shutil.disk_usage(str(HOME)).free / 1e9, 1),
    }
    ok = web["ok"] and checks["web_unit_active"] and checks["update_script"] and checks["venv_python"]
    return {"ok": ok, "checks": checks}


def _secret_values():
    vals = [ADMIN_TOKEN] if ADMIN_TOKEN else []
    try:
        for line in WEB_ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 4:
                    vals.append(v)
    except OSError:
        pass
    return vals


def log_tail(path, n=100):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    for v in _secret_values():
        text = text.replace(v, "[REDACTED]")
    return text.splitlines()[-n:]


def logs_data():
    return {"ok": True, "web": log_tail(WEB_LOG), "daily": log_tail(DAILY_LOG)}


# ---------- 改状态的动作 ----------
def do_backup():
    """打包 data/live 里的账本 (state/plan/confirm json), 不含预测缓存等大文件。"""
    live = LIVE / "data" / "live"
    BACKUP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = "live-%s.tar.gz" % datetime.now().strftime("%Y%m%d-%H%M%S")
    target = BACKUP_DIR / name
    files = [p for pat in ("state_*.json", "plan_*.json", "confirm_*.json") for p in live.glob(pat)]
    with tarfile.open(target, "w:gz") as tar:
        for p in sorted(files):
            tar.add(p, arcname=p.name)
    os.chmod(target, 0o600)
    with tarfile.open(target, "r:gz") as tar:
        n = len(tar.getmembers())
    if n != len(files):
        target.unlink()
        raise RuntimeError("backup member count mismatch")
    return {"file": name, "files": n, "bytes": target.stat().st_size}


def do_update():
    if not UPDATE_SCRIPT.is_file():
        raise RuntimeError("update script is not installed")
    env = {"HOME": str(HOME), "PATH": "/usr/local/bin:/usr/bin:/bin",
           "QA_LIVE_DIR": str(LIVE), "QA_STATE_DIR": str(STATE_DIR), "LANG": "C.UTF-8"}
    r = subprocess.run(["/bin/bash", str(UPDATE_SCRIPT)], cwd=str(LIVE), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300, check=False)
    out = (r.stdout or "").splitlines()[-40:]
    if r.returncode:
        raise RuntimeError("update failed (%d): %s" % (r.returncode, " | ".join(out[-6:])))
    changed = not any(line.startswith("already current:") for line in out)
    return {"updated": changed, "restart_required": changed, "output": out, "deployed_commit": deployed_commit()}


def do_restart():
    def later():
        time.sleep(1.0)
        subprocess.run(["systemctl", "--user", "restart", WEB_UNIT], check=False, timeout=60)
    threading.Thread(target=later, daemon=True).start()
    return {"restart_scheduled": True, "unit": WEB_UNIT, "old_main_pid": unit_info(WEB_UNIT).get("main_pid")}


def do_daily():
    state = unit_info(DAILY_UNIT).get("active_state")
    if state in ("active", "activating", "reloading"):
        raise RuntimeError("daily chain already running (%s)" % state)
    rc, out, err = _run(["systemctl", "--user", "start", "--no-block", DAILY_UNIT], timeout=30)
    if rc:
        raise RuntimeError("systemctl start failed: %s" % (err or out)[-200:])
    return {"started": True, "unit": DAILY_UNIT}


# ---------- HTTP ----------
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30
    server_version = "quant-admin"
    sys_version = ""

    def log_message(self, fmt, *args):  # 访问日志走审计, 不刷 stderr
        pass

    # --- 工具 ---
    def send_json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def fail(self, msg, code=400):
        self.send_json({"error": msg}, code)

    def body_raw(self, limit=4096):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > limit:
            return b""
        return self.rfile.read(n)

    def qpath(self):
        return self.path.split("?", 1)[0]

    def ip(self):
        return self.client_address[0]

    # --- 鉴权 ---
    def admin_auth(self, action, raw_body=b""):
        ip = self.ip()
        if self.headers.get("Origin"):
            audit(ip, action, "rejected", "browser origin not allowed")
            self.fail("unauthorized", 401)
            return False
        if len(ADMIN_TOKEN) < 32:
            self.fail("admin interface disabled", 503)
            return False
        now = time.time()
        with _LOCK:
            if len(_FAILURES) > 1000:
                _FAILURES.clear()
            recent = [t for t in _FAILURES.get(ip, []) if now - t < 600]
            _FAILURES[ip] = recent
            if len(recent) >= 5:
                audit(ip, action, "rate_limited")
                self.fail("too many failed attempts", 429)
                return False
        ts = self.headers.get(HDR_TS) or ""
        nonce = self.headers.get(HDR_NONCE) or ""
        supplied = self.headers.get(HDR_SIG) or ""
        try:
            fresh = abs(now - int(ts)) <= 60
        except ValueError:
            fresh = False
        shape_ok = bool(re.fullmatch(r"[0-9a-f]{32}", nonce) and re.fullmatch(r"[0-9a-f]{64}", supplied))
        msg = "\n".join((ts, nonce, self.command, self.qpath(), hashlib.sha256(raw_body).hexdigest()))
        expected = hmac.new(ADMIN_TOKEN.encode(), msg.encode(), hashlib.sha256).hexdigest()
        if not fresh or not shape_ok or not hmac.compare_digest(supplied, expected):
            with _LOCK:
                _FAILURES.setdefault(ip, []).append(now)
            audit(ip, action, "unauthorized", "invalid request signature")
            self.fail("unauthorized", 401)
            return False
        with _LOCK:
            for old, used_at in list(_NONCES.items()):
                if now - used_at > 120:
                    _NONCES.pop(old, None)
            if nonce in _NONCES:
                audit(ip, action, "rejected", "replayed nonce")
                self.fail("unauthorized", 401)
                return False
            _NONCES[nonce] = now
            _FAILURES.pop(ip, None)
        return True

    # --- 路由 ---
    def do_GET(self):
        p = self.qpath()
        action = {"/api/admin/status": "status", "/api/admin/logs": "logs"}.get(p)
        if action is None:
            return self.fail("not found", 404)
        if not self.admin_auth(action):
            return
        try:
            data = status_data() if action == "status" else logs_data()
            audit(self.ip(), action, "ok")
            return self.send_json(data)
        except Exception as e:  # noqa: BLE001
            audit(self.ip(), action, "error", type(e).__name__)
            return self.fail("admin action failed", 500)

    def do_POST(self):
        if self.qpath() != "/api/admin/action":
            self.body_raw()
            return self.fail("not found", 404)
        raw = self.body_raw()
        if not self.admin_auth("action", raw):
            return
        try:
            body = json.loads(raw.decode())
        except Exception:  # noqa: BLE001
            body = None
        if not isinstance(body, dict) or set(body) != {"action"}:
            return self.fail("exactly one action is required")
        action = body["action"]
        if action not in ACTIONS:
            audit(self.ip(), str(action)[:40], "rejected", "not allowlisted")
            return self.fail("action not allowed", 403)
        if action == "health":
            result = health_data()
            audit(self.ip(), action, "ok" if result["ok"] else "degraded")
            return self.send_json(result, 200 if result["ok"] else 503)
        if not _ACTION_LOCK.acquire(blocking=False):
            return self.fail("another admin action is running", 409)
        try:
            fn = {"backup": do_backup, "update": do_update, "restart": do_restart, "daily": do_daily}[action]
            result = {"ok": True, **fn()}
            audit(self.ip(), action, "ok", result.get("deployed_commit") or result.get("file") or "")
            return self.send_json(result)
        except subprocess.TimeoutExpired:
            audit(self.ip(), action, "error", "timeout")
            return self.fail("admin action timed out", 504)
        except Exception as e:  # noqa: BLE001
            audit(self.ip(), action, "error", "%s: %s" % (type(e).__name__, str(e)[:200]))
            sys.stderr.write("admin %s failed: %s\n" % (action, e))
            return self.fail("admin action failed: %s" % str(e)[:200], 500)
        finally:
            _ACTION_LOCK.release()

    def do_PUT(self):
        self.fail("not found", 404)

    do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_PUT


def make_server(host, port, certfile=None, keyfile=None):
    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile, keyfile)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    return srv


def main():
    ap = argparse.ArgumentParser(description="quant-web VPN-fallback admin plane")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8738)
    ap.add_argument("--cert", default=os.environ.get("QA_CERT") or str(HOME / ".config" / "quant-admin" / "cert.pem"))
    ap.add_argument("--key", default=os.environ.get("QA_KEY") or str(HOME / ".config" / "quant-admin" / "key.pem"))
    ap.add_argument("--insecure-http", action="store_true",
                    help="不启用 TLS, 仅供测试; 只允许绑 127.0.0.1")
    args = ap.parse_args()
    if args.insecure_http and args.host != "127.0.0.1":
        raise SystemExit("--insecure-http only allowed with --host 127.0.0.1")
    if len(ADMIN_TOKEN) < 32:
        sys.stderr.write("WARN: QA_ADMIN_TOKEN missing/short -> admin interface disabled (503)\n")
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    srv = make_server(args.host, args.port, None if args.insecure_http else args.cert,
                      None if args.insecure_http else args.key)
    sys.stderr.write("quant-admin listening on %s:%d (%s)\n"
                     % (args.host, args.port, "http" if args.insecure_http else "https"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
