#!/usr/bin/env python3
"""quant-web 应急维护面的本机客户端 (VPN 挂了、SSH 上不去时用)。

    python3 scripts/admin_client.py status|health|logs|backup|update|restart|daily

凭据 (都是 0600 的普通文件, 所有者必须是当前用户, group/other 不能有任何权限):
    ~/Library/Application Support/QuantAdmin/admin-token        256 位管理 token (与服务器 QA_ADMIN_TOKEN 相同)
    ~/Library/Application Support/QuantAdmin/server-cert.sha256 服务器自签证书 DER 的 SHA-256 (64 位十六进制)
可选环境变量 QA_ADMIN_URL (默认 https://143.89.46.41:8738)。

token 从不上网线: 每个请求用它对 "时间戳\\nnonce\\n方法\\n路径\\nsha256(body)" 做 HMAC-SHA256,
服务器 60 秒内有效、nonce 不可重放。TLS 用证书指纹固定, 不匹配立即中止 (不要用 -k 那种思路)。
restart 会闭环验证: 记旧 PID -> 发 restart -> 轮询 status 直到 MainPID 变化 -> 跑 health。
"""
import argparse
import hashlib
import hmac
import http.client
import json
import os
import secrets
import ssl
import stat
import sys
import time
import urllib.parse

DEFAULT_URL = "https://143.89.46.41:8738"
CONFIG_DIR = os.path.expanduser("~/Library/Application Support/QuantAdmin")
TOKEN_FILE = os.path.join(CONFIG_DIR, "admin-token")
FINGERPRINT_FILE = os.path.join(CONFIG_DIR, "server-cert.sha256")
HDR = "X-QA-"
ACTIONS = ("status", "health", "logs", "backup", "update", "restart", "daily")


class AdminError(Exception):
    """服务器回了 4xx/5xx 或非 JSON —— 与安全类中止 (SystemExit) 区开, restart 轮询里只吞这一种。"""


def read_secret(path):
    try:
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SystemExit("Insecure owner or permissions on %s (need 0600, owned by you)" % path)
        value = open(path, encoding="utf-8").read().strip()
    except OSError as e:
        raise SystemExit("Missing %s: %s" % (path, e))
    if not value:
        raise SystemExit("Empty credential file: " + path)
    return value


def request(method, path, body=None, timeout=330):
    url = urllib.parse.urlparse(os.environ.get("QA_ADMIN_URL", DEFAULT_URL))
    if url.scheme != "https" or not url.hostname:
        raise SystemExit("QA_ADMIN_URL must be an https URL")
    expected = read_secret(FINGERPRINT_FILE).replace(":", "").lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SystemExit("Invalid certificate fingerprint file (need 64 hex chars)")
    token = read_secret(TOKEN_FILE)
    # 自签证书: 不走 CA 校验, 改为固定整张证书的 SHA-256; 连上后先比指纹再发任何东西
    context = ssl._create_unverified_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    conn = http.client.HTTPSConnection(url.hostname, url.port or 443, timeout=timeout, context=context)
    conn.connect()
    actual = hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest()
    if not hmac.compare_digest(actual, expected):
        conn.close()
        raise SystemExit("TLS certificate fingerprint mismatch; refusing connection\n"
                         "  expected %s\n  actual   %s\n"
                         "  证书换过? 通过 VPN/校园网 SSH 重新核对后再更新本地指纹文件。" % (expected, actual))
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode() if body is not None else b""
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    msg = "\n".join((ts, nonce, method, path, hashlib.sha256(raw).hexdigest()))
    sig = hmac.new(token.encode(), msg.encode(), hashlib.sha256).hexdigest()
    headers = {HDR + "Timestamp": ts, HDR + "Nonce": nonce, HDR + "Signature": sig,
               "Accept": "application/json", "User-Agent": "quant-admin-client/1"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    conn.request(method, path, body=raw, headers=headers)
    resp = conn.getresponse()
    payload = resp.read(2_000_000)
    conn.close()
    try:
        data = json.loads(payload.decode())
    except Exception:  # noqa: BLE001
        raise AdminError("Server returned a non-JSON response (HTTP %d)" % resp.status)
    if resp.status >= 400:
        raise AdminError("HTTP %d: %s" % (resp.status, data.get("error", "request failed")))
    return data


def pp(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="quant-web VPN-fallback admin client")
    ap.add_argument("action", choices=ACTIONS)
    args = ap.parse_args()

    if args.action == "restart":
        before = request("GET", "/api/admin/status")
        old_pid = before["web"].get("main_pid")
        pp(request("POST", "/api/admin/action", {"action": "restart"}))
        for _ in range(45):
            time.sleep(1)
            try:
                st = request("GET", "/api/admin/status")
            except (OSError, http.client.HTTPException, AdminError):
                continue
            web = st["web"]
            if web.get("main_pid") and web.get("main_pid") != old_pid and web.get("active_state") == "active" \
                    and web.get("ok"):
                print("Restart verified: PID %s -> %s, http %s" % (old_pid, web["main_pid"], web.get("http_status")))
                pp(request("POST", "/api/admin/action", {"action": "health"}))
                return
        raise SystemExit("Restart was scheduled but quant-web did not come back healthy within 45 seconds; "
                         "run `logs` to see why")

    if args.action in ("status", "logs"):
        data = request("GET", "/api/admin/" + args.action)
    else:
        data = request("POST", "/api/admin/action", {"action": args.action})
    if args.action == "logs":
        for key in ("web", "daily"):
            print("===== %s (last %d lines) =====" % (key, len(data.get(key, []))))
            print("\n".join(data.get(key, [])))
    elif args.action == "update":
        for line in data.get("output", []):
            print("  " + line)
        print("updated=%s restart_required=%s deployed=%s"
              % (data.get("updated"), data.get("restart_required"), (data.get("deployed_commit") or "")[:10]))
        if data.get("restart_required"):
            print("-> 现在跑: python3 scripts/admin_client.py restart")
    else:
        pp(data)


if __name__ == "__main__":
    try:
        main()
    except AdminError as e:
        raise SystemExit(str(e))
