"""应急维护面 (scripts/admin_plane.py) 的安全不变量。

不起 TLS (127.0.0.1 --insecure-http 等价路径), 只测鉴权与白名单:
  无签名/错 token/过期/重放/浏览器 Origin -> 401; 连错 5 次 -> 429; token 未配置 -> 503;
  非白名单动作 -> 403; 请求体多字段 -> 400; 正确签名 status/health 可用且每次请求都落审计。
"""
import hashlib
import hmac
import http.client
import json
import secrets
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import admin_plane as ap  # noqa: E402

TOKEN = "t" * 48


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    state = tmp_path_factory.mktemp("qa_state")
    ap.ADMIN_TOKEN = TOKEN
    ap.STATE_DIR = state
    ap.AUDIT_PATH = state / "audit.log"
    ap.MARKER_PATH = state / "deployed-commit"
    ap.BACKUP_DIR = state / "backups"
    ap.UPDATE_SCRIPT = state / "missing_update.sh"
    srv = ap.make_server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset_limits():
    ap._FAILURES.clear()
    ap._NONCES.clear()
    yield


def call(addr, method, path, body=None, token=TOKEN, ts=None, nonce=None, headers=None, sign=True):
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode() if body is not None else b""
    ts = str(int(time.time())) if ts is None else ts
    nonce = secrets.token_hex(16) if nonce is None else nonce
    h = dict(headers or {})
    if sign:
        msg = "\n".join((ts, nonce, method, path, hashlib.sha256(raw).hexdigest()))
        h.update({ap.HDR_TS: ts, ap.HDR_NONCE: nonce,
                  ap.HDR_SIG: hmac.new(token.encode(), msg.encode(), hashlib.sha256).hexdigest()})
    if body is not None:
        h["Content-Type"] = "application/json"
    c = http.client.HTTPConnection(*addr, timeout=10)
    c.request(method, path, body=raw, headers=h)
    r = c.getresponse()
    data = json.loads(r.read().decode())
    c.close()
    return r.status, data


def test_unsigned_and_wrong_token_rejected(server):
    assert call(server, "GET", "/api/admin/status", sign=False)[0] == 401
    assert call(server, "GET", "/api/admin/status", token="x" * 48)[0] == 401


def test_stale_timestamp_rejected(server):
    assert call(server, "GET", "/api/admin/status", ts=str(int(time.time()) - 120))[0] == 401


def test_replayed_nonce_rejected(server):
    nonce = secrets.token_hex(16)
    assert call(server, "GET", "/api/admin/status", nonce=nonce)[0] == 200
    assert call(server, "GET", "/api/admin/status", nonce=nonce)[0] == 401


def test_browser_origin_rejected_even_with_valid_signature(server):
    assert call(server, "GET", "/api/admin/status", headers={"Origin": "https://evil.example"})[0] == 401


def test_rate_limit_after_five_failures(server):
    for _ in range(5):
        assert call(server, "GET", "/api/admin/status", token="bad" * 16)[0] == 401
    assert call(server, "GET", "/api/admin/status")[0] == 429  # 正确签名也被拒: 限速按 IP


def test_status_ok_and_audited(server):
    code, data = call(server, "GET", "/api/admin/status")
    assert code == 200 and data["service"] == "quant-admin"
    assert {"web", "daily", "deployed_commit", "disk_free_gb"} <= set(data)
    rows = [json.loads(l) for l in ap.AUDIT_PATH.read_text().splitlines()]
    assert rows and rows[-1]["action"] == "status" and rows[-1]["result"] == "ok"
    assert oct(ap.AUDIT_PATH.stat().st_mode & 0o777) == "0o600"


def test_action_allowlist_and_shape(server):
    assert call(server, "POST", "/api/admin/action", {"action": "shell"})[0] == 403
    assert call(server, "POST", "/api/admin/action", {"action": "backup", "path": "/etc"})[0] == 400
    assert call(server, "POST", "/api/admin/action", ["update"])[0] == 400
    code, data = call(server, "POST", "/api/admin/action", {"action": "health"})
    assert code in (200, 503) and "checks" in data and "web_http" in data["checks"]


def test_update_without_script_fails_closed(server):
    code, data = call(server, "POST", "/api/admin/action", {"action": "update"})
    assert code == 500 and "not installed" in data["error"]


def test_unknown_paths_404(server):
    assert call(server, "GET", "/api/admin/exec")[0] == 404
    assert call(server, "GET", "/")[0] == 404
    assert call(server, "PUT", "/api/admin/action", {"action": "health"})[0] == 404


def test_disabled_without_token(server):
    old = ap.ADMIN_TOKEN
    ap.ADMIN_TOKEN = ""
    try:
        assert call(server, "GET", "/api/admin/status", token="x" * 48)[0] == 503
    finally:
        ap.ADMIN_TOKEN = old
