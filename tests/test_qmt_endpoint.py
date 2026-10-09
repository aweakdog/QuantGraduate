"""/api/qmt/snapshot: 不走登录但必须签名正确; 只存数据, 离线/不一致发告警。"""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

KEY = b"z" * 64


@pytest.fixture
def client(tmp_path, monkeypatch):
    import web_server as ws
    from fastapi.testclient import TestClient
    monkeypatch.setattr(ws, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(ws.qmt_sync, "load_key", lambda path=None: KEY)
    monkeypatch.setattr(ws, "_state_of", lambda pid: {"cash": 100000.0, "lots": []} if pid == "qmt10w" else None)
    alerts = []
    monkeypatch.setattr(ws, "_qmt_alert", lambda key, text: alerts.append((key, text)) or True)
    c = TestClient(ws.app)
    c.alerts = alerts
    return c


def _post(client, snap, key=KEY, ts=None, body=None):
    from qmt_sync import sign
    body = body if body is not None else json.dumps(snap).encode()
    ts = str(int(time.time())) if ts is None else ts
    return client.post("/api/qmt/snapshot", content=body,
                       headers={"x-qmt-ts": ts, "x-qmt-sig": sign(body, ts, key), "content-type": "application/json"})


def _snap(**kw):
    s = {"probe_at": "2026-10-09T15:10:00", "kind": "close", "account_online": True,
         "asset": {"cash": 503121.8}, "positions": [], "orders": [], "trades": []}
    s.update(kw)
    return s


def test_signed_snapshot_is_stored_and_reconciled(client, tmp_path):
    r = _post(client, _snap())
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["stored"] == "snap_20261009_151000_close.json"
    assert d["reconcile"]["qmt10w"]["positions_match"] is True
    assert d["reconcile"]["qmt10w"]["outside_cash"] == 403121.8
    assert (tmp_path / "qmt" / "snap_20261009_151000_close.json").exists()
    assert client.alerts == []


def test_bad_signature_stale_or_garbage_rejected(client, tmp_path):
    assert _post(client, _snap(), key=b"wrong" * 10).status_code == 401
    assert _post(client, _snap(), ts=str(int(time.time()) - 3600)).status_code == 401
    assert client.post("/api/qmt/snapshot", content=b"{}").status_code == 401      # 无签名头
    assert _post(client, None, body=b"not json").status_code == 400
    assert _post(client, {"probe_at": "2026-10-09T15:10:00", "kind": "noon"}).status_code == 400
    assert not (tmp_path / "qmt").exists()


def test_offline_and_mismatch_trigger_alerts(client):
    assert _post(client, _snap(kind="morning", probe_at="2026-10-09T09:35:00", account_online=False,
                               error="无账户")).status_code == 200
    assert client.alerts[-1][0] == "2026-10-09:morning:offline"
    assert _post(client, _snap(positions=[{"stock_code": "600000.SH", "volume": 100}])).status_code == 200
    assert client.alerts[-1][0] == "2026-10-09:close:mismatch:qmt10w"


def test_status_needs_login(client):
    assert client.get("/api/qmt/status").status_code == 401


def _signed(client, path, payload, key=KEY):
    from qmt_sync import sign
    body = json.dumps(payload).encode()
    ts = str(int(time.time()))
    return client.post(path, content=body, headers={"x-qmt-ts": ts, "x-qmt-sig": sign(body, ts, key),
                                                    "content-type": "application/json"})


def test_ticket_requires_signature_and_returns_problems_when_not_today(client, tmp_path, monkeypatch):
    import web_server as ws
    st = {"cash": 100000.0, "lots": [], "pending": {"signal_date": "2026-10-08"}, "awaiting_confirm": None}
    monkeypatch.setattr(ws, "_state_of", lambda pid: st)
    monkeypatch.setattr(ws, "_exec_date_of", lambda s: "2026-10-09")
    (tmp_path / "plan_qmt10w_2026-10-08.json").write_text(json.dumps(
        {"signal_date": "2026-10-08", "in_cash": False, "equity": 100000.0,
         "buy": [{"code": "600000", "shares": 300, "ref_close": 10.0}], "sell": []}))
    assert client.post("/api/qmt/ticket", content=b'{"profile": "qmt10w"}').status_code == 401
    r = _signed(client, "/api/qmt/ticket", {"profile": "qmt10w", "today": "2026-10-09"})
    assert r.status_code == 200 and r.json()["problems"] == []
    t = r.json()["ticket"]
    assert t["buy"][0]["code"] == "600000" and t["trading_enabled"] is False     # 开关默认关
    assert _signed(client, "/api/qmt/ticket", {"profile": "qmt10w", "today": "2026-10-10"}).json()["problems"]
    assert _signed(client, "/api/qmt/ticket", {"profile": "steady5w", "today": "2026-10-09"}).status_code == 400


def test_exec_report_stored_and_summarised(client, tmp_path):
    rep = {"today": "2026-10-09", "mode": "shadow", "status": "done",
           "orders": [{"code": "600000", "side": "buy", "volume": 300, "price": 10.2, "phase": "auction"}],
           "skips": [{"code": "600111", "reason": "涨停买不进"}]}
    r = _signed(client, "/api/qmt/exec_report", rep)
    assert r.status_code == 200 and (tmp_path / "qmt" / "exec_20261009_shadow.json").exists()
    key, text = client.alerts[-1]
    assert key == "2026-10-09:exec:shadow" and "买 600000 ×300" in text and "涨停买不进" in text
    assert _signed(client, "/api/qmt/exec_report", {"today": "2026-10-09", "mode": "yolo"}).status_code == 400


def test_switch_only_admin_or_owner(client, tmp_path):
    import web_server as ws

    def login(code):
        c = ws.app
        from fastapi.testclient import TestClient
        tc = TestClient(c)
        tc.cookies.set(ws.VIEW_COOKIE, _view_token(ws, code))
        return tc

    assert login("px").post("/api/profile/qmt_switch", json={"enabled": True}).status_code == 403
    assert login("213213").post("/api/profile/qmt_switch", json={"enabled": True}).status_code == 403
    r = login("llx").post("/api/profile/qmt_switch", json={"enabled": True})
    assert r.status_code == 200 and r.json()["switch"]["enabled"] is True and r.json()["switch"]["by"] == "llx"
    assert login("611611").post("/api/profile/qmt_switch", json={"enabled": "yes"}).status_code == 400
    assert login("611611").post("/api/profile/qmt_switch", json={"enabled": False}).json()["switch"]["enabled"] is False


def _view_token(ws, code):
    exp = int(time.time()) + 3600
    return f"{exp}.c.{ws._code_id(code)}.{ws._view_sign_code(exp, code)}"
