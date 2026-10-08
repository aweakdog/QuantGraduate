"""qmt_autoconfirm: 对得上才调 live_signal --confirm; 对不上停在待确认并告警(同日只告一次)。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qmt_autoconfirm as qa  # noqa: E402
from qmt_sync import store_snapshot  # noqa: E402

PID = "qmt10w"
EXEC = "2026-10-09"


def _setup(tmp_path, lots=(), awaiting=True, trades=(), positions=(), plan_buys=()):
    st = {"cash": 100000.0, "lots": [{"code": c, "shares": s, "buy_price": 10.0} for c, s in lots],
          "awaiting_confirm": {"signal_date": "2026-10-08", "exec_date": EXEC, "since": "x"} if awaiting else None}
    (tmp_path / f"state_{PID}.json").write_text(json.dumps(st))
    (tmp_path / f"plan_{PID}_2026-10-08.json").write_text(json.dumps({"buy": [{"code": c} for c in plan_buys], "alternates": []}))
    store_snapshot({"probe_at": f"{EXEC}T15:10:00", "kind": "close", "account_online": True,
                    "asset": {"cash": 100000.0}, "orders": [],
                    "positions": [{"stock_code": c, "volume": v} for c, v in positions],
                    "trades": [{"stock_code": c, "order_type": ot, "traded_volume": v, "traded_price": p}
                               for c, ot, v, p in trades]}, tmp_path)


def _runner_that_settles(tmp_path, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        p = tmp_path / f"state_{PID}.json"
        st = json.loads(p.read_text())
        st["awaiting_confirm"] = None
        p.write_text(json.dumps(st))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    return run


def test_nothing_to_do_when_not_awaiting(tmp_path):
    _setup(tmp_path, awaiting=False)
    assert qa.run_one(PID, live=tmp_path)["status"] == "nothing_to_confirm"


def test_empty_day_confirms_with_empty_fills(tmp_path):
    _setup(tmp_path)
    calls, sent = [], []
    rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: sent.append(t) or True)
    assert rec["status"] == "confirmed" and rec["fills"] == []
    cmd = calls[0]
    assert cmd[cmd.index("--confirm") + 1].endswith("_qmt.json")
    assert json.loads((tmp_path / rec["confirm_file"]).read_text()) == []
    assert "--state" in cmd and cmd[cmd.index("--state") + 1] == f"state_{PID}.json"
    assert "无成交" in sent[0]


def test_real_fills_are_passed_through(tmp_path):
    _setup(tmp_path, lots=[("000001", 200)], trades=[("000001.SZ", 24, 200, 12.0), ("600000.SH", 23, 400, 10.3)],
           positions=[("600000.SH", 400)], plan_buys=["600000"])
    calls = []
    rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: True)
    assert rec["status"] == "confirmed"
    assert json.loads((tmp_path / rec["confirm_file"]).read_text()) == [
        {"code": "000001", "action": "sell", "shares": 200, "price": 12.0},
        {"code": "600000", "action": "buy", "shares": 400, "price": 10.3}]


def test_mismatch_blocks_and_alerts_once(tmp_path):
    _setup(tmp_path, positions=[("600519.SH", 100)])        # 线外持仓
    calls, sent = [], []
    for _ in range(2):
        rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: sent.append(t) or True)
        assert rec["status"] == "blocked"
    assert calls == [] and len(sent) == 1 and "没有自动确认" in sent[0]
    assert json.loads((tmp_path / f"state_{PID}.json").read_text())["awaiting_confirm"] is not None


def test_failed_live_signal_is_reported(tmp_path):
    _setup(tmp_path)
    sent = []

    def bad(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: 还没到执行日的行情")
    rec = qa.run_one(PID, live=tmp_path, runner=bad, send=lambda t: sent.append(t) or True)
    assert rec["status"] == "failed" and "rc=1" in sent[0]


def test_dry_run_writes_nothing(tmp_path):
    _setup(tmp_path)
    rec = qa.run_one(PID, live=tmp_path, dry_run=True, runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert rec["status"] == "dry_run" and not list(tmp_path.glob("confirm_*"))
