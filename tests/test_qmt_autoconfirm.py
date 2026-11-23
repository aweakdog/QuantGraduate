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


def _offline_exec_day(tmp_path, lots=(), next_positions=(), next_online=True):
    """执行日 QMT 离线(只有 offline 快照), 次日早上有一份快照。"""
    _setup(tmp_path, lots=lots)
    for p in (tmp_path / "qmt").glob("snap_*"):
        p.unlink()
    store_snapshot({"probe_at": f"{EXEC}T15:10:04", "kind": "close", "account_online": False,
                    "error": "连接 miniQMT 失败(rc=-1)"}, tmp_path)
    store_snapshot({"probe_at": "2026-10-12T09:35:01", "kind": "morning", "account_online": next_online,
                    "asset": {"cash": 100000.0}, "orders": [], "trades": [],
                    "positions": [{"stock_code": c, "volume": v} for c, v in next_positions]}, tmp_path)


def test_offline_exec_day_confirms_empty_when_next_snapshot_matches_ledger(tmp_path):
    _offline_exec_day(tmp_path, lots=[("600000", 300)], next_positions=[("600000.SH", 300)])
    calls, sent = [], []
    rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: sent.append(t) or True)
    assert rec["status"] == "confirmed" and rec["fallback"] is True and rec["fills"] == []
    assert rec["snapshot_at"] == "2026-10-12T09:35:01" and "QMT 离线" in sent[0] and "无成交" in sent[0]
    assert json.loads((tmp_path / rec["confirm_file"]).read_text()) == []


def test_offline_exec_day_blocks_when_positions_changed_or_no_later_snapshot(tmp_path):
    _offline_exec_day(tmp_path, lots=[("600000", 300)], next_positions=[("600000.SH", 300), ("601111.SH", 100)])
    calls = []
    rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: True)
    assert rec["status"] == "blocked" and calls == [] and "推不出执行日成交" in rec["problems"][0]
    (tmp_path / "b").mkdir()
    _offline_exec_day(tmp_path / "b", next_online=False)
    rec = qa.run_one(PID, live=tmp_path / "b", runner=_runner_that_settles(tmp_path / "b", calls), send=lambda t: True)
    assert rec["status"] == "blocked" and calls == [] and "之后也还没有在线快照" in rec["problems"][0]


def test_dry_run_writes_nothing(tmp_path):
    _setup(tmp_path)
    rec = qa.run_one(PID, live=tmp_path, dry_run=True, runner=lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert rec["status"] == "dry_run" and not list(tmp_path.glob("confirm_*"))


# ── 补账 (11-20): 全自动线不停下等人 ──────────────────────────────
def _stuck_week(tmp_path, latest_positions=(), lots=(), avg_price=None, trades=()):
    """11-16~11-20 的原型: 执行日收盘读屏离线, 次日读屏看到人工买的股票, 之后某天的最新读屏是 latest_positions。"""
    _offline_exec_day(tmp_path, lots=lots, next_positions=[("600673.SH", 800)])
    store_snapshot({"probe_at": "2026-10-14T09:37:00", "kind": "morning", "account_online": True,
                    "asset": {"cash": 457000.0, "total_asset": 457000.0}, "orders": [],
                    "trades": [{"stock_code": c, "order_type": ot, "traded_volume": v, "traded_price": p} for c, ot, v, p in trades],
                    "positions": [{"stock_code": c, "volume": v, "avg_price": avg_price} for c, v in latest_positions]}, tmp_path)


def test_same_night_still_waits_but_next_day_catches_up_to_broker(tmp_path):
    _stuck_week(tmp_path)                                    # 人工买了又卖了, 最新读屏是空仓 = 账本
    calls, sent = [], []
    run = _runner_that_settles(tmp_path, calls)
    rec = qa.run_one(PID, live=tmp_path, runner=run, send=lambda t: sent.append(t) or True, today=EXEC)
    assert rec["status"] == "blocked" and calls == [] and "自动补账" in sent[0]          # 当晚: 不凭单次读屏改账
    rec = qa.run_one(PID, live=tmp_path, runner=run, send=lambda t: sent.append(t) or True, today="2026-10-14")
    assert rec["status"] == "confirmed" and rec["mode"] == "catchup" and rec["fills"] == []
    assert rec["snapshot_at"] == "2026-10-14T09:37:00" and "按无成交补账" in rec["notes"][0] and "已补账" in sent[-1]
    assert json.loads((tmp_path / rec["confirm_file"]).read_text()) == []


def test_catchup_adopts_and_sells_to_match_broker(tmp_path):
    _stuck_week(tmp_path, lots=[("000001", 200)], latest_positions=[("600673.SH", 800)], avg_price=21.33,
                trades=[("000001.SZ", 24, 200, 12.5)])
    calls = []
    rec = qa.run_one(PID, live=tmp_path, runner=_runner_that_settles(tmp_path, calls), send=lambda t: True, today="2026-10-14")
    assert rec["status"] == "confirmed" and rec["fills"] == [
        {"code": "000001", "action": "sell", "shares": 200, "price": 12.5},              # 最新读屏当天的真实成交价
        {"code": "600673", "action": "buy", "shares": 800, "price": 21.33}]              # 新建仓: 券商成本价
    assert any("计划外, 并入策略线" in n for n in rec["notes"])


def test_catchup_refuses_what_it_cannot_explain(tmp_path):
    from qmt_sync import build_catchup_confirm
    snap = {"probe_at": "2026-10-14T09:37:00", "trades": [],
            "positions": [{"stock_code": "600673.SH", "volume": 850, "avg_price": 21.33}]}
    st = {"cash": 100000.0, "lots": []}
    assert "不是整百" in build_catchup_confirm(snap, st, {}, EXEC)[1][0]
    snap["positions"][0]["volume"] = 8000                                               # 17 万 > 线内 10 万
    assert "负数" in build_catchup_confirm(snap, st, {}, EXEC)[1][0]
    snap["positions"][0].update(volume=800, avg_price=None)                             # 没成本价、没成交、没 K 线
    assert "估不出成交价" in build_catchup_confirm(snap, st, {}, EXEC, kline_dir="/nonexistent")[1][0]
    assert build_catchup_confirm(None, st, {}, EXEC)[1]


def test_bank_transfer_block_is_lifted_once_a_later_snapshot_confirms_total(tmp_path, monkeypatch):
    import qmt_sync
    _setup(tmp_path, trades=[("600000.SH", 23, 400, 10.3)], positions=[("600000.SH", 400)], plan_buys=["600000"])
    real = qmt_sync.build_autoconfirm
    monkeypatch.setattr(qmt_sync, "build_autoconfirm", lambda *a: (lambda f, p, n: (
        f, p + ["总资产 503,000.00 → 457,000.00 变化超出合理范围 ±1,700 (可能读错位数, 或当天有银证转账)"], n))(*real(*a)))
    snap = json.loads(next((tmp_path / "qmt").glob("snap_*_close.json")).read_text())
    snap["asset"].update(total_asset=457000.0, market_value=4120.0)
    store_snapshot(snap, tmp_path)
    calls = []
    run = _runner_that_settles(tmp_path, calls)
    assert qa.run_one(PID, live=tmp_path, runner=run, send=lambda t: True, today="2026-10-12")["status"] == "blocked"   # 还没有后续读屏
    store_snapshot({"probe_at": "2026-10-12T09:37:00", "kind": "morning", "account_online": True, "orders": [], "trades": [],
                    "asset": {"cash": 452900.0, "total_asset": 457050.0},
                    "positions": [{"stock_code": "600000.SH", "volume": 400}]}, tmp_path)
    rec = qa.run_one(PID, live=tmp_path, runner=run, send=lambda t: True, today="2026-10-12")
    assert rec["status"] == "confirmed" and rec["mode"] == "close" and rec["fills"][0]["price"] == 10.3
    assert "银证转账" in rec["notes"][-1]
