"""migrate_config: 带着持仓换策略参数的安全边界

判据来自 live_signal.settle() 实际读了什么:
  冻结在 pending 里的 -> 安全, 可带持仓原地切
  settle() 实时读的   -> 危险, 需要干净边界(无在途挂单; 有持仓时须先排空)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_every_fingerprint_key_is_classified():
    """漏归类会让第二档参数被当第一档放过 —— 模块导入时就该炸"""
    import migrate_config as m
    from live_config import FINGERPRINT_KEYS
    assert set(FINGERPRINT_KEYS) == m.FREE_KEYS | m.BOUNDED_KEYS
    assert not (m.FREE_KEYS & m.BOUNDED_KEYS), "同一个键不能既在第一档又在第二档"


def test_regime_keys_are_free_and_execution_keys_are_bounded():
    """这组分类是整个工具的正确性根基, 显式钉住"""
    import migrate_config as m
    for k in ("regime_filter", "regime_ma", "regime_breadth", "regime_confirm",
              "reversal_guard"):
        assert k in m.FREE_KEYS, k
    for k in ("hold_days", "tranche_n", "exec_mode", "slippage", "portfolio_mode",
              "train_file", "pit_universe", "label"):
        assert k in m.BOUNDED_KEYS, k


def test_settlement_would_act_logic():
    """迁移窗口的判据。pending 几乎总是存在, 所以不能拿它当拦路条件"""
    import migrate_config as m
    buy_pending = {"signal_date": "2026-08-03", "in_cash": False, "is_rebal": True}
    # 有持仓 -> 结算要判到期, 会动账
    assert m.settlement_would_act({"lots": [{"code": "600000"}]})[0] is True
    # 空仓 + 挂单会买入 -> 会动账
    assert m.settlement_would_act({"lots": [], "pending": buy_pending})[0] is True
    # 空仓 + 挂单是"继续空仓" -> 空操作, 这就是排空后的迁移窗口
    assert m.settlement_would_act(
        {"lots": [], "pending": dict(buy_pending, in_cash=True)})[0] is False
    # 空仓 + 挂单非换仓日 -> 空操作
    assert m.settlement_would_act(
        {"lots": [], "pending": dict(buy_pending, is_rebal=False)})[0] is False
    assert m.settlement_would_act({"lots": [], "pending": None})[0] is False


def test_diff_of_detects_changes_both_ways():
    import migrate_config as m
    d = m.diff_of({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert d == {"b": (2, 3), "c": (None, 4)}
    assert m.diff_of({"a": 1}, {"a": 1}) == {}


# ── 端到端: 用临时 live 目录跑真实脚本 ─────────────────────────
FP_OLD = {
    "train_file": "t.parquet", "pit_universe": "u.parquet", "label": "5d",
    "hold_days": 5, "tranche_n": 3, "portfolio_mode": "periodic",
    "exec_mode": "t1close", "slippage": 0.002, "regime_filter": "off",
    "regime_ma": 20, "regime_breadth": 0.40, "regime_confirm": 2,
    "reversal_guard": 0.0,
}


def _state(lots=0, pending=None, history=1, cash=50000.0):
    return {
        "config": dict(FP_OLD),
        "cash": cash, "initial_capital": 50000.0,
        "lots": [{"id": i + 1, "code": f"60000{i}", "shares": 100,
                  "buy_price": 10.0, "open_signal_date": "2026-07-31",
                  "open_date": "2026-08-03"} for i in range(lots)],
        "next_lot_id": lots + 1,
        "last_rebal_signal_date": "2026-07-31",
        "last_signal_date": "2026-08-03",
        "pending": pending,
        "history": [{"signal_date": "2026-07-24", "fills": []}] * history,
        "calendar": ["2026-07-31", "2026-08-03"],
    }


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """把 migrate_config 的 LIVE 指到临时目录, 并桩掉取指纹的子进程调用"""
    import migrate_config as m
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(m, "LIVE", live)

    def run(state, new_fp, argv):
        p = live / "state_steady5w.json"
        p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(m, "current_fingerprint", lambda pid: new_fp)
        monkeypatch.setattr(sys, "argv", ["migrate_config.py", *argv])
        try:
            m.main()
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if isinstance(e.code, str):
                code = ("REFUSED", e.code)
        return code, json.loads(p.read_text(encoding="utf-8"))

    return run


def test_free_change_with_lots_migrates_in_place(runner, capsys):
    """只改 regime_filter: 带着 3 笔持仓也能直接切, 且什么都不丢"""
    new = dict(FP_OLD, regime_filter="breadth")
    code, st = runner(_state(lots=3), new, ["--profile", "steady5w"])
    assert code == 0
    assert st["config"]["regime_filter"] == "breadth"
    assert len(st["lots"]) == 3, "持仓不能被清掉"
    assert st["cash"] == 50000.0, "现金不能被重置"
    assert len(st["history"]) == 1, "历史不能被清掉"
    # 必须留下分段记录, 否则累计收益会静默混合两个策略
    eps = st["strategy_epochs"]
    assert len(eps) == 2
    assert eps[0]["config"]["regime_filter"] == "off"
    assert eps[1]["config"]["regime_filter"] == "breadth"
    assert eps[1]["changed"]["regime_filter"] == ["off", "breadth"]


BUY_PENDING = {"signal_date": "2026-08-03", "in_cash": False,
               "is_rebal": True, "ranked": ["600000"], "blocked": []}
CASH_PENDING = {"signal_date": "2026-08-03", "in_cash": True,
                "is_rebal": True, "ranked": ["600000"], "blocked": []}


def test_free_key_migrates_even_with_pending_and_lots(runner):
    """稳态下 pending 几乎总存在, 第一档参数必须仍能切 —— 否则永无窗口

    这是 2026-08-04 端到端实测发现的问题: 最初的实现拿"有在途挂单"当拦路
    条件, 结果 5 条线全都无法迁移, 而钉死参数已经上线, 当晚流水线会全线报错。
    """
    new = dict(FP_OLD, regime_filter="breadth")
    code, st = runner(_state(lots=3, pending=BUY_PENDING), new,
                      ["--profile", "steady5w"])
    assert code == 0
    assert st["config"]["regime_filter"] == "breadth"
    assert len(st["lots"]) == 3
    assert len(st["strategy_epochs"]) == 2


def test_refuses_bounded_change_when_settlement_would_act(runner):
    """改 tranche_n 且结算会动账 -> 必须先排空"""
    new = dict(FP_OLD, tranche_n=5)
    code, st = runner(_state(lots=3), new, ["--profile", "steady5w"])
    assert isinstance(code, tuple) and code[0] == "REFUSED"
    assert "--drain" in code[1]
    assert st["config"]["tranche_n"] == 3


def test_refuses_bounded_change_when_pending_will_buy(runner):
    """空仓但挂单会买入: 改 tranche_n 会让买入股数与计划不符"""
    new = dict(FP_OLD, tranche_n=5)
    code, st = runner(_state(lots=0, pending=BUY_PENDING), new,
                      ["--profile", "steady5w"])
    assert isinstance(code, tuple) and code[0] == "REFUSED"
    assert st["config"]["tranche_n"] == 3


def test_bounded_change_migrates_when_settlement_is_noop(runner):
    """空仓 + 挂单是"继续空仓" -> 结算空操作, 可以切。这就是排空后的窗口"""
    new = dict(FP_OLD, tranche_n=5)
    code, st = runner(_state(lots=0, pending=CASH_PENDING), new,
                      ["--profile", "steady5w"])
    assert code == 0
    assert st["config"]["tranche_n"] == 5
    assert len(st["strategy_epochs"]) == 2


def test_drain_sets_flag_without_changing_config(runner):
    """--drain 只置标记, 不能提前改指纹(否则排空那一单就按新参数结算了)"""
    new = dict(FP_OLD, hold_days=10)
    code, st = runner(_state(lots=3), new, ["--profile", "steady5w", "--drain"])
    assert code == 0
    assert st["drain_requested"] is True
    assert st["config"]["hold_days"] == 5, "排空期间指纹必须保持旧值"
    assert "strategy_epochs" not in st or not st.get("strategy_epochs")


def test_dry_run_writes_nothing(runner):
    new = dict(FP_OLD, regime_filter="breadth")
    code, st = runner(_state(lots=3), new,
                      ["--profile", "steady5w", "--dry-run"])
    assert code == 0
    assert st["config"]["regime_filter"] == "off"
    assert "strategy_epochs" not in st


def test_locked_baseline_refused_by_default(runner):
    """基准线存在的意义就是不动 —— 锁定检查发生在读状态之前"""
    new = dict(FP_OLD, regime_filter="breadth")
    code, _ = runner(_state(lots=3), new, ["--profile", "base5w_steady"])
    assert isinstance(code, tuple) and code[0] == "REFUSED"
    assert "基准线" in code[1]
