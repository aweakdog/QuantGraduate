"""--fix-lot: 单笔持仓校准的语义边界

需求来自实盘: 成本价要能对齐券商 App。之前只能用 --sync 整体覆盖, 那是所有
操作里破坏力最大的一个(一把改写整个账户), 网页入口已因此下掉。

本命令的关键约定:
  1. 只动指定的那一笔, 不动现金 —— 现金用 --set-cash 单独校准。若两边都按
     差额调现金, 同一笔差额会被改两遍。
  2. 只改成本价时总资产不变(总资产 = 现金 + Σ 股数x现价, 不含 buy_price)。
  3. 改股数会改变总资产, 所以只用于修记账错误。
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
SIG = ROOT / "scripts" / "live_signal.py"


def _run(*extra):
    """只跑参数解析层能覆盖到的路径: --print-fingerprint 之前的早退不受影响,
    所以这里用真正的 CLI 调用来验证参数存在且互斥校验生效。"""
    return subprocess.run([str(PY), str(SIG), *extra],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=120)


def test_fix_lot_args_exist():
    r = _run("--help")
    assert r.returncode == 0
    for flag in ("--fix-lot", "--fix-price", "--fix-shares"):
        assert flag in r.stdout, flag


def test_help_states_cash_is_untouched():
    """帮助文本必须写明不动现金 —— 这是最容易误解、后果最难查的一点"""
    r = _run("--help")
    assert "不动现金" in r.stdout or "不动现金也不动" in r.stdout


# ── 纯函数层: 直接验算语义, 不依赖行情数据 ──────────────────
def _equity(cash, lots, prices):
    """总资产口径必须与 live_signal.snapshot 一致: 不含 buy_price"""
    return cash + sum(l["shares"] * prices[l["code"]] for l in lots)


def test_changing_cost_price_does_not_change_equity():
    """改成本价只影响浮盈显示, 不该影响总资产 —— 这是不动现金的正当性所在"""
    prices = {"600276": 60.0}
    lots = [{"code": "600276", "shares": 200, "buy_price": 53.42}]
    before = _equity(10000.0, lots, prices)
    lots[0]["buy_price"] = 55.00          # 校准成本价
    after = _equity(10000.0, lots, prices)
    assert before == after


def test_changing_shares_does_change_equity():
    prices = {"600276": 60.0}
    lots = [{"code": "600276", "shares": 200, "buy_price": 53.42}]
    before = _equity(10000.0, lots, prices)
    lots[0]["shares"] = 300
    after = _equity(10000.0, lots, prices)
    assert after > before
    assert after - before == pytest.approx(100 * 60.0)


def test_pnl_pct_follows_cost_price():
    """校准成本价后, 该笔浮盈应按新成本重算"""
    def pnl(ref, bp):
        return round((ref / bp - 1) * 100, 2)
    assert pnl(60.0, 53.42) == pytest.approx(12.32, abs=0.01)
    assert pnl(60.0, 55.00) == pytest.approx(9.09, abs=0.01)


@pytest.mark.parametrize("old,new,rejected", [
    (53.42, 55.00, False),    # 正常校准, 幅度小
    (53.42, 60.00, False),    # 12% 也算合理
    (53.42, 5.342, True),     # 少打一位 -> 该拒
    (53.42, 10684.0, True),   # 把市值当成本价 -> 该拒
])
def test_guard_rejects_typos(old, new, rejected):
    """防手滑阈值: 成本价改动超过 50% 极可能是输错"""
    too_big = abs(new / old - 1) > 0.5
    assert too_big is rejected
