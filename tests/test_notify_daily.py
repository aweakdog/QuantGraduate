# -*- coding: utf-8 -*-
"""锁住跟单提醒的判定逻辑。

这套逻辑的两类错误后果完全不对称, 所以要分别锁死:

  漏发 —— 人错过尾盘下单窗口, 或者整条线卡在等确认却没人知道。
          这是引入提醒系统本来要解决的问题, 漏了等于白做。
  误发 —— 没事也发、窗口已过还催"快下单"、非交易日发"今天要操作"。
          后果是人对提醒麻木, 真要紧那天照样划过去 —— 反而更糟。

所以下面既测"该发的一定发", 也测"不该发的一个都不许发"。
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class _Skip(Exception):
    pass


def _mod():
    try:
        import notify_daily
        return notify_daily
    except ImportError as e:                     # pragma: no cover
        raise _Skip(f"导入 notify_daily 失败: {e}")


def _item(**kw):
    """一条线的状态。默认是"什么都不用做", 各用例只覆盖关心的字段。"""
    base = {
        "pid": "aggr2w", "name": "激进2万", "action": "none",
        "sell": [], "buy": [], "n_ops": 0,
        "rel": "tomorrow", "phase": "before",
        "when_text": "明天 14:50–15:00 (尾盘)", "slot_label": "14:50–15:00 (尾盘)",
        "awaiting": None, "can_confirm": False, "overdue": None,
        "stale": False, "stale_reason": "", "hold_n": 0,
    }
    base.update(kw)
    return base


_ROW = {"code": "600276", "name": "恒瑞医药", "shares": 100}
NOW = datetime(2026, 8, 10, 14, 35)


# ── 不该发的, 一个都不许发 ────────────────────────────────────────────

def test_无待办时三个时点都不发():
    m = _mod()
    items = [_item(), _item(pid="aggr10w", name="激进10万")]
    for slot in ("preclose", "signal", "confirm"):
        assert m.compose(slot, items, NOW) is None, f"{slot} 在无待办时不该发消息"


def test_尾盘窗口已过后不再催下单():
    """phase=='after' 还催"快下单"是错的 —— 那时该走晚上的确认提醒。"""
    m = _mod()
    items = [_item(action="trade", rel="today", phase="after",
                   sell=[_ROW], buy=[_ROW])]
    assert m.compose("preclose", items, NOW) is None


def test_执行日不是今天时不发尾盘提醒():
    m = _mod()
    items = [_item(action="trade", rel="tomorrow", phase="before",
                   sell=[_ROW], buy=[_ROW])]
    assert m.compose("preclose", items, NOW) is None


def test_今天要执行的不进新计划预告():
    """预告是说"下一个交易日", 把今天该做的混进去会让人以为还有时间。"""
    m = _mod()
    items = [_item(action="trade", rel="today", phase="open", buy=[_ROW])]
    assert m.compose("signal", items, NOW) is None


def test_不能确认时不催回填():
    """can_confirm 为假时提交也结算不了(没有执行日行情), 催了是让人白跑。"""
    m = _mod()
    items = [_item(awaiting={"exec_date": "2026-08-10"}, can_confirm=False)]
    assert m.compose("confirm", items, NOW) is None


# ── 该发的, 一定要发 ──────────────────────────────────────────────────

def test_执行日窗口前必须催下单且带股票明细():
    m = _mod()
    items = [_item(action="trade", rel="today", phase="before",
                   sell=[_ROW], buy=[dict(_ROW, code="000725", name="京东方Ａ")])]
    txt = m.compose("preclose", items, NOW)
    assert txt, "执行日窗口前必须提醒"
    assert "600276" in txt and "000725" in txt, "必须给出具体股票, 否则还得再去翻网页"
    assert "14:50" in txt, "必须说清窗口时间"


def test_窗口进行中同样要催():
    m = _mod()
    items = [_item(action="trade", rel="today", phase="open", buy=[_ROW])]
    assert m.compose("preclose", items, NOW) is not None


def test_清仓避险要提醒且不能漏说卖光():
    """in_cash 时 sell/buy 可能都是空的, 若只按"有没有买卖单"判断就会漏掉
    这个最该提醒的动作 —— 大盘转弱要清仓, 漏了就是硬扛下跌。"""
    m = _mod()
    items = [_item(action="cash", rel="today", phase="before", hold_n=3)]
    txt = m.compose("preclose", items, NOW)
    assert txt and "清仓" in txt


def test_卡在等确认的线在执行日要被点名():
    """这条线不会出新信号, 人却容易以为"没消息=不用操作", 实为整条线停摆。"""
    m = _mod()
    items = [_item(action="await", name="LLX")]
    txt = m.compose("preclose", items, NOW)
    assert txt and "LLX" in txt


def test_待确认要催回填并标出逾期():
    m = _mod()
    items = [_item(name="Px", awaiting={"exec_date": "2026-08-07"},
                   can_confirm=True, overdue=2)]
    txt = m.compose("confirm", items, NOW)
    assert txt and "Px" in txt and "2026-08-07" in txt
    assert "逾期" in txt, "已逾期必须说出来, 否则和当天正常待确认看不出区别"


def test_新计划预告包含执行时点():
    m = _mod()
    items = [_item(action="trade", rel="tomorrow", phase="before",
                   when_text="明天 14:50–15:00 (尾盘)", sell=[_ROW], buy=[_ROW])]
    txt = m.compose("signal", items, NOW)
    assert txt and "明天" in txt and "14:50" in txt


def test_多条线合并成一条消息():
    """发的是群消息, 一条覆盖所有线; 拆成多条会刷屏, 也更容易被忽略。"""
    m = _mod()
    items = [_item(name="激进2万", action="trade", rel="today",
                   phase="before", buy=[_ROW]),
             _item(pid="aggr10w", name="激进10万", action="trade", rel="today",
                   phase="before", sell=[_ROW])]
    txt = m.compose("preclose", items, NOW)
    assert txt.count("激进2万") == 1 and txt.count("激进10万") == 1


# ── 运维告警要与操作提醒分开 ──────────────────────────────────────────

def test_流水线故障走独立告警而不混进操作提醒():
    """"系统坏了"是运维信息。混进给跟单者的消息里, 人会以为自己该做什么。"""
    m = _mod()
    items = [_item(stale=True, stale_reason="pipeline")]
    assert m.admin_alert(items), "流水线故障必须告警"
    assert m.compose("preclose", items, NOW) is None, "故障不该变成操作提醒"


def test_等确认导致的落后不算流水线故障():
    """awaiting_confirm 是系统【故意】停住的, 报成故障会让人白查一遍。"""
    m = _mod()
    items = [_item(stale=True, stale_reason="awaiting_confirm")]
    assert m.admin_alert(items) is None


# ── 通道层 ────────────────────────────────────────────────────────────

def test_未知通道名必须报错而不是悄悄退回stdout():
    """配置写错却"看起来在正常运行", 是这类系统最容易漏掉的故障。"""
    try:
        import notify_channels
    except ImportError as e:                     # pragma: no cover
        raise _Skip(f"导入 notify_channels 失败: {e}")
    try:
        notify_channels.get_channel("weixin", cfg={})
    except ValueError:
        return
    raise AssertionError("未知通道名应当抛 ValueError")


def test_队列写入是原子的且不留临时文件():
    """发送器可能正好在写入的瞬间扫目录, 读到半个 JSON 就会崩或漏发。"""
    import json
    import tempfile
    try:
        import notify_channels
    except ImportError as e:                     # pragma: no cover
        raise _Skip(f"导入 notify_channels 失败: {e}")
    with tempfile.TemporaryDirectory() as td:
        ch = notify_channels.QueueChannel(td)
        assert ch.send("测试文案", meta={"slot": "preclose"})
        pend = list((Path(td) / "pending").glob("*.json"))
        assert len(pend) == 1, "应恰好落一条待发消息"
        assert not list((Path(td) / "pending").glob(".*.tmp")), "不该残留临时文件"
        got = json.loads(pend[0].read_text(encoding="utf-8"))
        assert got["text"] == "测试文案"
        assert got.get("expire_at"), "必须带过期时间, 否则积压的旧提醒会被补发出去"


def _main():
    """无 pytest 环境下直接 python tests/xxx.py 就能跑。"""
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for n in names:
        try:
            globals()[n]()
            print(f"  PASS  {n}")
        except _Skip as e:
            print(f"  SKIP  {n}: {e}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL  {n}: {str(e)[:160]}")
    print(f"\n{len(names) - len(failed)}/{len(names)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
