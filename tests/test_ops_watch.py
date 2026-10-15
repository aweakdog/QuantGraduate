"""数据日更告警: 硬失败/软失败(token·网络·其他)/信号失败/逐笔陈旧 当场报; 截止检查兜住没跑或卡死; 不泄密钥。"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ops_watch as ow  # noqa: E402

TUSHARE_PERM = "抱歉，您没有接口(moneyflow)访问权限，权限的具体详情访问：https://tushare.pro/document/1?doc_id=108。"


def status(**kw):
    s = {"started_at": "2026-10-15T17:30:38", "finished_at": "2026-10-15T18:35:00", "ok": True,
         "skipped_reason": None, "kline_max_date": "2026-10-15", "error": None,
         "stages": {"kline_full_market": {"ok": True}, "build_features": {"ok": True}},
         "signals": {"qmt10w": {"ok": True}, "steady5w": {"ok": True}}, "new_train_info": {"warnings": []}}
    s.update(kw)
    return s


def test_classify_and_mask():
    assert ow.classify(TUSHARE_PERM) == "token"
    assert ow.classify("您的token不对，请确认。") == "token"
    assert ow.classify("HTTP Error 401: Unauthorized") == "token"
    assert ow.classify("Command '[...pull_macro]' timed out after 900 seconds") == "network"
    assert ow.classify("新训练集校验不通过: 行数缩水") == "other"
    leak = "Authorization: eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjt"
    assert "eyJhbGci" not in ow.mask(leak) and "***" in ow.mask(leak)


def test_clean_run_and_skip_run_send_nothing():
    assert ow.evaluate_run(ow.summarize_run(status())) == []
    assert ow.evaluate_run(ow.summarize_run(status(skipped_reason="K线最新日未超过训练集", signals={}))) == []


def test_hard_failure_reports_stage_and_retry():
    s = status(ok=False, error="新训练集校验不通过:\n  - 行数缩水", signals={},
               stages={"build_features": {"ok": True}, "validate": {"ok": False, "problems": ["行数缩水"]}})
    (kind, text, email), = ow.evaluate_run(ow.summarize_run(s))
    assert kind == "fail" and "训练集校验" in text and "19:30 / 21:30 会自动重试" in text
    assert "训练集校验" in email and "行数缩水" not in email          # 邮件不带原始报错


def test_soft_failures_grouped_by_cause_token_first_class():
    s = status(stages={"pull_moneyflow": {"ok": False, "error": TUSHARE_PERM},
                       "pull_margin_detail": {"ok": False, "error": "您的token不对，请确认。"},
                       "pull_macro": {"ok": False, "error": "timed out after 900 seconds"},
                       "build_features": {"ok": True}})
    alerts = {k: (t, e) for k, t, e in ow.evaluate_run(ow.summarize_run(s))}
    assert set(alerts) == {"soft_token", "soft_network"}
    t, e = alerts["soft_token"]
    assert "token 失效或接口权限不足" in t and "资金流(Tushare)" in t and "两融(Tushare)" in e and "tushare.pro" not in e
    assert "宏观数据" in alerts["soft_network"][1]


def test_signal_failures_and_tick_staleness():
    s = status(signals={"qmt10w": {"ok": True}, "fyf100w": {"ok": False, "error": "指纹不匹配"}},
               stages={"build_tick1": {"ok": True, "log": "新鲜度: 基矩阵末日 2026-10-15, 逐笔面板末日 2026-10-13, 陈旧 2 个交易日; ..."}})
    alerts = {k: (t, e) for k, t, e in ow.evaluate_run(ow.summarize_run(s))}
    assert set(alerts) == {"signal", "tick_stale"}
    assert "fyf100w" in alerts["signal"][1] and "断供 2 个交易日" in alerts["tick_stale"][1]


def test_report_run_appends_history_and_alerts_once(tmp_path, monkeypatch):
    import qmt_sync
    got = []
    monkeypatch.setattr(qmt_sync, "_default_senders",
                        lambda: [("main", lambda t: got.append(("main", t)) or True),
                                 ("email", lambda t: got.append(("email", t)) or True)])
    s = status(stages={"pull_moneyflow": {"ok": False, "error": TUSHARE_PERM}})
    rec, sent = ow.report_run(s, live_dir=tmp_path)
    assert sent == ["soft_token"] and len((tmp_path / ow.RUNS).read_text().splitlines()) == 1
    assert got[0][0] == "main" and "tushare.pro" in got[0][1]                 # 企业微信给详情
    assert got[1][0] == "email" and "tushare.pro" not in got[1][1] and "eez041.ece.ust.hk:8737" in got[1][1]
    ow.report_run(s, live_dir=tmp_path)                                        # 同日同类不重发
    assert len(got) == 2 and len((tmp_path / ow.RUNS).read_text().splitlines()) == 2


def test_service_check_alerts_only_for_down_units(tmp_path):
    sent = []
    send = lambda t: sent.append(t) or True  # noqa: E731
    at = datetime.fromisoformat("2026-10-15T21:15:00")
    assert ow.service_check(at, tmp_path, active_of=lambda u: True, send=send) == [] and sent == []
    down = ow.service_check(at, tmp_path, active_of=lambda u: u != "quant-web.service", send=send)
    assert down == ["quant-web.service"] and "quant-web.service" in sent[0]


def test_deadline_check(tmp_path):
    sent = []
    send = lambda t: sent.append(t) or True  # noqa: E731
    at = datetime.fromisoformat("2026-10-15T21:15:00")
    assert ow.deadline_check(at, tmp_path, is_trading=False, send=send) is None          # 非交易日不管
    assert ow.deadline_check(at, tmp_path, is_trading=True, send=send) == "deadline"     # 没有任何运行记录
    assert "还没有任何一次日更运行记录" in sent[-1]
    with (tmp_path / ow.RUNS).open("a") as fh:
        fh.write(json.dumps(ow.summarize_run(status(skipped_reason="K线最新日 2026-10-14 未超过训练集", ok=True,
                                                  kline_max_date="2026-10-14", signals={}))) + "\n")
    late = datetime.fromisoformat("2026-10-15T23:45:00")
    assert ow.deadline_check(late, tmp_path, is_trading=True, send=send) == "deadline_final"
    assert "K 线没有更新到今天" in sent[-1] and "彻底" in sent[-1]
    with (tmp_path / ow.RUNS).open("a") as fh:
        fh.write(json.dumps(ow.summarize_run(status())) + "\n")
    assert ow.deadline_check(late, tmp_path, is_trading=True, send=send) is None         # 有一次成功推进到当天
