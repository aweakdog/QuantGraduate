"""每日重启重登: 健康且不强制就不动; 不健康或强制才重启, 等到交易账户状态 0 为止, 超时报失败。"""
from qmt_bridge.ensure_qmt import ensure, healthy

OK, WAITING, DOWN = {"connect": 0, "status": 0}, {"connect": 0, "status": 1}, {"connect": -1, "status": None}


def run(seq, force):
    calls, probes = [], iter(seq)
    res = ensure(force, probe_fn=lambda: next(probes), kill_fn=lambda: calls.append("kill"),
                 launch_fn=lambda: calls.append("launch"), sleep_fn=lambda s: None, wait_s=60, every_s=15)
    return res, calls


def test_healthy_means_connected_and_status_zero():
    assert healthy(OK) and not healthy(WAITING) and not healthy(DOWN) and not healthy({})


def test_healthy_and_not_forced_leaves_qmt_alone():
    (ok, relaunched, _), calls = run([OK], force=False)
    assert ok and not relaunched and calls == []


def test_waiting_login_triggers_restart_until_status_ok():
    (ok, relaunched, last), calls = run([WAITING, DOWN, WAITING, OK], force=False)
    assert ok and relaunched and last == OK and calls == ["kill", "launch"]


def test_force_restarts_even_if_healthy():
    (ok, relaunched, _), calls = run([OK, OK], force=True)
    assert ok and relaunched and calls == ["kill", "launch"]


def test_gives_up_after_wait_and_reports_unhealthy():
    (ok, relaunched, last), calls = run([DOWN] * 10, force=True)    # 权限被拒: 登录一直失败
    assert not ok and relaunched and last == DOWN
