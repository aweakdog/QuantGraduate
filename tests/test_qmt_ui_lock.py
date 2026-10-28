"""界面互斥锁: 同一时间只能一个程序操作 QMT; 崩溃留下的旧锁会被清掉。"""
import os
import time

import pytest

from qmt_bridge.ui_lock import STALE_S, ui_lock


def test_lock_released_after_use(tmp_path):
    p = tmp_path / "ui.lock"
    with ui_lock("a", path=p):
        assert p.exists() and '"name": "a"' in p.read_text(encoding="utf-8")
    assert not p.exists()


def test_second_user_times_out_while_held(tmp_path):
    p = tmp_path / "ui.lock"
    with ui_lock("a", path=p), pytest.raises(TimeoutError, match="被占用"), ui_lock("b", path=p, wait_s=0.2, poll_s=0.05):
        pass
    assert not p.exists()


def test_stale_lock_is_cleared(tmp_path):
    p = tmp_path / "ui.lock"
    p.write_text('{"name": "crashed"}', encoding="utf-8")
    old = time.time() - STALE_S - 10
    os.utime(p, (old, old))
    with ui_lock("b", path=p, wait_s=0):
        assert '"name": "b"' in p.read_text(encoding="utf-8")
