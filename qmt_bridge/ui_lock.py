"""QMT 界面互斥锁 (2026-10-28): 读屏、逆回购、填单演练、手动测试同一时间只能有一个在操作界面, 免得两个程序抢鼠标。

锁文件 D:\\qmtcode\\ui.lock 里写着谁在用; 超过 20 分钟没删的锁视为上一个程序崩了留下的, 直接清掉。
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOCK = Path(r"D:\qmtcode\ui.lock")
STALE_S = 20 * 60


@contextmanager
def ui_lock(name, wait_s=300, path=None, poll_s=5.0):
    """拿到锁才执行; 等 wait_s 秒还被占着就抛 TimeoutError (带上占用者)。"""
    path = Path(path or LOCK)
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > STALE_S:
                path.unlink(missing_ok=True)
                continue
            if time.time() - t0 >= wait_s:
                raise TimeoutError(f"QMT 界面被占用: {path.read_text(encoding='utf-8', errors='replace')[:200]}") from None
            time.sleep(poll_s)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"name": name, "pid": os.getpid(), "at": datetime.now().isoformat(timespec="seconds")}, f,
                      ensure_ascii=False)
        break
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
