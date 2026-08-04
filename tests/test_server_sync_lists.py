"""主节点核对清单必须覆盖全部工作节点文件

2026-08-05 踩过的坑 (旧版本反向传播):
  pull_fundflow_shard.py 只出现在 WORKER_FILES 里, 没进主节点清单, 于是主节点
  那份从未被核对过、一直是旧的。--push 把新版推给了 eez040/042, 但过夜编排
  脚本随后【从主节点 rsync 分发】, 又把旧版覆盖回两台 —— 三台一起退回旧版,
  而工具还报告"全部一致"。

  表现是子进程秒退 (`unrecognized arguments: --codes-file`), 排查花了不少时间,
  因为同步工具明确说了没问题。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_server_sync import MAIN_FILES, WATCHED, WORKER_FILES  # noqa: E402


def test_main_covers_all_worker_files():
    """否则从主节点分发时会把旧版推回工作节点"""
    missing = [f for f in WORKER_FILES if f not in WATCHED]
    assert not missing, (
        f"这些文件工作节点会用但主节点不核对, 会导致旧版本反向传播: {missing}")


def test_no_duplicates_in_watched():
    assert len(WATCHED) == len(set(WATCHED)), "WATCHED 有重复项"


def test_watched_files_exist_locally():
    """清单里写了却不存在的文件会让核对静默跳过"""
    missing = [f for f in WATCHED if not (ROOT / f).exists()]
    assert not missing, f"清单里的文件本地不存在: {missing}"


def test_orchestrator_is_watched():
    """过夜编排脚本自己也必须同步 —— 它跑在服务器上并调度另两台"""
    assert "scripts/expand_2015_overnight.py" in WATCHED


def test_main_files_is_the_base_not_the_union():
    """MAIN_FILES 是人工维护的基础清单, WATCHED 才是并集。
    若有人把 WATCHED 改回手写清单, 这条会提醒并集关系已被破坏。"""
    assert set(MAIN_FILES).issubset(set(WATCHED))
    assert set(WORKER_FILES).issubset(set(WATCHED))
