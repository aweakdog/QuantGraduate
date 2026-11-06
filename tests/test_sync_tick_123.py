"""123 云盘同步: 分卷包 (11-04 起) 要认、齐了要拼回整包; 自己挂的符号链接不能挡住重发后的重新下载。"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sync_tick_123 as s  # noqa: E402


def test_pack_names():
    assert s.is_pack("20261103.7z") and s.is_pack("20261104.7z.001") and s.is_pack("20261104.7z.012")
    assert not s.is_pack("202611月开始的新源 parquet格式 各文件列内容.docx")
    assert not s.is_pack("20261104.7z.part") and not s.is_pack("20261104.7z.vols.json")
    assert s.volumes({"20261103.7z": 9, "20261104.7z.001": 3, "20261104.7z.002": 2}) == {
        "20261104": {"20261104.7z.001": 3, "20261104.7z.002": 2}}


@pytest.fixture
def dest(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "DEST", str(tmp_path / "123"))
    monkeypatch.setattr(s, "BAIDU_TREE", str(tmp_path / "baidu"))
    (tmp_path / "123" / "202611").mkdir(parents=True)
    return tmp_path / "123" / "202611"


def test_assemble_concatenates_in_order_and_is_idempotent(dest):
    (dest / "20261104.7z.002").write_bytes(b"world")
    (dest / "20261104.7z.001").write_bytes(b"hello ")
    vols = {"20261104.7z.001": 6, "20261104.7z.002": 5}
    assert s.assemble("202611", "20261104", vols)
    assert (dest / "20261104.7z").read_bytes() == b"hello world"
    assert s.assemble("202611", "20261104", vols) is None            # 分卷没变就不重拼
    (dest / "20261104.7z.003").write_bytes(b"!")                      # 供应商补传了第 3 卷 -> 重拼
    assert s.assemble("202611", "20261104", vols | {"20261104.7z.003": 1})
    assert (dest / "20261104.7z").read_bytes() == b"hello world!"


def test_assemble_waits_for_all_volumes(dest):
    (dest / "20261105.7z.001").write_bytes(b"abc")
    assert s.assemble("202611", "20261105", {"20261105.7z.001": 3, "20261105.7z.002": 4}) is None   # 第 2 卷没下完
    assert s.assemble("202611", "20261105", {"20261105.7z.001": 3, "20261105.7z.003": 4}) is None   # 编号不连续
    assert not (dest / "20261105.7z").exists()


def test_own_symlink_in_baidu_tree_does_not_count(dest, tmp_path):
    (dest / "20261102.7z").write_bytes(b"old-encrypted")
    link = tmp_path / "baidu" / "2026" / "202611" / "20261102.7z"
    link.parent.mkdir(parents=True)
    os.symlink(dest / "20261102.7z", link)
    assert s.local_ok("202611", "20261102.7z", 13)                    # 大小对: 已有
    assert s.local_ok("202611", "20261102.7z", 4) is None             # 供应商重发换了大小: 要重新下载
    link.unlink()
    link.write_bytes(b"baidu-copy")                                   # 百度渠道的独立副本: 大小不同也算已有
    assert s.local_ok("202611", "20261102.7z", 4) == str(link)
