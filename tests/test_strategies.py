"""
策略模块 — 语法与结构检验

策略模块有模块级数据加载副作用，无法直接导入测试。
改用 ast.parse 验证语法正确性和路径引用的完整性。
"""
import ast
import pytest
from pathlib import Path


def _get_strategy_modules():
    strategies_dir = Path(__file__).resolve().parent.parent / "strategies"
    return sorted(
        f for f in strategies_dir.glob("*.py")
        if f.stem != "__init__" and not f.stem.startswith("test_")
    )


STRATEGY_FILES = _get_strategy_modules()


class TestStrategySyntax:
    """验证每个策略文件的语法正确性"""

    @pytest.mark.parametrize("filepath", STRATEGY_FILES, ids=lambda p: p.stem)
    def test_strategy_syntax(self, filepath):
        """验证语法树可正常解析"""
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        assert tree.body, f"{filepath.name}: 无有效语句"

    @pytest.mark.parametrize("filepath", STRATEGY_FILES, ids=lambda p: p.stem)
    def test_no_hardcoded_paths(self, filepath):
        """验证没有遗留的硬编码 D:\\myAI\\ 路径"""
        source = filepath.read_text(encoding="utf-8")
        lines = source.splitlines()
        bad = [i + 1 for i, line in enumerate(lines) if "D:\\myAI" in line or "D:/myAI" in line]
        assert len(bad) == 0, f"{filepath.name}: {bad}行存在硬编码路径"

    def test_strategy_count(self):
        """验证策略模块数量 >= 10"""
        assert len(STRATEGY_FILES) >= 10, f"仅发现 {len(STRATEGY_FILES)} 个策略文件"
