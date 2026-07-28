"""
测试日志系统 — logger 初始化与基本功能
"""
import logging
import pytest
from pipeline.logger import get_logger, log


class TestLogger:

    def test_get_logger(self):
        l = get_logger("test")
        assert l is not None
        assert "quant.test" in l.name

    def test_multiple_loggers(self):
        a = get_logger("A")
        b = get_logger("B")
        assert a.name == "quant.A"
        assert b.name == "quant.B"

    def test_log_levels(self, caplog):
        l = get_logger("levels_test")
        # caplog 需要设置到根 quant logger
        caplog.set_level(logging.DEBUG, logger="quant")

        l.debug("debug msg")
        l.info("info msg")
        l.warning("warning msg")

        assert "debug msg" in caplog.text
        assert "info msg" in caplog.text
        assert "warning msg" in caplog.text

    def test_error_logging(self, caplog):
        l = get_logger("error_test")
        caplog.set_level(logging.ERROR, logger="quant")

        l.error("error msg")
        assert "error msg" in caplog.text

    def test_child_logger(self):
        l = get_logger("parent.child")
        assert l.name == "quant.parent.child"

    def test_root_log(self):
        """验证快捷引用 log 可正常调用"""
        assert log.name == "quant"
        # 不报错即可
        log.debug("root debug")
        log.info("root info")
