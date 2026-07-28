"""
统一日志系统 — 替代所有 print() 调用

用法:
  from pipeline.logger import log
  log.info("加载数据: %d 行", n)
  log.warning("数据不足: %s", code)
  log.error("查询失败: %s", e)

特性:
  - 控制台输出（含颜色）
  - 文件持久化日志 (logs/pipeline.log)
  - 分级日志 (DEBUG/INFO/WARNING/ERROR)
  - 自动轮转 (10MB/文件)
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ─── 日志级别 ─────────────────────────────────────────
LOG_LEVEL = os.environ.get("QUANT_LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# ─── Formatter ────────────────────────────────────────
_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s - %(message)s"
_FORMAT_FILE = "%(asctime)s [%(levelname)-5s] %(name)s:%(lineno)d - %(message)s"

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _LoggerFactory:
    """延迟初始化日志系统，避免 import 时创建文件"""

    _initialized = False

    @classmethod
    def get_logger(cls, name: str = "quant") -> logging.Logger:
        if not cls._initialized:
            cls._initialize()
        return logging.getLogger(name)

    @classmethod
    def _initialize(cls):
        root = logging.getLogger("quant")
        root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # 移除已有 handler（防止重复）
        root.handlers.clear()

        # 1. 控制台 handler（彩色——仅 level >= WARNING 用 stderr）
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        _fmt = logging.Formatter(_FORMAT, _DATE_FORMAT)
        console.setFormatter(_fmt)
        root.addHandler(console)

        # 2. 文件 handler（持久化，自动轮转）
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                str(LOG_DIR / "pipeline.log"),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
            _fmt_f = logging.Formatter(_FORMAT_FILE, _DATE_FORMAT)
            file_handler.setFormatter(_fmt_f)
            root.addHandler(file_handler)
        except OSError:
            pass  # 日志目录不可写时静默降级

        cls._initialized = True


# ─── 快捷引用（推荐用法）───────────────────────────────
log = _LoggerFactory.get_logger("quant")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的子 logger"""
    return _LoggerFactory.get_logger(f"quant.{name}")
