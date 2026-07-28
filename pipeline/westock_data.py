"""
westock_data.py — westock-data 腾讯自选股数据 Python 客户端

数据源: npx westock-data-clawhub@1.0.4 (腾讯自选股)
支持: A股(sh/sz/bj), 港股(hk), 美股(us)
用途: 获取美股链主历史K线数据，补充本地A股K线覆盖不到的品种

用法:
  from pipeline.westock_data import WestockClient
  client = WestockClient()
  df = client.kline("usNVDA", limit=500)
"""
import json
import subprocess
import re
import io
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np

from pipeline.logger import get_logger
log = get_logger("westock")


WESTOCK_PKG = "westock-data-clawhub@1.0.4"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "westock_cache")

# Windows 下 npx 路径（查找 .cmd 而非 .ps1）
_NPX_CMD = None
for _candidate in [
    r"C:\Users\admin\AppData\Local\hermes\node\npx.cmd",
    "npx.cmd",
    "npx",
]:
    try:
        import subprocess
        r = subprocess.run([_candidate, "--version"], capture_output=True, timeout=5, shell=True)
        if r.returncode == 0:
            _NPX_CMD = _candidate
            break
    except (OSError, subprocess.SubprocessError):
        log.debug("npx candidate %s unavailable", _candidate)
        continue
if _NPX_CMD is None:
    _NPX_CMD = "npx.cmd"  # fallback


class WestockClient:
    """westock-data CLI 的 Python 封装"""

    def __init__(self, cache: bool = True):
        self.pkg = WESTOCK_PKG
        self.cache = cache
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)

    def _run(self, subcmd: str, *args) -> dict[str, Any]:
        """执行 npx westock-data-clawhub 命令，返回结构化结果"""
        cmd = [_NPX_CMD, "-y", self.pkg, subcmd] + list(args)
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace"
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "超时"}

        stdout = r.stdout.strip()
        stderr_text = r.stderr.strip()[:300] if r.stderr else ""

        if r.returncode != 0:
            return {"success": False, "error": f"exit {r.returncode}: {stderr_text}"}

        # 尝试解析 JSON
        start = stdout.find("{")
        if start >= 0:
            depth, end = 0, start
            for i, ch in enumerate(stdout[start:], start):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0: end = i + 1; break
            if end > start:
                try:
                    return json.loads(stdout[start:end])
                except json.JSONDecodeError:
                    pass

        # 非 JSON 输出（Markdown 表格等）
        return {"success": True, "data": stdout}

    def _cache_path(self, symbol: str, limit: int) -> str:
        return os.path.join(CACHE_DIR, f"{symbol}_{limit}.parquet")

    def kline(self, symbol: str, limit: int = 100, use_cache: bool = True) -> pd.DataFrame | None:
        """
        获取K线数据，返回 DataFrame (columns: date, open, close, high, low, volume, amount).
        支持 A股(sh600000), 港股(hk00700), 美股(usNVDA)
        """
        # 检查缓存
        cache_path = self._cache_path(symbol, limit)
        if use_cache and self.cache and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except (ValueError, OSError, pd.errors.EmptyDataError) as _e:
                log.debug("read cache fail: %s", _e)
                pass

        result = self._run("kline", symbol, "--period", "day", "--limit", str(limit))
        if not result.get("success"):
            print(f"  [westock] {symbol}: {result.get('error', '未知错误')}")
            return None

        data_str = result.get("data", "")
        if not data_str or data_str == "数据为空":
            return None

        # 解析 Markdown 表格
        df = self._parse_markdown_table(data_str)
        if df is None or len(df) == 0:
            return None

        # 标准化列名
        col_map = {"last": "close", "open": "open", "high": "high",
                    "low": "low", "volume": "volume", "amount": "amount", "date": "date"}
        df.rename(columns=col_map, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 缓存
        if self.cache:
            try:
                df.to_parquet(cache_path)
            except OSError as _e:
                log.debug("write cache fail: %s", _e)
                pass

        return df

    def _parse_markdown_table(self, text: str) -> pd.DataFrame | None:
        """解析 Markdown 表格为 DataFrame"""
        lines = text.strip().splitlines()
        if len(lines) < 3:
            return None

        # 找表头行
        header_idx = None
        for i, line in enumerate(lines):
            if "| ---" in line and i > 0:
                header_idx = i - 1
                break

        if header_idx is None:
            return None

        # 解析表头
        header = [h.strip() for h in lines[header_idx].split("|") if h.strip()]

        # 解析数据行
        rows = []
        for line in lines[header_idx + 2:]:
            if "|" not in line:
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) == len(header):
                rows.append(cols)

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=header)

        # 数值列转换
        numeric_cols = ["open", "last", "high", "low", "volume", "amount"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def search(self, keyword: str) -> list[dict]:
        """搜索股票"""
        result = self._run("search", keyword)
        # 返回可能是 markdown，做基础解析
        return [{"keyword": keyword, "result": result.get("data", "")}]

    def quote(self, symbol: str) -> dict[str, Any]:
        """实时行情"""
        result = self._run("quote", symbol)
        data_str = result.get("data", "")
        if not data_str:
            return {}
        return self._parse_markdown_table(data_str).to_dict("records")[0] \
            if self._parse_markdown_table(data_str) is not None and len(self._parse_markdown_table(data_str)) > 0 else {}


# ─── 快速测试 ───

if __name__ == "__main__":
    import sys

    client = WestockClient(cache=True)

    symbol = sys.argv[1] if len(sys.argv) > 1 else "usNVDA"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print(f"\n获取 {symbol} K线数据 (limit={limit})...")
    df = client.kline(symbol, limit=limit, use_cache=False)

    if df is not None:
        print(f"  OK: {len(df)} 行, {df['date'].min().date()} ~ {df['date'].max().date()}")
        print(df.tail(3).to_string(index=False))
    else:
        print("  FAIL: 无数据")
