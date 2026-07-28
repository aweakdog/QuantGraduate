"""
thsdk 工具库 — KQ2026 正式账号
统一连接、查询、限频处理
"""
import time
import json
import os
import pandas as pd
from datetime import datetime
from typing import Any, Optional

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("ths_utils")

# thsdk python 解释器路径（由环境变量覆盖）
THS_PY = os.environ.get("THS_PYTHON",
    r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe")
KQ_USER = settings.THS_USERNAME
KQ_PASS = settings.THS_PASSWORD

def connect() -> Any:
    """建立 thsdk 连接并返回连接实例"""
    import thsdk
    ths = thsdk.THS({"username": KQ_USER, "password": KQ_PASS})
    ths.connect()
    return ths

def safe_wencai(ths: Any, query: str, retries: int = 2) -> Optional[Any]:
    """带限频+重试的 wencai_nlp"""
    for i in range(retries):
        try:
            r = ths.wencai_nlp(query)
            time.sleep(0.4)
            if r.success and r.data is not None:
                return r
        except Exception as e:
            log.warning("wencai retry %d: %s", i + 1, e)
            time.sleep(1)
    return None

def safe_query_data(ths: Any, params: dict, retries: int = 2) -> Optional[Any]:
    """带重试的 query_data"""
    for i in range(retries):
        try:
            r = ths.query_data(params)
            time.sleep(0.2)
            if r.success:
                return r
        except Exception as e:
            log.warning("query_data retry %d: %s", i + 1, e)
            time.sleep(1)
    return None

def to_df(wencai_result: Any, date_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """wencai_nlp 结果转 DataFrame"""
    if wencai_result is None or not wencai_result.success:
        return None
    data = wencai_result.data
    if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        return df
    return None

def save_parquet(df: pd.DataFrame, path: str) -> None:
    """保存 parquet，带目录创建"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Saved: %s (%d rows)", path, len(df))

def save_json(data: Any, path: str) -> None:
    """保存 JSON"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved: {path}")

def stock_code_to_ths(code: str) -> str:
    """股票代码转 thsdk 内部代码格式"""
    market = code.split(".")[0]
    if code.endswith(".SH"):
        return f"USHA{market}"
    elif code.endswith(".SZ"):
        return f"USZA{market}"
    elif code.endswith(".BJ"):
        return f"USBA{market}"
    return code

def ths_code_to_stock(ths_code: str) -> str:
    """thsdk 内部代码转回股票代码"""
    if ths_code.startswith("USHA"):
        return ths_code[4:] + ".SH"
    elif ths_code.startswith("USZA"):
        return ths_code[4:] + ".SZ"
    elif ths_code.startswith("USBA"):
        return ths_code[4:] + ".BJ"
    return ths_code
