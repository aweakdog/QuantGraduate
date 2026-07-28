"""
SuperMind 六维数据引擎 — 历史数据管道

将 thsdk + SuperMind + 外部数据 统一为 pandas DataFrame
"""
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from pipeline.config import settings
from pipeline.logger import get_logger
log = get_logger("data_engine")

DATA_DIR = settings.DATA_DIR

# ──────── 1. wencai 历史资金面 ────────

THS_PYTHON = os.environ.get("THS_PYTHON",
    r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe")

WENCAI_METRICS = {
    "主力资金流向": "main_force_net",
    "dde大单净额": "dde_net",
    "ddx": "ddx",
    "融资融券余额": "mtss_balance",
    "资金流向(万元)": "money_flow_wan",
    "主力增仓占比": "main_force_pct",
}

def query_wencai_historical(stock: str, date: str, metric: str) -> float:
    """
    查 thsdk wencai 历史某日的资金面数据。
    返回数值，None 表示查不到。
    """
    import subprocess
    code = f"""
from thsdk import THS
import time
ths = THS()
ths.connect()
time.sleep(0.3)
r = ths.wencai_nlp("{stock} {date} {metric}")
if r.success and r.data:
    df = r.df
    cols = [c for c in df.columns if c not in ['股票代码','股票简称','最新价','最新涨跌幅']]
    if cols:
        log.info(str(df[cols[0]].iloc[0]))
    else:
        log.info("EMPTY")
else:
    log.info("ERROR:" + str(r.error[:50]))
"""
    try:
        result = subprocess.run(
            [THS_PYTHON, "-c", code],
            capture_output=True, text=True, timeout=15
        )
        out = result.stdout.strip()
        if out and out not in ["EMPTY", "None", ""]:
            return float(out.replace(',', ''))
    except (subprocess.TimeoutExpired, ValueError, OSError) as _e:
        log.warning("query_wencai_historical(%s, %s): %s", stock, date, _e)
    return None


def fetch_fund_flow_history(stock: str, start_date: str, end_date: str,
                            metrics: list = None) -> pd.DataFrame:
    """
    拉取历史资金流向数据。
    逐日查 thsdk wencai，速率受限 ~0.4s/次。
    
    返回 DataFrame，index=date，columns=各指标。
    """
    if metrics is None:
        metrics = list(WENCAI_METRICS.keys())
    
    # 获取交易日历（从 SuperMind 或本地）
    trade_days = _get_trade_days(start_date, end_date)
    log.info(f"Total trading days: {len(trade_days)}")
    
    rows = []
    total = len(trade_days) * len(metrics)
    done = 0
    
    for day in trade_days:
        row = {"date": day}
        for metric in metrics:
            val = query_wencai_historical(stock, day, metric)
            row[WENCAI_METRICS.get(metric, metric)] = val
            done += 1
            if done % 50 == 0:
                pct = done / total * 100
                log.info(f"  Progress: {done}/{total} ({pct:.1f}%)")
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    # 保存
    path = DATA_DIR / f"{stock}_fund_flow_{start_date}_{end_date}.parquet"
    df.to_parquet(path)
    log.info(f"Saved: {path}")
    return df


def _get_trade_days(start: str, end: str) -> List[str]:
    """获取交易日列表（简化版：周一到周五，排除节假日）"""
    # TODO: 用 get_all_trade_days() 从 SuperMind 获取精确交易日
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    days = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


# ──────── 2. 公告解析（query_iwencai 返回） ────────

def parse_announcements(wencai_result_df) -> pd.DataFrame:
    """
    解析 query_iwencai("601689.SH 近期公告") 的返回。
    关键词资讯 列是 Base64 → JSON 数组。
    """
    import base64
    records = []
    for _, row in wencai_result_df.iterrows():
        raw = row.get("关键词资讯", "")
        if not raw or raw == "None":
            continue
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            items = json.loads(decoded)
            for item in items:
                records.append({
                    "title": item.get("PageRawTitle", ""),
                    "url": item.get("URL", ""),
                    "publish_time": item.get("PublishTime"),
                    "channel": item.get("Channel_", ""),
                    "size": item.get("Size", 0),
                })
        except (ValueError, json.JSONDecodeError, KeyError, TypeError) as _e:
            log.debug("parse_announcements row: %s", _e)
    return pd.DataFrame(records)


def fetch_announcements(stock: str, lookback_days: int = 30) -> pd.DataFrame:
    """拉取近期公告"""
    from thsdk import THS
    ths = THS()
    ths.connect()
    
    r = ths.wencai_nlp(f"{stock} 近期公告")
    if r.success and r.data:
        return parse_announcements(r.df)
    return pd.DataFrame()


# ──────── 3. 7x24 快讯（thsdk.news） ────────

def fetch_news_flow(max_pages: int = 5) -> pd.DataFrame:
    """
    拉取同花顺 7x24 快讯。
    thsdk.news() 每次返回20条。
    """
    from thsdk import THS
    ths = THS()
    ths.connect()
    
    all_items = []
    for i in range(max_pages):
        r = ths.news()
        if r.success and r.data:
            all_items.extend(r.data)
        time.sleep(0.3)
    
    df = pd.DataFrame(all_items)
    if not df.empty and "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], unit="s")
    return df


# ──────── 4. 合并数据 (SuperMind get_price + 资金面) ────────

def merge_price_fundflow(price_df: pd.DataFrame, fundflow_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 get_price 的价量数据 与 wencai 资金面数据合并。
    按日期对齐，forward fill 资金面空缺。
    """
    df = price_df.join(fundflow_df, how="left")
    # 资金面数据可能不是每天都有（非交易日）
    # forward fill 最多3天
    df = df.fillna(method="ffill", limit=3)
    return df


if __name__ == "__main__":
    log.info("六维数据引擎框架就绪")
    log.info(f"数据目录: {DATA_DIR}")
