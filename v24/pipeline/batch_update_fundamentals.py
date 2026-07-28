"""
批量更新资金流和基本面历史数据

资金流: akshare stock_individual_fund_flow (6个月/股, 1次调用)
基本面: thsdk wencai_nlp 按年查历史PE/PB/市值 (10年, 每只10次调用)
"""
import sys, os, json, time, pandas as pd, numpy as np
from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("batch_fundamentals")

WATCHLIST_PATH = str(settings.WATCHLIST_PATH)
FUND_FLOW_DIR = str(settings.FUND_FLOW_DIR)
FUNDA_DIR = str(settings.PROCESSED_DIR.parent / "raw" / "fundamentals")

os.makedirs(FUND_FLOW_DIR, exist_ok=True)
os.makedirs(FUNDA_DIR, exist_ok=True)

# 加载关注圈
with open(WATCHLIST_PATH, encoding="utf-8") as f:
    watch = json.load(f)

stocks = watch.get("watchlist", [])
log.info("关注圈: %d 只", len(stocks))

# ─── 1. 资金流 (akshare, 6个月/次) ────────────────────────
log.info("=== 资金流 (akshare) ===")
import akshare as ak
for i, s in enumerate(stocks):
    code = s["code"]
    code6 = code[:6]
    market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(code[7:], "sh")

    out_path = os.path.join(FUND_FLOW_DIR, f"{code6}.parquet")
    if os.path.exists(out_path):
        continue  # 已有

    try:
        df = ak.stock_individual_fund_flow(stock=code6, market=market)
        if df is not None and len(df) > 0:
            df.rename(columns={
                "日期": "date", "收盘价": "close",
                "主力净流入-净额": "main_force_net",
                "主力净流入-净占比": "main_force_pct",
                "超大单净流入-净额": "super_large_net",
                "大单净流入-净额": "large_net",
                "中单净流入-净额": "mid_net",
                "小单净流入-净额": "small_net",
            }, inplace=True, errors='ignore')
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(out_path, index=False)
        time.sleep(0.3)
    except (ValueError, TypeError, OSError, KeyError) as e:
        log.warning("资金流失败 %s: %s", code, e)

    if (i+1) % 20 == 0:
        log.info("  进度: %d/%d", i+1, len(stocks))

log.info("  资金流完成")

# ─── 2. 基本面历史 (wencai_nlp 按年查) ────────────────────
log.info("=== 基本面历史 (thsdk wencai_nlp) ===")
import thsdk
ths = thsdk.THS({"username": settings.THS_USERNAME, "password": settings.THS_PASSWORD})
ths.connect()

years = list(range(2015, 2027))  # 2015~2026

for i, s in enumerate(stocks):
    code = s["code"]
    code6 = code[:6]

    out_path = os.path.join(FUNDA_DIR, f"{code6}.parquet")
    if os.path.exists(out_path):
        continue

    records = []
    for yr in years:
        date_str = f"{yr}-06-30"
        try:
            r = ths.wencai_nlp(f"{code} {date_str} 市盈率(pe),市净率(pb),总市值,营业收入,归属于母公司所有者的净利润,基本每股收益,每股净资产bps,净资产收益率roe(加权,公布值)")
            time.sleep(0.4)
            if r.success and r.data:
                row = r.data[0] if isinstance(r.data, list) else r.data
                yr6 = f"{yr}0630"
                records.append({
                    "date": f"{yr}-06-30",
                    "pe": row.get(f"市盈率(pe)[{yr6}]") or row.get("市盈率(pe)"),
                    "pb": row.get(f"市净率(pb)[{yr6}]") or row.get("市净率(pb)"),
                    "mcap": row.get(f"总市值[{yr6}]") or row.get("总市值"),
                    "revenue": row.get(f"营业收入[{yr6}]") or row.get("营业收入"),
                    "profit": row.get(f"归属于母公司所有者的净利润[{yr6}]") or row.get("归属于母公司所有者的净利润"),
                    "eps": row.get(f"基本每股收益[{yr6}]") or row.get("基本每股收益"),
                    "bps": row.get(f"每股净资产bps[{yr6}]") or row.get("每股净资产bps"),
                    "roe": row.get(f"净资产收益率roe(加权,公布值)[{yr6}]") or row.get("净资产收益率roe(加权,公布值)"),
                })
        except (ValueError, TypeError, KeyError) as e:
            log.debug("基本面查询 %s %d: %s", code6, yr, e)

    if records:
        pd.DataFrame(records).to_parquet(out_path, index=False)

    if (i+1) % 10 == 0:
        log.info("  进度: %d/%d", i+1, len(stocks))

log.info("  基本面历史完成")
log.info("全部完成")
