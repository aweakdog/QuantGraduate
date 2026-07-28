"""akshare 批量采集 — K线(新浪) + 资金流(东财) → data/raw/

用法:
  python pipeline/collect_daily_akshare.py            # 全量关注圈
  python pipeline/collect_daily_akshare.py --limit 5  # 只采前5只（测试）

K线保存: data/raw/kline/{code6}.parquet   (date/open/high/low/close/volume/amount)
资金流:  data/raw/fund_flow/{code6}.parquet
已存在的文件自动跳过。
"""
import argparse
import json
import time

import akshare as ak
import pandas as pd

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("collect_akshare")

KLINE_START = "20210101"


def load_watchlist() -> list[dict]:
    with open(settings.WATCHLIST_PATH, encoding="utf-8") as f:
        return json.load(f).get("watchlist", [])


def collect_kline(stocks: list[dict], refresh: bool = False) -> None:
    settings.KLINE_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now().strftime("%Y%m%d")
    done = skip = fail = 0
    for i, s in enumerate(stocks):
        code = s["code"]
        code6 = code[:6]
        symbol = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(code[7:], "sh") + code6
        out = settings.KLINE_DIR / f"{code6}.parquet"
        if out.exists() and not refresh:
            skip += 1
            continue
        df = None
        try:
            df = ak.stock_zh_a_daily(symbol=symbol, start_date=KLINE_START, end_date=end, adjust="qfq")
        except Exception as e:
            log.debug("sina失败 %s: %s, 试腾讯", code6, e)
            try:
                df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=KLINE_START, end_date=end, adjust="qfq")
            except Exception as e2:
                log.warning("K线失败 %s: %s", code6, e2)
        if df is not None and len(df) > 0:
            df = df.reset_index() if "date" not in df.columns else df
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(out, index=False)
            done += 1
        else:
            fail += 1
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            log.info("  K线进度: %d/%d (新增%d 跳过%d 失败%d)", i + 1, len(stocks), done, skip, fail)
    log.info("K线完成: 新增%d 跳过%d 失败%d", done, skip, fail)


def collect_fund_flow(stocks: list[dict]) -> None:
    settings.FUND_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0
    for i, s in enumerate(stocks):
        code = s["code"]
        code6 = code[:6]
        market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(code[7:], "sh")
        out = settings.FUND_FLOW_DIR / f"{code6}.parquet"
        if out.exists():
            skip += 1
            continue
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_individual_fund_flow(stock=code6, market=market)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    log.warning("资金流失败 %s: %s", code6, e)
        try:
            if df is not None and len(df) > 0:
                df.rename(columns={
                    "日期": "date", "收盘价": "close",
                    "主力净流入-净额": "main_force_net",
                    "主力净流入-净占比": "main_force_pct",
                    "超大单净流入-净额": "super_large_net",
                    "大单净流入-净额": "large_net",
                    "中单净流入-净额": "mid_net",
                    "小单净流入-净额": "small_net",
                }, inplace=True, errors="ignore")
                df["date"] = pd.to_datetime(df["date"])
                df.to_parquet(out, index=False)
                done += 1
            else:
                fail += 1
        except Exception as e:
            log.warning("资金流失败 %s: %s", code6, e)
            fail += 1
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            log.info("  资金流进度: %d/%d (新增%d 跳过%d 失败%d)", i + 1, len(stocks), done, skip, fail)
    log.info("资金流完成: 新增%d 跳过%d 失败%d", done, skip, fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只采前N只（测试用）")
    parser.add_argument("--skip-kline", action="store_true")
    parser.add_argument("--skip-fund", action="store_true")
    parser.add_argument("--refresh-kline", action="store_true")
    args = parser.parse_args()

    stocks = load_watchlist()
    if args.limit:
        stocks = stocks[: args.limit]
    log.info("关注圈: %d 只", len(stocks))

    if not args.skip_kline:
        collect_kline(stocks, refresh=args.refresh_kline)
    if not args.skip_fund:
        collect_fund_flow(stocks)
    log.info("全部完成")
