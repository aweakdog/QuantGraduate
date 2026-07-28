"""
采集 关注圈200只 近3个月资金流数据 → parquet
单进程 + thsdk THS 上下文管理器

策略: 每只股票每日1次 wencai_nlp 批量查全部指标
约 200 × 65 × 0.4s ≈ 87min

支持断点续采: 已有 .parquet 的股票跳过
"""
import time, json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import pandas as pd
from thsdk import THS

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("batch_fund_flow")

# ── 路径 ──
_DATA_DIR = settings.DATA_DIR
DATA_DIR = settings.FUND_FLOW_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
WATCHLIST_PATH = str(settings.WATCHLIST_PATH)
KQ = {"username": settings.THS_USERNAME, "password": settings.THS_PASSWORD}

# ── 1. 读取关注圈 ──
with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
    watch_data = json.load(f)
stocks = watch_data["watchlist"]
log.info(f"[WATCH] 关注圈共 {len(stocks)} 只股票")

# ── 2. 交易日（近3个月，周一到周五） ──
END = datetime.now()
START = END - timedelta(days=90)
trade_days = []
d = START
while d <= END:
    if d.weekday() < 5:
        trade_days.append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)
log.info(f"[DAYS] 交易日: {len(trade_days)} ({trade_days[0]} ~ {trade_days[-1]})")

# ── 3. 指标映射 ──
# wencai_nlp 返回列名带 [YYYYMMDD] 后缀
def fmt_date(day):
    return day.replace("-", "")

COL_MAP = {
    "主力资金流向[{}]": "main_force_net",
    "主力增仓占比[{}]": "main_force_pct",
    "dde大单净额[{}]": "dde_net",
    "融资融券余额[{}]": "mtss_balance",
    "资金流向[{}]": "fund_flow",  # 原"资金流向(万元)"在返回中是"资金流向"
}

# ── 4. 主循环 ──
total = len(stocks)
done_stocks = 0
skipped_stocks = 0
failed_stocks = []
batch_start = time.time()

with THS(KQ) as ths:
    log.info("[THS] THS 连接成功")
    
    for idx, stock_info in enumerate(stocks):
        code = stock_info["code"]
        name = stock_info["name"]
        theme = stock_info.get("theme", "")
        
        parquet_path = DATA_DIR / f"{code}.parquet"
        
        # 断点续采
        if parquet_path.exists():
            skipped_stocks += 1
            if skipped_stocks <= 3 or skipped_stocks % 20 == 0:
                log.info(f"  [{idx+1}/{total}] ⏭ {code} {name} — 已存在")
            continue
        
        rows = []
        stock_ok = True
        for day_idx, day in enumerate(trade_days):
            try:
                date_key = fmt_date(day)
                r = ths.wencai_nlp(
                    f"{code} {day} 主力资金流向,主力增仓占比,dde大单净额,融资融券余额,资金流向(万元)"
                )
                if r.success and r.data and len(r.data) > 0:
                    row_data = r.data[0]
                    row = {"date": day}
                    for col_pattern, col_name in COL_MAP.items():
                        key = col_pattern.format(date_key)
                        val = row_data.get(key, None)
                        if val is not None:
                            try:
                                row[col_name] = float(str(val).replace(",", ""))
                            except (ValueError, TypeError):
                                row[col_name] = None
                        else:
                            row[col_name] = None
                    rows.append(row)
            except Exception as e:
                log.info(f"    ⚠ {code} {day}: {e}")
            time.sleep(0.35)  # 限频
        
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(parquet_path)
            done_stocks += 1
            elapsed = time.time() - batch_start
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            log.info(f"  [{idx+1}/{total}] ✅ {code} {name} ({theme[:6]}) — {len(df)} rows | "
                  f"⏱ {elapsed/60:.1f}min | {rate:.1f}stk/min")
        else:
            log.info(f"  [{idx+1}/{total}] ⚠ {code} {name} — 0 rows")
            failed_stocks.append(code)
            stock_ok = False
        
        if not stock_ok:
            failed_stocks.append(code)

# ── 5. 总结 ──
elapsed = time.time() - batch_start
log.info("\n" + "=" * 60)
log.info(f"[DONE] 采集完成! 耗时 {elapsed/60:.1f} 分钟")
log.info(f"  关注圈: {total} 只")
log.info(f"  ✅ 成功: {done_stocks} 只")
log.info(f"  ⏭ 已存在: {skipped_stocks} 只")
log.info(f"  ❌ 失败: {len(failed_stocks)} 只")
if failed_stocks:
    log.info(f"  失败列表: {failed_stocks}")
log.info(f"  数据目录: {DATA_DIR}")
# 列出已有文件
existing = list(DATA_DIR.glob("*.parquet"))
log.info(f"  已有 parquet: {len(existing)} 个文件")
log.info("=" * 60)
