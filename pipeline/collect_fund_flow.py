"""采集拓普集团历史资金流向 — 用同花顺正式账号"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from thsdk import THS
import time

from pipeline.config import settings
from pipeline.logger import get_logger
log = get_logger("collect_fund_flow")

STOCK = "601689.SH"
D = settings.DATA_DIR / "raw" / "fund_flow"
D.mkdir(parents=True, exist_ok=True)
KQ = {"username": settings.THS_USERNAME, "password": settings.THS_PASSWORD}

METRICS = [
    ("主力资金流向", "main_force_net"),
    ("dde大单净额", "dde_net"),
    ("ddx", "ddx"),
    ("融资融券余额", "mtss_balance"),
    ("资金流向(万元)", "money_flow_wan"),
    ("主力增仓占比", "main_force_pct"),
]

end = datetime.now()
start = end - timedelta(days=180)
dates = [(start + timedelta(days=x)).strftime("%Y-%m-%d")
         for x in range((end - start).days + 1)
         if (start + timedelta(days=x)).weekday() < 5]

log.info(f"交易日: {len(dates)}")

with THS(KQ) as ths:
    log.info("正式账号连接成功")
    rows = []
    for i, date in enumerate(dates):
        row = {"date": date}
        for mn, cn in METRICS:
            try:
                r = ths.wencai_nlp(f"{STOCK} {date} {mn}")
                if r.success and r.data:
                    cols = [c for c in r.df.columns if c not in ("股票代码","股票简称","最新价","最新涨跌幅")]
                    if cols:
                        v = str(r.df[cols[0]].iloc[0])
                        row[cn] = float(v.replace(",","")) if v and v != "None" else None
            except (ValueError, TypeError, AttributeError) as _e:
                log.debug("wencai %s %s %s -> %s", STOCK, date, mn, _e)
            time.sleep(0.35)
        rows.append(row)
        if (i + 1) % 20 == 0:
            log.info(f"  [{i+1}/{len(dates)}] {date}")

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()
df.to_csv(D / f"{STOCK}_fund_flow_6m.csv")
log.info(f"\nDone! {len(df)} rows")
for c in df.columns:
    filled = df[c].notna().sum()
    log.info(f"  {c}: {filled}/{len(df)} ({filled/len(df)*100:.0f}%)")
