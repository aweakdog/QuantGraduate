"""Tushare 落盘数据完整性检查

回填是多进程/可中断的, 必须能独立验证"到底齐不齐", 而不是相信日志。
逐接口比对: 已落盘的交易日 vs 交易日历应有的交易日。

    python scripts/check_tushare_data.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "data" / "raw" / "tushare"
START, END = "20190101", "20260805"


def calendar():
    p = TS / "trade_cal.parquet"
    if not p.exists():
        print("没有 trade_cal 缓存, 无法比对")
        return None
    cal = pd.read_parquet(p)
    cal["cal_date"] = cal["cal_date"].astype(str)
    m = (cal["is_open"] == 1) & cal["cal_date"].between(START, END)
    return sorted(cal.loc[m, "cal_date"])


def main():
    want = calendar()
    print(f"交易日历: {len(want)} 个交易日 ({START}~{END})\n")
    print(f"{'接口':<20}{'行数':>13}{'覆盖天数':>10}{'缺失天数':>10}  状态")
    print("-" * 68)

    for d in sorted(TS.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.parquet"))
        if not files:
            print(f"{d.name:<20}{'-':>13}{'-':>10}{'-':>10}  空目录")
            continue
        rows, days, dupe = 0, set(), 0
        datecol = None
        for f in files:
            df = pd.read_parquet(f)
            rows += len(df)
            for c in ("trade_date", "ann_date", "in_date"):
                if c in df.columns:
                    datecol = c
                    days |= set(df[c].astype(str).dropna())
                    break
            keys = [c for c in ("ts_code", datecol) if c and c in df.columns]
            if keys:
                dupe += len(df) - len(df.drop_duplicates(subset=keys))
        if datecol == "trade_date" and want:
            miss = [x for x in want if x not in days]
            status = "完整" if not miss else f"缺 {len(miss)} 天"
            if miss and len(miss) <= 6:
                status += " " + ",".join(miss)
            print(f"{d.name:<20}{rows:>13,}{len(days):>10}{len(miss):>10}  {status}"
                  + (f" | 重复行 {dupe}" if dupe else ""))
        else:
            print(f"{d.name:<20}{rows:>13,}{len(days):>10}{'n/a':>10}  "
                  f"(按 {datecol or '无日期列'} 存)"
                  + (f" | 重复行 {dupe}" if dupe else ""))


if __name__ == "__main__":
    main()
