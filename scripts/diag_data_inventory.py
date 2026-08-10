"""数据资产盘点: 每个源的行数/股票数/日期范围。

写文档和对外讲方法时要报准确数字, 不能凭印象。
用法: python scripts/diag_data_inventory.py
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rng(df, dcol):
    s = pd.to_datetime(df[dcol], errors="coerce", format="mixed")
    return f"{s.min():%Y-%m-%d} ~ {s.max():%Y-%m-%d}"


def main():
    kl = glob.glob(str(ROOT / "data/raw/kline/*.parquet"))
    print(f"K线(个股日线): {len(kl)} 个文件")
    if kl:
        k = pd.read_parquet(kl[0])
        print(f"  样本 {Path(kl[0]).stem}: {len(k)} 行, {rng(k, 'date')}, 列={list(k.columns)[:9]}")

    ff = ROOT / "data/raw/fund_flow_full/fundflow_history.parquet"
    if ff.exists():
        d = pd.read_parquet(ff)
        print(f"资金流(合并表): {len(d):,} 行, {d['code'].nunique()} 只, {rng(d, 'date')}")
        print(f"  列: {[c for c in d.columns if c not in ('code', 'date')]}")

    specs = [
        ("tushare/daily 全市场日线", "data/raw/tushare/daily/*.parquet", "trade_date"),
        ("tushare/daily_basic 估值", "data/raw/tushare/daily_basic/*.parquet", "trade_date"),
        ("tushare/margin_detail 两融", "data/raw/tushare/margin_detail/*.parquet", "trade_date"),
        ("tushare/moneyflow 资金流", "data/raw/tushare/moneyflow/*.parquet", "trade_date"),
        ("tushare/moneyflow_hsgt 北向", "data/raw/tushare/moneyflow_hsgt/*.parquet", "trade_date"),
        ("tushare/fina_indicator 财报", "data/raw/tushare/fina_indicator/*.parquet", "end_date"),
        ("tushare/index_daily 指数", "data/raw/tushare/index_daily/*.parquet", "trade_date"),
        ("tushare/stk_limit 涨跌停", "data/raw/tushare/stk_limit/*.parquet", "trade_date"),
        ("tushare/suspend_d 停牌", "data/raw/tushare/suspend_d/*.parquet", "trade_date"),
        ("tushare/top_list 龙虎榜", "data/raw/tushare/top_list/*.parquet", "trade_date"),
        ("tushare/adj_factor 复权", "data/raw/tushare/adj_factor/*.parquet", "trade_date"),
    ]
    for name, pat, dc in specs:
        fs = sorted(glob.glob(str(ROOT / pat)))
        if not fs:
            print(f"{name}: 无")
            continue
        try:
            d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            nc = d["ts_code"].nunique() if "ts_code" in d.columns else "-"
            print(f"{name}: {len(d):,} 行, {nc} 只, {rng(d, dc)}")
        except Exception as e:
            print(f"{name}: 读取失败 {type(e).__name__} {e}")

    for label, sub in [("宏观", "macro"), ("公告", "announcements"),
                       ("事件", "events"), ("iFinD事件", "events_ifind")]:
        n = len(glob.glob(str(ROOT / "data/raw" / sub / "*.parquet")))
        print(f"{label}({sub}): {n} 个文件")

    tp = ROOT / "data/processed/training_data_pit_v24.parquet"
    if tp.exists():
        t = pd.read_parquet(tp, columns=["date", "code"])
        print(f"训练矩阵: {len(t):,} 行, {t['code'].nunique()} 只, {rng(t, 'date')}")
        import pyarrow.parquet as pq
        print(f"  列数: {len(pq.read_schema(tp).names)}")

    up = ROOT / "data/universe/universe_pit.parquet"
    if up.exists():
        u = pd.read_parquet(up)
        print(f"PIT 股票池: {len(u):,} 行, 列={list(u.columns)}")
        if "code" in u.columns:
            print(f"  去重股票数: {u['code'].nunique()}")


if __name__ == "__main__":
    main()
