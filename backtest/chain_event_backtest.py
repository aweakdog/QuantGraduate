"""
链主事件驱动回测 — NVDA 财报 → A股供应商
核心逻辑: NVDA财报发布后 → 买入A股供应商 → 持有不同周期看是否跑赢基准

使用方法: python -m backtest.chain_event_backtest
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from pipeline.config import settings
from pipeline.logger import get_logger as _gl

log = _gl("chain_event_bt")

# ─── 配置 ─────────────────────────────────────────────────────

NVDA_EARNINGS = [
    # date, EPS_surprise (1=beat, -1=miss, 0=expected)
    ("2026-05-20", 1),   # Q1 FY27: beat
    ("2026-02-25", 1),   # Q4 FY26: beat
    ("2025-11-19", 1),   # Q3 FY26: beat
    ("2025-08-26", 1),   # Q2 FY26
    ("2025-05-20", 1),   # Q1 FY26
    ("2025-02-25", 1),   # Q4 FY25
    ("2024-11-20", 0),   # Q3 FY25
    ("2024-08-28", 1),   # Q2 FY25
    ("2024-05-22", 1),   # Q1 FY25
    ("2024-02-21", 1),   # Q4 FY24
    ("2023-11-21", 1),   # Q3 FY24
    ("2023-08-23", 1),   # Q2 FY24
]

# 14 NVDA A股供应商
SUPPLIERS = [
    "000938", "000977", "002281", "002463", "002916",
    "002938", "300308", "300394", "300476", "300502",
    "601138", "603019", "603986", "688008",
]

KLINE_DIR = str(settings.KLINE_DIR)

WINDOWS = [1, 3, 5, 10, 20]  # 持有天数
BENCHMARK_CODES = ["510300"]  # 沪深300 ETF

# ─── 核心函数 ─────────────────────────────────────────────────

def load_kline(code6: str) -> pd.DataFrame | None:
    """加载单只股票K线, 统一列名为英文"""
    path = os.path.join(KLINE_DIR, f"{code6}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    # 统一列名
    col_map = {
        "时间": "date", "收盘价": "close", "开盘价": "open",
        "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount",
    }
    df.rename(columns=col_map, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def find_next_trading_day(df: pd.DataFrame, target_date: str, offset: int = 0) -> str:
    """找 target_date 之后第 offset 个交易日"""
    dt = pd.to_datetime(target_date)
    mask = df["date"] >= dt
    available = df.loc[mask, "date"].values
    idx = min(offset, len(available) - 1)
    return str(available[idx])[:10] if len(available) > offset else None

def price_change(df: pd.DataFrame, entry_date: str, hold_days: int) -> float | None:
    """计算从 entry_date 开始持有 hold_days 个交易日的收益率"""
    entry = find_next_trading_day(df, entry_date, 0)
    if entry is None:
        return None
    exit_d = find_next_trading_day(df, entry, hold_days)
    if exit_d is None:
        return None
    entry_p = df.loc[df["date"] == entry, "close"].values
    exit_p = df.loc[df["date"] == exit_d, "close"].values
    if len(entry_p) == 0 or len(exit_p) == 0:
        return None
    return (exit_p[0] - entry_p[0]) / entry_p[0] * 100  # %

# ─── 回测主逻辑 ───────────────────────────────────────────────

def run_backtest():
    log.info("=" * 60)
    log.info("NVDA 财报事件 → A股供应商 回测")
    log.info(f"供应商数量: {len(SUPPLIERS)}")
    log.info(f"财报事件: {len(NVDA_EARNINGS)} 次 ({NVDA_EARNINGS[0][0]} ~ {NVDA_EARNINGS[-1][0]})")
    log.info(f"持有窗口(交易日): {WINDOWS}")
    log.info("=" * 60)

    # 预加载所有K线
    log.info("\n[LOAD] 加载K线数据...")
    klines = {}
    for code6 in SUPPLIERS:
        df = load_kline(code6)
        if df is not None:
            klines[code6] = df
    log.info(f"  成功加载 {len(klines)}/{len(SUPPLIERS)} 只")

    if len(klines) == 0:
        log.info("❌ 没有可用K线数据")
        return

    # 对每个持有窗口做回测
    results = {}

    for hold in WINDOWS:
        log.info(f"\n{'─' * 40}")
        log.info(f"[HOLD] 持有 {hold} 个交易日")
        log.info(f"{'─' * 40}")

        event_returns = []

        for earn_date, surprise in NVDA_EARNINGS:
            # 计算每只供应商的收益
            stock_returns = []
            for code6, df in klines.items():
                ret = price_change(df, earn_date, hold)
                if ret is not None:
                    stock_returns.append(ret)

            if len(stock_returns) == 0:
                continue

            avg_ret = np.mean(stock_returns)
            med_ret = np.median(stock_returns)
            positive = sum(1 for r in stock_returns if r > 0)
            total = len(stock_returns)
            win_rate = positive / total * 100

            direction = "[BEAT]超预期" if surprise == 1 else ("[MISS]低于预期" if surprise == -1 else "[EXP]符合预期")
            event_returns.append({
                "date": earn_date,
                "direction": direction,
                "avg_return": round(avg_ret, 2),
                "median_return": round(med_ret, 2),
                "win_rate": round(win_rate, 1),
                "positive": positive,
                "total": total,
            })

            bar = "█" * int(abs(avg_ret) / 2) if abs(avg_ret) < 50 else "█" * 25
            log.info(f"  {earn_date} {direction:>10s}: {avg_ret:>+6.2f}% {bar} (胜率{win_rate:.0f}%)")

        # 汇总
        if not event_returns:
            continue

        df_events = pd.DataFrame(event_returns)
        avg_all = df_events["avg_return"].mean()
        med_all = df_events["avg_return"].median()
        win_all = sum(1 for r in event_returns if r["avg_return"] > 0) / len(event_returns) * 100
        total_ret = (1 + df_events["avg_return"] / 100).prod() - 1

        # 计算夏普比
        ret_series = df_events["avg_return"].values
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252 / hold) if np.std(ret_series) > 0 else 0

        results[hold] = {
            "hold_days": hold,
            "total_events": len(event_returns),
            "avg_return_per_event": round(avg_all, 2),
            "median_return_per_event": round(med_all, 2),
            "accumulated_return": round(total_ret * 100, 2),
            "win_rate": round(win_all, 1),
            "sharpe": round(sharpe, 3),
            "details": event_returns,
        }

        log.info(f"\n  [SUMMARY] 汇总 (持有{hold}天):")
        log.info(f"     平均单次收益: {avg_all:>+6.2f}%")
        log.info(f"     累计总收益: {total_ret*100:>+6.2f}%")
        log.info(f"     胜率: {win_all:.1f}%")
        log.info(f"     夏普: {sharpe:.3f}")

    return results

# ─── 输出 ─────────────────────────────────────────────────────

def save_results(results: dict):
    out_dir = str(settings.BACKTEST_DIR)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "nvda_chain_backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "chain_leader": "NVIDIA (NVDA)",
            "suppliers_count": len(SUPPLIERS),
            "earnings_count": len(NVDA_EARNINGS),
            "earnings_range": f"{NVDA_EARNINGS[-1][0]} ~ {NVDA_EARNINGS[0][0]}",
            "results": {str(k): v for k, v in results.items()},
        }, f, ensure_ascii=False, indent=2)
    log.info(f"\n✅ 结果已保存: {path}")

    # 输出简洁CSV
    rows = []
    for hold, r in results.items():
        rows.append({
            "持有天数": hold,
            "事件次数": r["total_events"],
            "平均单次收益%": r["avg_return_per_event"],
            "累计收益%": r["accumulated_return"],
            "胜率%": r["win_rate"],
            "夏普": r["sharpe"],
        })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "nvda_chain_backtest.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"\n[CSV] CSV摘要已保存: {csv_path}")
    log.info(df_out.to_string(index=False))

if __name__ == "__main__":
    results = run_backtest()
    if results:
        save_results(results)
