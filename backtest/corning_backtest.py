"""
康宁链主回测 — 验证"事件评分 × 供应商涨幅"的相关性

核心问题:
  1. 康宁重大事件后，A股供应商平均涨幅多少？
  2. 评分高的供应商是否比评分低的涨得多？
  3. 不同类型事件（投资vs订单vs合作vs技术）的弹性差异？

数据源:
  - 康宁历史事件时间线（手工整理）
  - A股供应商日线数据（data/raw/kline/）

用法:
  python -m backtest.corning_backtest
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# 注入 scorer 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from pipeline.config import settings
from pipeline.logger import get_logger as _gl
from chain_leader_scorer import score_event_for_leader, classify_event

log = _gl("corning_bt")

# ─── 康宁重大历史事件 ─────────────────────────────────────────

CORNING_EVENTS = [
    # (date, event_text, event_type_label)
    ("2026-06-26", "康宁发布Glass Bridge玻璃光互连新技术", "技术发布"),
    ("2026-06-25", "康宁Glass Bridge光互连技术报道", "技术发布"),
    ("2026-06-09", "亚马逊与康宁签署数十亿美元光纤长约", "订单合作"),
    ("2026-05-20", "京东方与康宁签署合作备忘录", "战略合作"),
    ("2026-05-07", "英伟达宣布投资康宁32亿美元，锁定光连接产能", "巨头投资"),
    ("2026-05-05", "英伟达与康宁达成长期合作，光通信概念股全线走高", "巨头投资"),
    ("2026-04-29", "康宁2026Q1财报：光通信营收18亿美元同比+36%，净利+93%", "财报超预期"),
    ("2026-03-15", "康宁CEO带队访问中国长飞/亨通/中天，洽谈代工合作", "高管动向"),
    ("2026-01-15", "Meta与康宁签署超60亿美元光纤长期供货协议", "订单合作"),
    ("2025-12-10", "康宁宣布光纤产能扩张计划，应对AI数据中心需求", "产能扩张"),
    ("2025-10-29", "康宁2025Q3财报：光通信营收增长强劲，上调全年指引", "财报超预期"),
    ("2025-07-30", "康宁2025Q2财报：光通信业务创新高", "财报超预期"),
    ("2025-05-15", "康宁宣布与多家云厂商洽谈光纤供应合作", "高管动向"),
    ("2025-04-28", "康宁2025Q1财报：光通信营收同比+28%", "财报超预期"),
    ("2025-01-30", "康宁2024Q4财报：全年光通信营收创纪录", "财报超预期"),
    ("2024-10-30", "康宁2024Q3财报：光通信需求回暖", "财报超预期"),
    ("2024-07-31", "康宁2024Q2财报：显示玻璃业务企稳", "财报"),
    ("2024-04-30", "康宁2024Q1财报：光通信业务复苏迹象", "财报"),
    ("2024-01-31", "康宁2023Q4财报：全年业绩符合预期", "财报"),
]

# ─── 核心供应商（只选有实锤且K线数据可能存在的）───────────────

CORE_SUPPLIERS = {
    # code6: name
    "300570": "太辰光",
    "300398": "飞凯材料",
    "300395": "菲利华",
    "000070": "特发信息",
    "002491": "通鼎互联",
    "688313": "仕佳光子",
    "688502": "茂莱光学",
    "300757": "罗博特科",
    "300394": "天孚通信",
    "300308": "中际旭创",
    "603688": "石英股份",
    "603938": "三孚股份",
    "000725": "京东方A",
    "603773": "沃格光电",
    "301051": "信濠光电",
}

KLINE_DIR = str(settings.KLINE_DIR)
OUT_DIR = str(settings.BACKTEST_DIR)


# ─── 数据加载 ─────────────────────────────────────────────────

def load_kline(code6: str) -> pd.DataFrame | None:
    path = os.path.join(KLINE_DIR, f"{code6}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    col_map = {"时间": "date", "收盘价": "close", "开盘价": "open",
               "最高价": "high", "最低价": "low", "成交量": "volume", "总金额": "amount"}
    df.rename(columns=col_map, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def find_trading_day(df: pd.DataFrame, target: str, offset: int = 0) -> str | None:
    """找 target 之后第 offset 个交易日"""
    dt = pd.to_datetime(target)
    mask = df["date"] >= dt
    available = df.loc[mask, "date"].values
    if len(available) <= offset:
        return None
    return str(available[offset])[:10]


def calc_return(df: pd.DataFrame, entry_date: str, hold_days: int) -> float | None:
    """从 entry_date 持有 hold_days 个交易日的收益率(%)"""
    entry = find_trading_day(df, entry_date, 0)
    if entry is None:
        return None
    exit_d = find_trading_day(df, entry, hold_days)
    if exit_d is None:
        return None
    ep = df.loc[df["date"] == entry, "close"].values
    xp = df.loc[df["date"] == exit_d, "close"].values
    if len(ep) == 0 or len(xp) == 0 or ep[0] == 0:
        return None
    return (xp[0] - ep[0]) / ep[0] * 100


# ─── 回测主逻辑 ───────────────────────────────────────────────

def run_backtest():
    log.info("=" * 70)
    log.info("康宁(GLW) 链主事件 → A股供应商 回测")
    log.info(f"事件数量: {len(CORNING_EVENTS)} 次 ({CORNING_EVENTS[-1][0]} ~ {CORNING_EVENTS[0][0]})")
    log.info(f"供应商: {len(CORE_SUPPLIERS)} 只")
    log.info("=" * 70)

    # 预加载K线
    log.info("\n加载K线数据...")
    klines = {}
    for code6 in CORE_SUPPLIERS:
        df = load_kline(code6)
        if df is not None:
            klines[code6] = df
    log.info(f"  成功加载 {len(klines)}/{len(CORE_SUPPLIERS)} 只")

    if not klines:
        log.info("无可用K线数据")
        return

    # 预计算每个事件的评分
    log.info("\n评分预计算...")
    event_scores = {}
    for date_str, text, _ in CORNING_EVENTS:
        # 传入 event_date 防止未来数据泄露，回测模式用价格代理
        result = score_event_for_leader(text, "Corning", event_date=date_str, mode="backtest")
        event_scores[date_str] = {
            "text": text,
            "scores": {s["code"].split(".")[0]: s["score"] for s in result["suppliers"]},
            "intensity": result["event"]["intensity"],
            "tag": result["event"]["tag"],
        }
    log.info(f"  已评分 {len(event_scores)} 个事件")

    # 三个持有窗口
    WINDOWS = [1, 3, 5, 10]

    all_results = {}

    for hold in WINDOWS:
        log.info(f"\n{'─' * 70}")
        log.info(f"持有 {hold} 个交易日")
        log.info(f"{'─' * 70}")

        event_rows = []

        for date_str, text, ev_type in CORNING_EVENTS:
            scores = event_scores[date_str]["scores"]
            intensity = event_scores[date_str]["intensity"]
            tag = event_scores[date_str]["tag"]

            stock_returns = []
            for code6, df in klines.items():
                ret = calc_return(df, date_str, hold)
                if ret is not None:
                    stock_returns.append({
                        "code": code6,
                        "name": CORE_SUPPLIERS.get(code6, code6),
                        "return": ret,
                        "score": scores.get(code6, 0),
                        "event_type": ev_type,
                        "event_tag": tag,
                        "intensity": intensity,
                    })

            if not stock_returns:
                continue

            avg_ret = np.mean([r["return"] for r in stock_returns])
            med_ret = np.median([r["return"] for r in stock_returns])
            positive = sum(1 for r in stock_returns if r["return"] > 0)
            win_rate = positive / len(stock_returns) * 100

            # 按评分分组：高分(≥6.0) vs 低分(<6.0)
            high_score = [r for r in stock_returns if r["score"] >= 6.0]
            low_score = [r for r in stock_returns if r["score"] < 6.0]

            high_avg = np.mean([r["return"] for r in high_score]) if high_score else None
            low_avg = np.mean([r["return"] for r in low_score]) if low_score else None

            # 最佳/最差个股
            best = max(stock_returns, key=lambda r: r["return"])
            worst = min(stock_returns, key=lambda r: r["return"])

            row = {
                "date": date_str,
                "event": text[:50],
                "type": ev_type,
                "intensity": intensity,
                "tag": tag,
                "avg_return": round(avg_ret, 2),
                "median_return": round(med_ret, 2),
                "win_rate": round(win_rate, 1),
                "positive": positive,
                "total": len(stock_returns),
                "high_score_avg": round(high_avg, 2) if high_avg is not None else None,
                "low_score_avg": round(low_avg, 2) if low_avg is not None else None,
                "best_stock": f"{best['name']}({best['return']:+.2f}%)",
                "worst_stock": f"{worst['name']}({worst['return']:+.2f}%)",
            }
            event_rows.append(row)

            # 打印
            bar = "█" * int(abs(avg_ret) * 2) if abs(avg_ret) < 25 else "█" * 50
            hs = f" 高分:{high_avg:+.2f}%" if high_avg is not None else ""
            ls = f" 低分:{low_avg:+.2f}%" if low_avg is not None else ""
            log.info(f"  {date_str} {tag:>12s} {avg_ret:>+6.2f}% {bar} (胜率{win_rate:.0f}%){hs}{ls}")

        # 汇总
        df_events = pd.DataFrame(event_rows)

        # 整体统计
        avg_all = df_events["avg_return"].mean()
        med_all = df_events["avg_return"].median()
        win_all = (df_events["avg_return"] > 0).mean() * 100
        total_ret = (1 + df_events["avg_return"] / 100).prod() - 1
        ret_series = df_events["avg_return"].values
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252 / hold) if np.std(ret_series) > 0 else 0

        # 高分 vs 低分区分度
        hs_vals = df_events["high_score_avg"].dropna().values
        ls_vals = df_events["low_score_avg"].dropna().values
        if len(hs_vals) > 0 and len(ls_vals) > 0:
            hs_mean = hs_vals.mean()
            ls_mean = ls_vals.mean()
            spread = hs_mean - ls_mean
        else:
            hs_mean = ls_mean = spread = None

        log.info(f"\n  [SUMMARY] 汇总 (持有{hold}天):")
        log.info(f"     平均单次收益: {avg_all:>+6.2f}%")
        log.info(f"     累计总收益:  {total_ret*100:>+6.2f}%")
        log.info(f"     胜率:         {win_all:.1f}%")
        log.info(f"     夏普:         {sharpe:.3f}")
        if spread is not None:
            log.info(f"     评分区分度:   高分{hs_mean:+.2f}% vs 低分{ls_mean:+.2f}% (差{spread:+.2f}%)")

        all_results[hold] = {
            "hold_days": hold,
            "total_events": len(event_rows),
            "avg_return_per_event": round(avg_all, 2),
            "median_return_per_event": round(med_all, 2),
            "accumulated_return": round(total_ret * 100, 2),
            "win_rate": round(win_all, 1),
            "sharpe": round(sharpe, 3),
            "high_score_avg": round(hs_mean, 2) if hs_mean is not None else None,
            "low_score_avg": round(ls_mean, 2) if ls_mean is not None else None,
            "score_spread": round(spread, 2) if spread is not None else None,
            "details": event_rows,
        }

        # 按事件类型汇总
        log.info(f"\n  [BY TYPE] 按事件类型:")
        for ev_type in df_events["type"].unique():
            sub = df_events[df_events["type"] == ev_type]
            log.info(f"     {ev_type}: {len(sub)}次, 均收益{sub['avg_return'].mean():+.2f}%")

    return all_results


# ─── 保存结果 ─────────────────────────────────────────────────

def save_results(results: dict):
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(OUT_DIR, "corning_chain_backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "chain_leader": "Corning (GLW)",
            "events_count": len(CORNING_EVENTS),
            "events_range": f"{CORNING_EVENTS[-1][0]} ~ {CORNING_EVENTS[0][0]}",
            "suppliers_count": len(CORE_SUPPLIERS),
            "results": {str(k): v for k, v in results.items()},
        }, f, ensure_ascii=False, indent=2)
    log.info(f"\n✅ 回测结果已保存: {path}")

    # CSV摘要
    rows = []
    for hold, r in results.items():
        rows.append({
            "持有天数": hold,
            "事件次数": r["total_events"],
            "平均收益%": r["avg_return_per_event"],
            "累计收益%": r["accumulated_return"],
            "胜率%": r["win_rate"],
            "夏普": r["sharpe"],
            "高分均收益%": r.get("high_score_avg", ""),
            "低分均收益%": r.get("low_score_avg", ""),
            "评分区分度": r.get("score_spread", ""),
        })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "corning_chain_backtest.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"\n[CSV] CSV摘要已保存:")
    log.info(df_out.to_string(index=False))


if __name__ == "__main__":
    results = run_backtest()
    if results:
        save_results(results)
