"""
supplier_self_event.py — A 股供应商自身事件检测与评分修正

用途:
  在链主事件驱动的评分体系中，加入"供应商自身事件修正"维度。
  当链主有正面事件时，供应商如果自身有负面问题（窗口指导/产能不及预期等），
  该修正系数会降低其综合评分，反之亦然。

数据源:
  - UFD wencai (同花顺问财): 结构化的利空/利好资讯
  - 价格异常代理: 回测模式下用价格行为估算

修正公式:
  综合评分 = (事件强度×0.35 + 绑定度×0.25) × 资金乘数 × 历史乘数 × 自身事件乘数

  self_event_multiplier 范围: 0.85 ~ 1.15 (中性=1.0)
  严重利空 → 0.85  一般利空 → 0.95  中性 → 1.0
  一般利好 → 1.05  重大利好 → 1.15

用法:
  # 实时模式
  from pipeline.supplier_self_event import SelfEventChecker
  checker = SelfEventChecker()
  result = checker.check_supplier("002463", event_date="2026-06-01")
  print(result)  # {代码, 名称, 事件列表, 修正系数, ...}

  # 批量模式
  results = checker.check_suppliers(["002463", "300570"])
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import settings
from pipeline.logger import get_logger

log = get_logger("self_event")

# ─── UFD 路由路径 ──────────────────────────────────────────────

UFD_ROUTER = settings.UFD_ROUTER

# ─── 自身事件修正系数映射 ──────────────────────────────────────

SELF_EVENT_MULTIPLIERS = settings.SELF_EVENT_MULTIPLIERS

# ─── 负面事件关键词分类规则 ─────────────────────────────────────
# 按 severity 从高到低排列

NEGATIVE_RULES = [
    # 严重利空（severe_negative）— 股价直接受冲击
    ("severe_negative", [
        "立案", "调查", "被ST", "st", "退市", "暂停上市",
        "窗口指导", "监管函", "监管措施", "行政处罚", "责令改正",
        "业绩预亏", "业绩亏损", "大幅下降", "净利润下降", "亏损",
        "债务违约", "信用评级下调", "破产", "重组失败",
        "财务造假", "虚假陈述", "内幕交易",
    ]),
    # 中等利空（moderate_negative）— 基本面或预期恶化
    ("moderate_negative", [
        "产能爬坡", "不及预期", "产能", "低于预期",
        "订单下滑", "订单减少", "需求下滑", "需求疲软",
        "减持", "减持股份", "减持计划",
        "大股东", "质押", "质押平仓",
        "商誉减值", "资产减值", "计提",
        "毛利率下降", "市场份额", "下滑",
        "诉讼", "仲裁", "纠纷",
        "IPO过程预警", "风险澄清",
    ]),
    # 一般利空（minor_negative）— 短期情绪影响
    ("minor_negative", [
        "交易异常波动",
        "风险提示", "澄清公告",
        "董事辞职", "高管辞职",
        "董秘", "变更", "会计",
    ]),
]

# ─── 中性/忽略事件关键词（新闻快讯类，无实质影响）────────────

NEUTRAL_RULES = [
    "振幅超过", "振幅异常", "振幅",   # 波动通知，非利空
    "快讯", "7x24", "金融研究中心",    # 系统快讯
    # "有限公司"/"股份有限公司" 已移除 — 几乎所有A股公司名都含这些词，导致误判中性
    "分时", "日内",                   # 分时数据
    "派息", "除权除息", "股权登记日",  # 分红实施流程，中性
]

# ─── 正面事件关键词分类规则 ─────────────────────────────────────

POSITIVE_RULES = [
    # 重大利好（major_positive）
    ("major_positive", [
        "业绩预增", "净利润同比增长", "净利润增长", "营收增长",
        "中标", "大额合同", "大单", "订单",
        "产能", "满产", "扩产", "投产",
        "突破", "自主研发", "国产替代",
        "新产品", "新技术", "专利",
        "战略合作", "签", "备忘录",
    ]),
    # 轻度利好（minor_positive）
    ("minor_positive", [
        "股东增持", "回购", "分红",
        "机构调研", "买入评级", "增持评级",
        "入选", "纳入", "指数",
    ]),
]


def _query_wencai(query: str) -> list[dict]:
    """调用 UFD wencai 查询，返回数据列表"""
    if not os.path.exists(UFD_ROUTER):
        return []
    try:
        cmd = [sys.executable or "python", UFD_ROUTER, "wencai", query]
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            return []
        # 解析 JSON
        stdout = r.stdout.strip()
        start = stdout.find("{")
        if start < 0:
            return []
        depth, end = 0, start
        for i, ch in enumerate(stdout[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        if end <= start:
            return []
        result = json.loads(stdout[start:end])
        if result.get("success") and result.get("data"):
            return result["data"]
        return []
    except (json.JSONDecodeError, KeyError, TypeError) as _e:
        log.warning("_query_wencai failed for %s: %s", query, _e)
        return []


def _extract_field(item: dict, field_prefix: str) -> tuple[str, str]:
    """
    从 wencai 条目中提取特定前缀的字段值.
    因为列名有日期后缀如 `负面资讯标题[20260529-20260629]`，需要模糊匹配。

    返回 (值, 完整列名)
    """
    for key, val in item.items():
        if key.startswith(field_prefix):
            return str(val) if val is not None else "", key
    return "", ""


def _extract_news_items(data: list[dict]) -> list[dict]:
    """
    从 wencai 返回数据中提取新闻资讯事件。
    过滤掉只含"重要事件"（公告类）不含负面资讯的记录。
    """
    items = []
    seen_titles = set()
    for row in data:
        title, _ = _extract_field(row, "负面资讯标题")
        risk, _ = _extract_field(row, "负面资讯风险类别")
        news_time, _ = _extract_field(row, "负面资讯时间")
        importance, _ = _extract_field(row, "负面资讯重要性")

        if title and title not in seen_titles:
            seen_titles.add(title)
            items.append({
                "title": title,
                "time": news_time,
                "risk_category": risk,
                "importance": int(importance) if importance and importance.isdigit() else 0,
                "source": "wencai_negative",
            })

        # 也捕获重要事件（公告类）
        ann_name, _ = _extract_field(row, "重要事件名称")
        ann_content, _ = _extract_field(row, "重要事件内容")
        ann_time, _ = _extract_field(row, "重要事件公告时间")
        if ann_name and (ann_name, ann_time) not in {(i.get("title"), i.get("time")) for i in items}:
            items.append({
                "title": f"{ann_name}: {ann_content[:50]}" if ann_content else ann_name,
                "time": ann_time,
                "risk_category": "公告",
                "importance": 0,
                "source": "wencai_announcement",
            })

    return items


def classify_event(item: dict) -> tuple[str, str]:
    """
    对单个事件进行分类，返回 (severity_level, matched_keyword)

    severity_level: severe_negative / moderate_negative / minor_negative /
                    major_positive / minor_positive / neutral
    """
    title = item.get("title", "")
    risk = item.get("risk_category", "")
    source = item.get("source", "")

    title_lower = title.lower()

    # 1. 先检查中性规则（技术性通知/系统消息，无实质影响）
    for kw in NEUTRAL_RULES:
        if kw.lower() in title_lower:
            return ("neutral", f"系统-{kw}")

    # 2. 检查风险类别映射（优先于关键词，减少误伤）
    risk_map = {
        "交易异常波动": ("neutral", "振幅异常"),           # 技术性通知，非利空
        "风险澄清": ("neutral", "澄清公告"),               # 澄清不等于利空
        "IPO过程预警": ("minor_negative", "IPO预警"),
        "行政处罚": ("severe_negative", "行政处罚"),
    }
    if risk in risk_map:
        return risk_map[risk]

    # 3. 负面事件（按优先级匹配）
    for level, keywords in NEGATIVE_RULES:
        for kw in keywords:
            if kw.lower() in title_lower:
                return level, f"负面-{kw}"

    # 4. 正面事件
    for level, keywords in POSITIVE_RULES:
        for kw in keywords:
            if kw.lower() in title_lower:
                return level, f"正面-{kw}"

    # 5. 中性兜底
    return ("neutral", "一般信息")


def calculate_adjustment(
    events: list[dict],
    event_date: str | None = None,
    window_days: int = 7,
) -> dict[str, Any]:
    """
    根据事件列表计算自身事件修正系数。

    Args:
        events: 事件字典列表（含分类后的 severity）
        event_date: 链主事件日期 YYYY-MM-DD，如果提供则只考虑 window 内的事件
        window_days: 前后窗口天数（默认前后7天）

    Returns:
        {
            "self_event_multiplier": 0.92,
            "net_impact": -1,          # 综合影响（负面-3~正面+3）
            "severe_count": 1,         # 各等级计数
            "moderate_count": 0,
            "minor_count": 0,
            "positive_count": 0,
            "major_positive_count": 0,
            "events_in_window": [...],  # 窗口内的事件
            "all_events": [...],       # 全部事件
        }
    """
    score_map = {
        "severe_negative": -3,
        "moderate_negative": -2,
        "minor_negative": -1,
        "neutral": 0,
        "minor_positive": 1,
        "major_positive": 2,
    }

    # 筛选窗口内事件
    filtered = events
    if event_date:
        try:
            base_dt = datetime.strptime(event_date, "%Y-%m-%d")
            start_dt = base_dt - timedelta(days=window_days)
            end_dt = base_dt + timedelta(days=window_days)
            in_window = []
            for ev in events:
                ev_time = ev.get("time", "")
                if not ev_time or len(ev_time) < 8:
                    continue
                try:
                    ev_dt = datetime.strptime(str(ev_time)[:8], "%Y%m%d")
                    if start_dt <= ev_dt <= end_dt:
                        in_window.append(ev)
                except ValueError:
                    continue
            filtered = in_window
        except ValueError:
            pass

    if not filtered:
        return {
            "self_event_multiplier": 1.0,
            "net_impact": 0,
            "severe_count": 0,
            "moderate_count": 0,
            "minor_count": 0,
            "positive_count": 0,
            "major_positive_count": 0,
            "events_in_window": [],
            "all_events": events,
        }

    # 对每个事件分类
    raw_classified = []
    for ev in filtered:
        severity, matched = classify_event(ev)
        raw_classified.append({
            **ev,
            "classified_severity": severity,
            "matched_keyword": matched,
        })

    # ─── 去重逻辑 ──────────────────────────────
    # 同类事件（相同 severity + 相同关键词）只计 1 次，防止"振幅异常×6"堆叠
    # 但严重利空（severe_negative）不过滤
    seen_groups = set()
    deduped = []
    for ev in raw_classified:
        severity = ev["classified_severity"]
        keyword = ev.get("matched_keyword", "")
        # 严重利空不去重（每个都是独立事件）
        if severity == "severe_negative":
            deduped.append(ev)
            continue
        # 同类中性事件只保留一个
        group_key = f"{severity}|{keyword}"
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        deduped.append(ev)

    # ─── 统计 ──────────────────────────────────
    net_impact = 0
    severe_count = moderate_count = minor_count = positive_count = major_positive_count = 0

    for ev in deduped:
        severity = ev["classified_severity"]
        impact = score_map.get(severity, 0)
        net_impact += impact

        if severity == "severe_negative":
            severe_count += 1
        elif severity == "moderate_negative":
            moderate_count += 1
        elif severity == "minor_negative":
            minor_count += 1
        elif severity == "minor_positive":
            positive_count += 1
        elif severity == "major_positive":
            major_positive_count += 1

    # 计算修正系数
    if net_impact <= -5:
        mult = 0.85
    elif net_impact <= -3:
        mult = 0.88
    elif net_impact <= -2:
        mult = 0.92
    elif net_impact <= -1:
        mult = 0.95
    elif net_impact == 0:
        mult = 1.00
    elif net_impact <= 1:
        mult = 1.05
    elif net_impact <= 3:
        mult = 1.10
    else:
        mult = 1.15

    return {
        "self_event_multiplier": mult,
        "net_impact": net_impact,
        "severe_count": severe_count,
        "moderate_count": moderate_count,
        "minor_count": minor_count,
        "positive_count": positive_count,
        "major_positive_count": major_positive_count,
        "events_in_window": deduped,
        "all_events": events,
    }


def format_events(events: list[dict], max_display: int = 5) -> str:
    """格式化事件列表为可读文本"""
    if not events:
        return "  (无相关事件)"

    lines = []
    for ev in events[:max_display]:
        title = ev.get("title", "")[:60]
        time_str = ev.get("time", "")
        severity = ev.get("classified_severity", "unknown")
        sev_icon = {
            "severe_negative": "[!!]", "moderate_negative": "[!]",
            "minor_negative": "[-]", "neutral": "[=]",
            "minor_positive": "[+]", "major_positive": "[++]",
        }.get(severity, "[=]")
        lines.append(f"    {sev_icon} [{time_str}] {title} ({severity})")
    if len(events) > max_display:
        lines.append(f"    ... 还有 {len(events) - max_display} 条")

    return "\n".join(lines)


# ─── SelfEventChecker ──────────────────────────────────────────

class SelfEventChecker:
    """
    供应商自身事件检查器。

    实时模式:
      checker = SelfEventChecker()
      result = checker.check_supplier("002463", event_date="2026-06-01")
      # → {code, name, multiplier, events, analysis}

    回测代理模式:
      checker = SelfEventChecker()
      result = checker.backtest_proxy(kline_df, event_date)
      # → {self_event_multiplier, proxy_signal}
    """

    def __init__(self):
        pass

    def check_supplier(
        self,
        code6: str,
        name: str = "",
        event_date: str | None = None,
        window_days: int = 7,
    ) -> dict[str, Any]:
        """
        对单个供应商进行自身事件检测。

        Args:
            code6: 6位股票代码 (如 "002463")
            name: 股票简称
            event_date: 链主事件日期 YYYY-MM-DD，用于时间窗口过滤
            window_days: 前后窗口天数

        Returns:
            {
                "code": "002463",
                "name": "沪电股份",
                "self_event_multiplier": 0.95,
                "events_found": 5,
                "events": [...],
                "analysis": "...",
            }
        """
        # 1. 查询负面事件
        neg_query = f"{name or code6} 利空 重大事件"
        neg_data = _query_wencai(neg_query)

        # 2. 提取新闻条目
        neg_items = _extract_news_items(neg_data)

        # 3. 如果没有负面新闻，尝试查利好/重大事件
        if not neg_items:
            ann_query = f"{name or code6} 重大事件 公告"
            ann_data = _query_wencai(ann_query)
            ann_items = _extract_news_items(ann_data)
            all_items = ann_items
        else:
            all_items = neg_items

        # 4. 计算修正系数
        adj = calculate_adjustment(all_items, event_date, window_days)

        # 5. 分析摘要
        analysis_parts = []
        if adj["severe_count"] > 0:
            analysis_parts.append(f"严重利空{adj['severe_count']}项（窗口指导/监管/业绩预亏）")
        if adj["moderate_count"] > 0:
            analysis_parts.append(f"中等利空{adj['moderate_count']}项（产能/减持/订单下滑）")
        if adj["minor_count"] > 0:
            analysis_parts.append(f"一般利空{adj['minor_count']}项")
        if adj["positive_count"] > 0 or adj["major_positive_count"] > 0:
            analysis_parts.append(f"利好{adj['positive_count']+adj['major_positive_count']}项")
        analysis = "；".join(analysis_parts) if analysis_parts else "无明显自身事件"
        analysis += f" → 修正系数 {adj['self_event_multiplier']:.2f}"

        return {
            "code": code6,
            "name": name,
            "self_event_multiplier": adj["self_event_multiplier"],
            "net_impact": adj["net_impact"],
            "events_found": len(adj["events_in_window"]),
            "events_in_window": adj["events_in_window"],
            "all_events_count": len(all_items),
            "analysis": analysis,
            "breakdown": {
                "severe_count": adj["severe_count"],
                "moderate_count": adj["moderate_count"],
                "minor_count": adj["minor_count"],
                "positive_count": adj["positive_count"],
                "major_positive_count": adj["major_positive_count"],
            },
        }

    def check_suppliers(
        self,
        code_info: list[tuple[str, str]],
        event_date: str | None = None,
        window_days: int = 7,
    ) -> list[dict[str, Any]]:
        """
        批量检查多个供应商。

        Args:
            code_info: [(code6, name), ...]
            event_date: 链主事件日期
            window_days: 前后窗口天数

        Returns:
            [result_dict, ...]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        def _check(item):
            code, name = item
            try:
                return self.check_supplier(code, name, event_date, window_days)
            except Exception as e:
                return {
                    "code": code,
                    "name": name,
                    "self_event_multiplier": 1.0,
                    "error": str(e),
                    "analysis": f"查询失败: {e}",
                }

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_check, item): item for item in code_info}
            for f in as_completed(futures):
                result = f.result()
                results.append(result)

        # 按修正系数从小到大排序（最需关注的先排）
        results.sort(key=lambda r: r.get("self_event_multiplier", 1.0))
        return results

    # ─── 回测代理模式 ─────────────────────────────────────────

    def backtest_proxy(
        self,
        kline_df: pd.DataFrame,
        event_date: str,
        lookback_days: int = 5,
    ) -> dict[str, Any]:
        """
        回测模式下，通过价格异常信号代理自身事件检测。
        无法实时查询历史 wencai 数据时使用。

        原理:
          - 如果在链主事件前 N 天，供应商出现异常下跌（>2σ 或 >5%），
            视为潜在的自身负面事件信号
          - 此代理是保守的（宁可误报也不漏报）

        Args:
            kline_df: 供应商日线 DataFrame (含 date, close/收盘价)
            event_date: 链主事件日期
            lookback_days: 回溯天数

        Returns:
            {
                "self_event_multiplier": 0.95,
                "proxy_signal": "异常下跌: -6.2%",
                "price_change_pre_event": -6.2,
                "confidence": "medium",
            }
        """
        df = kline_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "时间" in df.columns:
            df["date"] = pd.to_datetime(df["时间"])
        else:
            return {"self_event_multiplier": 1.0, "proxy_signal": "无日期列"}

        df.sort_values("date", inplace=True)
        event_dt = pd.to_datetime(event_date)

        # 找事件前数据
        before_event = df[df["date"] < event_dt].tail(lookback_days + 20)
        if len(before_event) < lookback_days + 5:
            return {"self_event_multiplier": 1.0, "proxy_signal": "数据不足"}

        # 计算最近 lookback_days 的每日收益率
        close_col = "close" if "close" in df.columns else "收盘价"
        if close_col not in before_event.columns:
            return {"self_event_multiplier": 1.0, "proxy_signal": "无价格列"}

        pre_prices = before_event[close_col].values
        if len(pre_prices) < lookback_days + 1:
            return {"self_event_multiplier": 1.0, "proxy_signal": "价格数据不足"}

        # 事件前 lookback_days 个交易日的价格变化
        pre_returns = []
        for i in range(1, len(pre_prices)):
            ret = (pre_prices[i] - pre_prices[i - 1]) / pre_prices[i - 1] * 100
            pre_returns.append(ret)

        # 最近 N 天的累计收益
        recent_rets = pre_returns[-lookback_days:] if len(pre_returns) >= lookback_days else pre_returns
        cumulative_return = sum(recent_rets)

        # 更长期的日收益（用于计算 baseline）
        baseline_rets = pre_returns[:-lookback_days] if len(pre_returns) > lookback_days else pre_returns
        baseline_avg = np.mean(baseline_rets) if baseline_rets else 0
        baseline_std = np.std(baseline_rets) if len(baseline_rets) > 1 else 1.0

        # 异常判定
        event_day_return = recent_rets[-1] if recent_rets else 0

        if cumulative_return <= -8:
            # 严重异常：累计跌超8%
            multiplier = 0.88
            confidence = "high"
            signal = f"累计异常下跌: {cumulative_return:.1f}%"
        elif cumulative_return <= -5:
            # 明显异常：累计跌超5%
            multiplier = 0.92
            confidence = "medium"
            signal = f"明显下跌: {cumulative_return:.1f}%"
        elif cumulative_return <= -3:
            # 轻微异常
            multiplier = 0.96
            confidence = "low"
            signal = f"小幅下跌: {cumulative_return:.1f}%"
        elif cumulative_return >= 3:
            # 正面信号：事件前上涨（放在2σ之前，避免低波动数据的误判）
            multiplier = 1.05
            confidence = "low"
            signal = f"事件前上涨: {cumulative_return:+.1f}%"
        elif baseline_std > 0 and abs(event_day_return - baseline_avg) > 2 * baseline_std:
            # 单日异常波动（方向不确定时保守修正）
            mult_dir = 0.95 if event_day_return < 0 else 1.05
            multiplier = mult_dir
            confidence = "low"
            signal = f"单日异常波动: {event_day_return:+.1f}% (2σ={2*baseline_std:.1f}%)"
        else:
            multiplier = 1.0
            confidence = "none"
            signal = "无异常"

        return {
            "self_event_multiplier": multiplier,
            "proxy_signal": signal,
            "price_change_pre_event": round(cumulative_return, 2),
            "confidence": confidence,
            "lookback_days": lookback_days,
        }


# ─── 快捷函数 ──────────────────────────────────────────────────

def get_self_event_multiplier(
    code6: str,
    name: str = "",
    event_date: str | None = None,
    mode: str = "live",
    kline_df: pd.DataFrame | None = None,
) -> float:
    """
    获取供应商自身事件修正系数（快捷接口，供 scorer 集成使用）。

    Args:
        code6: 6位股票代码
        name: 股票简称
        event_date: 链主事件日期
        mode: "live" (实时查wencai) 或 "backtest" (用价格代理)
        kline_df: 回测模式下需要K线数据

    Returns:
        self_event_multiplier: 0.85 ~ 1.15
    """
    if mode == "live":
        checker = SelfEventChecker()
        result = checker.check_supplier(code6, name, event_date)
        return result["self_event_multiplier"]

    elif mode == "backtest" and kline_df is not None:
        checker = SelfEventChecker()
        result = checker.backtest_proxy(kline_df, event_date or "")
        return result["self_event_multiplier"]

    return 1.0  # fallback


def get_self_event_analysis(
    code6: str, name: str = "", event_date: str | None = None,
) -> str:
    """获取格式化分析文本"""
    checker = SelfEventChecker()
    result = checker.check_supplier(code6, name, event_date)
    lines = [
        f"[ANALYSIS] {name or code6} 自身事件分析",
        f"  修正系数: {result['self_event_multiplier']:.2f} (影响{result['net_impact']:+d}点)",
        f"  事件数量: {result['events_found']} 条窗口内 / {result['all_events_count']} 条总计",
        f"  严重利空: {result['breakdown']['severe_count']}",
        f"  中等利空: {result['breakdown']['moderate_count']}",
        f"  一般利空: {result['breakdown']['minor_count']}",
        f"  利好: {result['breakdown']['positive_count']+result['breakdown']['major_positive_count']}",
        "",
        "  事件明细:",
    ]
    lines.append(format_events(result.get("events_in_window", []), max_display=8))
    lines.append("")
    lines.append(f"  => {result['analysis']}")
    return "\n".join(lines)


# ─── 主入口 — 测试/演示 ─────────────────────────────────────

def main():
    """命令行接口"""
    args = sys.argv[1:]

    if not args or "--help" in args:
        print("供应商自身事件修正模块")
        print()
        print("用法:")
        print("  python supplier_self_event.py check <code6> [name] [date]")
        print("  python supplier_self_event.py backtest <code6> [date]")
        print("  python supplier_self_event.py --example")
        print()
        print("示例:")
        print("  python supplier_self_event.py check 002463 沪电股份 2026-06-01")
        print("  python supplier_self_event.py check 300570 太辰光")
        return

    if "--example" in args:
        print(get_self_event_analysis("002463", "沪电股份", "2026-06-15"))
        print()
        print(get_self_event_analysis("300570", "太辰光", "2026-06-20"))
        return

    if args[0] == "check":
        code = args[1] if len(args) > 1 else "002463"
        name = args[2] if len(args) > 2 else ""
        date = args[3] if len(args) > 3 else None
        print(get_self_event_analysis(code, name, date))
        return

    if args[0] == "backtest":
        code = args[1] if len(args) > 1 else "002463"
        date = args[2] if len(args) > 2 else "2026-01-15"
        # 本地加载K线
        kline_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "kline"
        kline_path = kline_dir / f"{code}.parquet"
        if os.path.exists(kline_path):
            df = pd.read_parquet(kline_path)
            checker = SelfEventChecker()
            result = checker.backtest_proxy(df, date)
            print(f"回测代理: {code} @ {date}")
            print(f"  修正系数: {result['self_event_multiplier']:.2f}")
            print(f"  信号: {result['proxy_signal']}")
            print(f"  置信度: {result['confidence']}")
        else:
            print(f"K线文件不存在: {kline_path}")
        return


if __name__ == "__main__":
    # Windows GBK 编码兼容: emoji/unicode 字符会导致 'gbk' codec can't encode character 错误
    if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    main()
