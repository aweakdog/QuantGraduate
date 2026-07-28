"""
链主事件评分引擎 — 端到端学习版

不再硬编码任何系数。使用已训练的 XGBoost 模型（含链主特征）直接评分。
对于链主事件，识别相关 A 股供应商 → 用模型预测评分 → 输出排序。

用法:
  python chain_leader_scorer.py --event "英伟达宣布投资康宁32亿美元" --leader "Corning"
  python chain_leader_scorer.py --date 2026-06-29  # 扫描当天新闻并评分
"""
import json
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import settings
from pipeline.logger import get_logger
from pipeline.feature_engine import KLINE_COL_MAP

log = get_logger("scorer")

DATA_DIR = str(settings.DATA_DIR)

# 优先使用统一后的合并数据，fallback 到旧的 supply_chain_map
_SUPPLY_CHAIN_PATH = str(settings.CHAIN_MAP_MERGED_PATH)
if not os.path.exists(_SUPPLY_CHAIN_PATH):
    _SUPPLY_CHAIN_PATH = str(settings.SUPPLY_CHAIN_PATH)

SUPPLY_CHAIN_PATH = _SUPPLY_CHAIN_PATH
MODEL_PATH = str(settings.DATA_DIR / "processed" / "model" / "xgb_model.pkl")
FEATURES_DIR = str(settings.DATA_DIR / "processed" / "features")

# ─── 模型加载 ────────────────────────────────────────────────

_model = None
_feature_names = None

def _load_model():
    """懒加载 XGBoost 模型"""
    global _model, _feature_names
    if _model is not None:
        return _model, _feature_names
    if not os.path.exists(MODEL_PATH):
        log.warning("模型不存在: %s", MODEL_PATH)
        return None, None
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    _feature_names = list(_model.feature_names_in_)
    log.info("模型加载: %s, %d 个特征", MODEL_PATH, len(_feature_names))
    return _model, _feature_names


# ─── 供应商识别 ──────────────────────────────────────────────

def find_suppliers(leader_name: str) -> list[dict]:
    """从统一供需链图谱中找到指定链主的所有 A 股供应商

    返回字段含 binding (绑定度 0-10) 用于事件感知评分。
    """
    if not os.path.exists(SUPPLY_CHAIN_PATH):
        log.warning("供需链文件不存在: %s", SUPPLY_CHAIN_PATH)
        return []
    with open(SUPPLY_CHAIN_PATH, encoding="utf-8") as f:
        sc = json.load(f)

    seen = set()
    suppliers = []
    for chain in sc.get("chains", []):
        cl = chain.get("chain_leader", {})
        if cl.get("name", "").lower() != leader_name.lower():
            continue
        for link in chain.get("demand_links", []):
            for s in link.get("a_share_suppliers", []):
                code = s.get("code", "")
                if code in seen:
                    continue
                seen.add(code)
                suppliers.append({
                    "code": code,
                    "name": s.get("name", ""),
                    "role": s.get("role", ""),
                    "exposure": s.get("exposure", ""),
                    "component": link.get("component", ""),
                    "theme": chain.get("theme", ""),
                    "direction": link.get("direction", ""),
                    "elasticity": link.get("elasticity", ""),
                    "binding": s.get("scoring", {}).get("binding", 5.0),
                })
    return suppliers


# ─── 评分 ────────────────────────────────────────────────────

def score_suppliers(suppliers: list[dict], leader_name: str, event_text: str = "") -> list[dict]:
    """
    事件感知评分：XGBoost 模型预测 × 事件强度 × 绑定度的后调融合。

    模型输出 (0-10): 股票基本面/技术面的端到端学习结果（49 特征）
    事件绑定分 (0-10): 事件强度 × 供应商绑定度的手工融合
      权重: event_intensity 0.55 + binding 0.45
    最终分: model_score × 0.6 + event_binding_score × 0.4

    这样同一只股票在不同事件下得到不同评分。
    """
    model, feature_names = _load_model()
    if model is None:
        log.error("模型未加载，无法评分")
        return []
    if not suppliers:
        return []

    # 1) 事件强度分类
    event_intensity, event_tag, _ = classify_event(event_text) if event_text else (5.0, "未知事件", "")

    results = []
    for s in suppliers:
        code = s["code"]
        code6 = code[:6]

        # 读取最新特征
        feat_path = os.path.join(FEATURES_DIR, f"{code6}.parquet")
        if not os.path.exists(feat_path):
            log.debug("特征不存在: %s", feat_path)
            continue
        try:
            df = pd.read_parquet(feat_path).sort_values("date")
            latest = df.iloc[[-1]]
        except (ValueError, OSError, pd.errors.EmptyDataError) as e:
            log.debug("读取特征失败 %s: %s", code6, e)
            continue

        # 模型预测
        try:
            X = latest[[c for c in feature_names if c in latest.columns]] \
                .fillna(0).replace([np.inf, -np.inf], 0)
            pred = float(model.predict(X)[0])
        except (ValueError, TypeError) as e:
            log.debug("预测失败 %s: %s", code6, e)
            continue

        # 归一化模型分 0~10
        model_score = round((pred + 0.05) / 0.01, 1)
        model_score = max(0.0, min(10.0, model_score))

        # 2) 事件绑定分（事件强度 × 供应商绑定度）
        binding = s.get("binding", 5.0)
        event_binding_score = (event_intensity / 10 * 0.55 + binding / 10 * 0.45) * 10
        event_binding_score = max(0.0, min(10.0, event_binding_score))

        # 3) 融合：模型 60% + 事件绑定 40%
        # 无事件时 (intensity=5.0, binding=5.0) → event_binding_score=5.0 → 接近原模型分
        final_score = round(model_score * 0.6 + event_binding_score * 0.4, 1)

        results.append({
            "name": s["name"],
            "code": code,
            "score": final_score,
            "model_score": model_score,
            "event_binding_score": round(event_binding_score, 1),
            "event_intensity": event_intensity,
            "event_tag": event_tag,
            "binding": binding,
            "raw_pred": round(float(pred), 6),
            "role": s["role"],
            "exposure": s["exposure"],
            "theme": s["theme"],
            "component": s["component"],
        })

    # 按融合评分排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ─── 事件强度分类（兼容 corning_backtest.py 的回测流程）─────────
# 生产评分已使用 XGBoost 端到端学习。以下仅作为回测兼容层。

EVENT_INTENSITY_RULES = [
    ("b1", {"patterns": ["超预期", "beat", "上调指引", "营收增长",
                          "净利增长", "净利润增长", "同比+"],
             "score": 7.0, "tag": "财报超预期"}),
    ("b2", {"patterns": ["投资", "注资", "invest", "认购"], "score": 9.5,
             "tag": "巨头投资锁产能"}),
    ("b3", {"patterns": ["亿美元", "亿美金", "billion", "数十亿", "百亿"],
             "score": 9.0, "tag": "百亿级大单"}),
    ("b4", {"patterns": ["大单", "长期合约", "长约", "长期供货"],
             "score": 8.0, "tag": "长期订单锁定"}),
    ("b5", {"patterns": ["扩产", "产能扩张", "新厂", "扩建", "新产线"],
             "score": 6.0, "tag": "产能扩张"}),
    ("b6", {"patterns": ["新品", "新技术", "突破", "发布", "launch",
                          "glass bridge", "新一代"],
             "score": 5.5, "tag": "新品/技术发布"}),
    ("b7", {"patterns": ["合作", "战略合作", "备忘录", "MOU", "签约"],
             "score": 5.0, "tag": "战略合作"}),
    ("b8", {"patterns": ["高管", "总裁", "董事长", "CEO", "表示", "认为"],
             "score": 3.0, "tag": "高管言论"}),
]
_DEFAULT_EVENT_SCORE = 4.0
_DEFAULT_EVENT_TAG = "一般消息"


def classify_event(event_text: str) -> tuple[float, str, str]:
    """对事件文本分类，返回 (分数, 标签, 匹配关键词)。仅回测兼容。"""
    text_lower = event_text.lower()
    for _name, rule in EVENT_INTENSITY_RULES:
        for p in rule["patterns"]:
            if p.lower() in text_lower:
                return rule["score"], rule["tag"], p
    return _DEFAULT_EVENT_SCORE, _DEFAULT_EVENT_TAG, "unknown"


def score_event_for_leader(
    event_text: str,
    leader_name: str,
    event_date: str | None = None,
    mode: str = "live",
) -> dict[str, Any]:
    """
    兼容层: 供 corning_backtest.py 使用。返回格式与旧版一致。
    mode/event_date 为保留参数（XGBoost 模型已含时间特征）。
    """
    intensity, tag, kw = classify_event(event_text)
    suppliers = find_suppliers(leader_name)
    scored = score_suppliers(suppliers, leader_name, event_text)
    return {
        "event": {"text": event_text, "intensity": intensity, "tag": tag, "date": event_date or ""},
        "leader": leader_name,
        "scored_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "suppliers": scored or [],
    }


# ─── 输出 ────────────────────────────────────────────────────

def update_format_result(result: dict) -> str:
    lines = []
    lines.append("[评分报告]")
    lines.append("=" * 50)
    lines.append("链主: %s" % result["leader"])
    if result.get("event"):
        ev = result["event"]
        if isinstance(ev, dict):
            lines.append("事件: %s (强度 %.1f, %s)" % (ev.get("text", ""), ev.get("intensity", 0), ev.get("tag", "")))
        else:
            lines.append("事件: %s" % ev)
    lines.append("评分方式: XGBoost 模型(60%%) + 事件绑定融合(40%%)")
    lines.append("")

    if not result["suppliers"]:
        lines.append("[无] 无可用数据（特征缺失或模型未加载）")
        return "\n".join(lines)

    lines.append("%4s %4s %-10s %-12s %-8s %-8s %-20s" % ("排名", "评分", "代码", "名称", "模型分", "事件绑定分", "角色"))
    lines.append("-" * 80)
    for i, s in enumerate(result["suppliers"], 1):
        ms = s.get("model_score", s["score"])
        eb = s.get("event_binding_score", 0)
        lines.append("%4d %4.1f %-10s %-12s %-8.1f %-8.1f %-20s" % (
            i, s["score"], s["code"][:8], s["name"][:12], ms, eb, s.get("role", "")[:20]))
    return "\n".join(lines)


# Alias for backward compatibility
format_result = update_format_result


# ─── 主入口 ──────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print("用法:")
        print("  python chain_leader_scorer.py --leader \"链主名\" [--event \"事件描述\"]")
        print("  python chain_leader_scorer.py --example")
        print()
        print("演示:")
        suppliers = find_suppliers("Corning")
        result = {
            "leader": "Corning",
            "event": "英伟达投资康宁32亿美元（示例）",
            "suppliers": score_suppliers(suppliers, "Corning"),
        }
        print(format_result(result))
        return

    if "--example" in args:
        for leader in ["NVIDIA", "Tesla", "Corning"]:
            suppliers = find_suppliers(leader)
            result = {
                "leader": leader,
                "event": "(批量演示)",
                "suppliers": score_suppliers(suppliers, leader),
            }
            print(format_result(result))
            print()
        return

    if "--leader" in args:
        idx = args.index("--leader")
        leader = args[idx + 1]
        event_text = ""
        if "--event" in args:
            eidx = args.index("--event")
            event_text = args[eidx + 1]
        suppliers = find_suppliers(leader)
        result = {
            "leader": leader,
            "event": event_text,
            "suppliers": score_suppliers(suppliers, leader),
        }
        print(format_result(result))
        return


if __name__ == "__main__":
    main()
