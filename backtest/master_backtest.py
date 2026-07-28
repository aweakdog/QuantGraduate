"""
master_backtest.py — 多链主事件驱动策略综合回测

核心: 对所有链主的历史重大事件，验证：
  1. 链主事件后，A股供应商 vs 基准 的收益
  2. 评分体系是否能区分优质/劣质机会
  3. 不同类型事件、不同链主的弹性差异
  4. 链主本身股价在事件后的反应 (验证事件有效性)
  5. 链主交叉关系传导效果 (如NVDA事件→康宁供应商也涨)
  6. 供应商自身事件修正 (第五维, wencai查询利空/利好信号)

数据:
  A股: data/raw/kline/*.parquet
  美股: westock-data (腾讯自选股)

用法:
  python -m backtest.master_backtest
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# 添加根目录
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.config import settings
from pipeline.logger import get_logger as _gl
from pipeline.westock_data import WestockClient
from pipeline.supplier_self_event import SelfEventChecker

log = _gl("master_backtest")

# ─── 路径 ───

KLINE_DIR = str(settings.KLINE_DIR)
UNIVERSE_PATH = str(settings.CHAIN_LEADER_UNIVERSE_PATH)
OUT_DIR = str(settings.BACKTEST_DIR)
CACHE_DIR = os.path.join(KLINE_DIR, "..", "westock_cache")

# ─── 链主事件数据库 ───
# 格式: (日期, 事件文本, 事件类型, 链主ID, 链主美股代码)
# 事件类型: 巨头投资/财报超预期/长期订单/产能扩张/技术发布/战略合作/高管言论/小作文

MASTER_EVENTS = [
    # ═══════════════════════════════════════════════════════════════
    # Corning (GLW) 事件 — 已手工整理 19 个，经过回测验证
    # ═══════════════════════════════════════════════════════════════
    ("2026-06-26", "康宁发布Glass Bridge玻璃光互连新技术", "技术发布", "corning", "usGLW"),
    ("2026-06-25", "康宁Glass Bridge光互连技术报道", "技术发布", "corning", "usGLW"),
    ("2026-06-09", "亚马逊与康宁签署数十亿美元光纤长约", "长期订单", "corning", "usGLW"),
    ("2026-05-20", "京东方与康宁签署合作备忘录", "战略合作", "corning", "usGLW"),
    ("2026-05-07", "英伟达宣布投资康宁32亿美元，锁定光连接产能", "巨头投资", "corning", "usGLW"),
    ("2026-05-05", "英伟达与康宁达成长期合作，光通信概念股全线走高", "巨头投资", "corning", "usGLW"),
    ("2026-04-29", "康宁2026Q1财报：光通信营收18亿美元同比+36%，净利+93%", "财报超预期", "corning", "usGLW"),
    ("2026-03-15", "康宁CEO带队访问中国长飞/亨通/中天，洽谈代工合作", "高管言论", "corning", "usGLW"),
    ("2026-01-15", "Meta与康宁签署超60亿美元光纤长期供货协议", "长期订单", "corning", "usGLW"),
    ("2025-12-10", "康宁宣布光纤产能扩张计划，应对AI数据中心需求", "产能扩张", "corning", "usGLW"),
    ("2025-10-29", "康宁2025Q3财报：光通信营收增长强劲，上调全年指引", "财报超预期", "corning", "usGLW"),
    ("2025-07-30", "康宁2025Q2财报：光通信业务创新高", "财报超预期", "corning", "usGLW"),
    ("2025-05-15", "康宁宣布与多家云厂商洽谈光纤供应合作", "高管言论", "corning", "usGLW"),
    ("2025-04-28", "康宁2025Q1财报：光通信营收同比+28%", "财报超预期", "corning", "usGLW"),
    ("2025-01-30", "康宁2024Q4财报：全年光通信营收创纪录", "财报超预期", "corning", "usGLW"),
    ("2024-10-30", "康宁2024Q3财报：光通信需求回暖", "财报超预期", "corning", "usGLW"),
    ("2024-07-31", "康宁2024Q2财报：显示玻璃业务企稳", "财报", "corning", "usGLW"),
    ("2024-04-30", "康宁2024Q1财报：光通信业务复苏迹象", "财报", "corning", "usGLW"),
    ("2024-01-31", "康宁2023Q4财报：全年业绩符合预期", "财报", "corning", "usGLW"),

    # ═══════════════════════════════════════════════════════════════
    # NVIDIA (NVDA) — 财报事件(AI boom期间)
    # ═══════════════════════════════════════════════════════════════
    ("2024-02-21", "英伟达FY24Q4财报：营收$22.1B超预期，AI需求爆发", "财报超预期", "nvidia", "usNVDA"),
    ("2024-05-22", "英伟达FY25Q1财报：营收$26B同比+262%，数据中心创纪录", "财报超预期", "nvidia", "usNVDA"),
    ("2024-08-28", "英伟达FY25Q2财报：营收$30B持续超预期，Blackwell需求强劲", "财报超预期", "nvidia", "usNVDA"),
    ("2024-11-20", "英伟达FY25Q3财报：营收$35B，Blackwell出货在即", "财报超预期", "nvidia", "usNVDA"),
    ("2025-02-26", "英伟达FY25Q4财报：营收$39B，Blackwell贡献超预期", "财报超预期", "nvidia", "usNVDA"),
    ("2025-05-28", "英伟达FY26Q1财报：营收$42B，数据中心翻倍增长", "财报超预期", "nvidia", "usNVDA"),
    ("2025-08-27", "英伟达FY26Q2财报：营收$45B，GB300需求强劲", "财报超预期", "nvidia", "usNVDA"),
    ("2025-11-19", "英伟达FY26Q3财报：营收$48B，Rubin架构进展超预期", "财报超预期", "nvidia", "usNVDA"),
    ("2026-02-25", "英伟达FY26Q4财报：营收$50B，AI推理需求爆发", "财报超预期", "nvidia", "usNVDA"),
    ("2026-05-27", "英伟达FY27Q1财报：营收$52B，持续超预期", "财报超预期", "nvidia", "usNVDA"),

    # NVIDIA — 产品/GTC事件
    ("2024-03-18", "GTC 2024：英伟达发布Blackwell B200 GPU，AI算力再翻倍", "技术发布", "nvidia", "usNVDA"),
    ("2025-03-17", "GTC 2025：英伟达发布Rubin架构，2026年量产", "技术发布", "nvidia", "usNVDA"),
    ("2026-03-16", "GTC 2026：英伟达发布下一代AI芯片，物理AI平台", "技术发布", "nvidia", "usNVDA"),

    # NVIDIA — 事件(非财报)
    ("2025-01-07", "CES 2025：英伟达发布Project Digits个人AI超级计算机", "技术发布", "nvidia", "usNVDA"),
    ("2026-05-07", "英伟达宣布投资康宁32亿美元，锁定光连接产能", "巨头投资", "nvidia", "usNVDA"),
    ("2025-04-15", "英伟达宣布CoWoS产能翻倍，台积电扩产加速", "产能扩张", "nvidia", "usNVDA"),

    # ═══════════════════════════════════════════════════════════════
    # Tesla (TSLA) — Optimus 机器人 + 财报
    # ═══════════════════════════════════════════════════════════════
    ("2024-01-24", "特斯拉FY23Q4财报：提及Optimus人形机器人进展", "财报超预期", "tesla", "usTSLA"),
    ("2024-06-13", "特斯拉2024股东大会：马斯克称Optimus将超电动车价值", "高管言论", "tesla", "usTSLA"),
    ("2024-10-10", "特斯拉We Robot活动：Optimus Gen3现场展示", "技术发布", "tesla", "usTSLA"),
    ("2025-01-29", "特斯拉FY24Q4财报：Optimus 2025年内产线试产", "财报超预期", "tesla", "usTSLA"),
    ("2025-06-12", "特斯拉2025股东大会：Optimus年产5万台产线建设中", "产能扩张", "tesla", "usTSLA"),
    ("2025-08-20", "特斯拉发布Optimus Gen4，灵巧手能力大幅提升", "技术发布", "tesla", "usTSLA"),
    ("2026-01-28", "特斯拉FY25Q4财报：Optimus开始首批内部部署", "财报超预期", "tesla", "usTSLA"),
    ("2026-04-15", "特斯拉Optimus量产线投产，年产目标10万台", "产能扩张", "tesla", "usTSLA"),

    # ═══════════════════════════════════════════════════════════════
    # Apple (AAPL)
    # ═══════════════════════════════════════════════════════════════
    ("2024-06-10", "WWDC 2024：苹果发布Apple Intelligence AI战略", "技术发布", "apple", "usAAPL"),
    ("2024-09-09", "苹果iPhone 16发布：AI功能集成，换机周期启动", "技术发布", "apple", "usAAPL"),
    ("2025-01-30", "苹果FY25Q1财报：AI手机带动换机潮，服务收入新高", "财报超预期", "apple", "usAAPL"),
    ("2025-09-09", "苹果iPhone 17发布：折叠屏+AI深度融合", "技术发布", "apple", "usAAPL"),
    ("2026-01-30", "苹果FY26Q1财报：AI手机出货量超预期", "财报超预期", "apple", "usAAPL"),

    # ═══════════════════════════════════════════════════════════════
    # TSMC (台积电)
    # ═══════════════════════════════════════════════════════════════
    ("2024-01-18", "台积电FY23Q4财报：AI芯片需求爆发，CoWoS产能翻倍", "财报超预期", "tsmc", "usTSM"),
    ("2024-04-18", "台积电FY24Q1财报：3nm营收大幅增长，AI占比提升", "财报超预期", "tsmc", "usTSM"),
    ("2024-07-18", "台积电FY24Q2财报：CoWoS产能供不应求，上调资本开支", "财报超预期", "tsmc", "usTSM"),
    ("2024-10-17", "台积电FY24Q3财报：AI芯片营收占比超30%", "财报超预期", "tsmc", "usTSM"),
    ("2025-01-16", "台积电FY24Q4财报：全年AI相关营收翻倍", "财报超预期", "tsmc", "usTSM"),
    ("2025-04-17", "台积电FY25Q1财报：2nm良率超预期，2026年量产", "财报超预期", "tsmc", "usTSM"),
    ("2026-01-15", "台积电FY25Q4财报：AI营收占比超50%", "财报超预期", "tsmc", "usTSM"),
    ("2026-04-16", "台积电FY26Q1财报：2nm量产在即", "财报超预期", "tsmc", "usTSM"),
]

# ─── 供应商列表（从 universe 文件中提取）───────────────────────
# 只包含有K线数据的，避免无效查询

def load_suppliers_for_leader(leader_id: str) -> dict:
    """从 universe 文件中提取指定链主的所有供应商"""
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    leader = None
    for l in universe["chain_leaders"]:
        if l["id"] == leader_id:
            leader = l
            break

    if not leader or "a_share_chain" not in leader:
        return {}

    suppliers = {}
    for chain in leader["a_share_chain"]:
        for s in chain.get("stocks", []):
            code = s["code"]
            code6 = code.split(".")[0]
            if code6 not in suppliers:
                # binding 评分简略版
                ev = s.get("evidence", {})
                level = ev.get("level", 4)
                binding_map = {1: 9.0, 2: 7.0, 3: 5.0, 4: 3.0}
                exposure = s.get("exposure", "中")
                binding = binding_map.get(level, 3.0)
                if exposure == "核心":
                    binding = min(binding + 2.0, 10)
                elif exposure == "高":
                    binding = min(binding + 1.0, 10)

                suppliers[code6] = {
                    "name": s["name"],
                    "code": code,
                    "role": s.get("role", ""),
                    "exposure": s.get("exposure", ""),
                    "binding": binding,
                    "evidence_level": level,
                }
    return suppliers


# ─── K线加载 ─────────────────────────────────────────────────

def load_akline(code6: str) -> pd.DataFrame | None:
    """从本地 parquet 加载A股K线"""
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


def find_trading_day(df: pd.DataFrame, target_dt, offset: int = 0):
    """找 target_dt 之后第 offset 个交易日"""
    if isinstance(target_dt, str):
        target_dt = pd.to_datetime(target_dt)
    mask = df["date"] >= target_dt
    available = df.loc[mask, "date"].values
    if len(available) <= offset:
        return None
    return available[offset]


def calc_return(df, entry_date, hold_days: int) -> float | None:
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
    return float((xp[0] - ep[0]) / ep[0] * 100)


# ─── 事件评分（轻量版，不依赖 scorer 模块的 supply_chain_map.json）─

EVENT_INTENSITY = [
    ("巨头投资", ["投资", "注资", "invest", "亿美元投资", "认购"], 9.5),
    ("百亿大单", ["亿美元", "billion", "数十亿", "百亿"], 9.0),
    ("长期订单", ["大单", "长期合约", "长约", "multi-year", "长期供货"], 8.0),
    ("财报超预期", ["超预期", "beat", "上调指引", "营收增长", "净利增长", "同比增长"], 7.0),
    ("产能扩张", ["扩产", "产能扩张", "新厂", "扩建"], 6.0),
    ("技术发布", ["新品", "新技术", "突破", "发布", "launch", "新一代"], 5.5),
    ("战略合作", ["合作", "战略合作", "备忘录", "MOU", "签约"], 5.0),
    ("研报上调", ["研报", "上调", "买入评级", "增持"], 3.5),
    ("高管言论", ["高管", "总裁", "董事长", "CEO", "表示", "认为"], 3.0),
    ("小作文", ["传闻", "消息称", "据传", "rumor"], 2.0),
]


def score_event(event_text: str, event_type: str) -> tuple[float, str]:
    """对事件评分，返回 (intensity, tag)"""
    text_lower = event_text.lower()

    # 先查规则库
    for tag, patterns, score in EVENT_INTENSITY:
        for p in patterns:
            if p.lower() in text_lower:
                return score, tag

    # 兜底：按事件类型
    type_defaults = {
        "财报超预期": 7.0, "巨头投资": 9.0, "长期订单": 8.0,
        "技术发布": 5.5, "战略合作": 5.0, "产能扩张": 6.0,
        "高管言论": 3.0, "财报": 5.0,
    }
    return (type_defaults.get(event_type, 4.0), event_type)


def calc_supplier_score(event_intensity: float, supplier: dict, self_mult: float = 1.0) -> float:
    """计算单个供应商的综合评分（v2.0 五维公式）"""
    binding = supplier["binding"]
    # 基础分 = (事件强度×0.35 + 绑定度×0.25) × 各乘数
    raw = (event_intensity * 0.35 + binding * 0.25) * 1.0 * 1.0 * self_mult
    if supplier["exposure"] == "中":
        raw *= 0.85
    return round(min(raw, 10.0), 1)


def get_recommendation(score: float) -> str:
    if score >= 7.0:
        return "买入"
    elif score >= 5.0:
        return "关注"
    elif score >= 3.0:
        return "观望"
    return "不参与"


# ─── 回测主逻辑 ────────────────────────────────────────────────

def run_master_backtest():
    log.info("=" * 80)
    log.info("  链主事件驱动策略 — 多链主综合回测")
    log.info(f"  {'=' * 76}")
    log.info(f"  事件总数: {len(MASTER_EVENTS)} 次")
    log.info(f"  时间跨度: {MASTER_EVENTS[-1][0]} ~ {MASTER_EVENTS[0][0]}")

    # 加载供应商
    all_suppliers = {}
    for leader_id in set(e[3] for e in MASTER_EVENTS):
        suppliers = load_suppliers_for_leader(leader_id)
        if suppliers:
            all_suppliers[leader_id] = suppliers
    # 加载链主名称
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)
    leader_name_map = {l["id"]: l["name"] for l in universe["chain_leaders"]}

    log.info(f"  链主覆盖: {', '.join(leader_name_map.get(e[3], e[3]) for e in MASTER_EVENTS if e[3] in all_suppliers)}")
    total_suppliers = sum(len(s) for s in all_suppliers.values())
    log.info(f"  供应商覆盖: {total_suppliers} 只")
    log.info("=" * 80)

    # 预加载A股K线
    log.info("\n加载A股K线数据...")
    akline_cache = {}
    all_used_codes = set()
    for leader_id, suppliers in all_suppliers.items():
        for code6 in suppliers:
            all_used_codes.add(code6)
    loaded_ak = 0
    for code6 in sorted(all_used_codes):
        df = load_akline(code6)
        if df is not None:
            akline_cache[code6] = df
            loaded_ak += 1
    log.info(f"  已加载 {loaded_ak}/{len(all_used_codes)} 只")

    # 初始化自身事件检查器（用于 backtest_proxy 价格代理模式）
    self_event_checker = SelfEventChecker()
    log.info("SelfEventChecker 已初始化")

    # 初始化westock客户端
    log.info("\n初始化美股数据源...")
    wc = WestockClient(cache=True)

    # 加载美股链主K线
    us_leaders_needed = list(set(e[4] for e in MASTER_EVENTS))
    us_kline_cache = {}
    for us_code in us_leaders_needed:
        # 尽量多拉数据
        df = wc.kline(us_code, limit=800)
        if df is not None:
            us_kline_cache[us_code] = df
            log.info(f"  {us_code}: {len(df)} 行, {df['date'].min().date()} ~ {df['date'].max().date()}")
        else:
            log.info(f"  {us_code}: 无数据")

    # ─── 回测主循环 ───
    WINDOWS = [1, 3, 5, 10]

    all_results = {}

    for hold in WINDOWS:
        log.info(f"\n{'═' * 80}")
        log.info(f"  持有 {hold} 个交易日")
        log.info(f"{'═' * 80}")

        event_log = []

        for date_str, event_text, ev_type, leader_id, us_code in MASTER_EVENTS:
            # 事件评分
            intensity, tag = score_event(event_text, ev_type)

            # 获取该链主的供应商
            suppliers = all_suppliers.get(leader_id, {})

            # 计算每个供应商的回报
            stock_returns = []
            self_event_stats = {"修正次数": 0, "平均系数": 1.0, "<1": 0, ">1": 0}
            for code6, info in suppliers.items():
                df = akline_cache.get(code6)
                if df is None:
                    continue
                ret = calc_return(df, date_str, hold)
                if ret is not None:
                    # 自身事件修正（backtest_proxy 价格代理模式）
                    proxy = self_event_checker.backtest_proxy(df, date_str)
                    self_mult = proxy["self_event_multiplier"]
                    if self_mult != 1.0:
                        self_event_stats["修正次数"] += 1
                        self_event_stats["平均系数"] += self_mult * 0.001
                        if self_mult < 1.0:
                            self_event_stats["<1"] += 1
                        else:
                            self_event_stats[">1"] += 1

                    score = calc_supplier_score(intensity, info, self_mult)
                    stock_returns.append({
                        "code": code6,
                        "name": info["name"],
                        "return": ret,
                        "score": score,
                        "binding": info["binding"],
                        "exposure": info["exposure"],
                        "role": info["role"],
                        "self_mult": self_mult,
                        "self_signal": proxy.get("proxy_signal", ""),
                    })

            if not stock_returns:
                continue

            # 链主自身回报（美股）
            leader_return = None
            leader_df = us_kline_cache.get(us_code)
            if leader_df is not None:
                leader_return = calc_return(leader_df, date_str, hold)

            # 汇总
            returns = [r["return"] for r in stock_returns]
            avg_ret = np.mean(returns)
            med_ret = np.median(returns)
            positive = sum(1 for r in returns if r > 0)
            win_rate = positive / len(returns) * 100

            # 按评分分组
            high_score = [r for r in stock_returns if r["score"] >= 6.0]
            low_score = [r for r in stock_returns if r["score"] < 6.0]
            high_avg = np.mean([r["return"] for r in high_score]) if high_score else None
            low_avg = np.mean([r["return"] for r in low_score]) if low_score else None

            # 按绑定度分组
            high_bind = [r for r in stock_returns if r["binding"] >= 7.0]
            high_bind_avg = np.mean([r["return"] for r in high_bind]) if high_bind else None

            # 最佳/最差
            best = max(stock_returns, key=lambda r: r["return"])
            worst = min(stock_returns, key=lambda r: r["return"])

            leader_name = leader_name_map.get(leader_id, leader_id)

            event_log.append({
                "date": date_str,
                "leader": leader_name,
                "event": event_text[:60],
                "type": ev_type,
                "tag": tag,
                "intensity": intensity,
                "avg_return": round(avg_ret, 2),
                "median_return": round(med_ret, 2),
                "win_rate": round(win_rate, 1),
                "positive": positive,
                "total": len(stock_returns),
                "high_score_avg": round(high_avg, 2) if high_avg is not None else None,
                "low_score_avg": round(low_avg, 2) if low_avg is not None else None,
                "high_bind_avg": round(high_bind_avg, 2) if high_bind_avg is not None else None,
                "leader_return": round(leader_return, 2) if leader_return is not None else None,
                "best": f"{best['name']}({best['return']:+.2f}%)",
                "worst": f"{worst['name']}({worst['return']:+.2f}%)",
                "self_event_corrected": self_event_stats["修正次数"],
                "self_event_avg_mult": round(self_event_stats["平均系数"] /
                    max(self_event_stats["修正次数"], 1), 3),
                "self_event_positive": self_event_stats[">1"],
                "self_event_negative": self_event_stats["<1"],
            })

            # 打印
            bar = "█" * max(min(int(abs(avg_ret) * 2), 50), 0) if abs(avg_ret) < 30 else "█" * 50
            hs = f" 高{high_avg:+.1f}" if high_avg is not None else ""
            ls = f"低{low_avg:+.1f}" if low_avg is not None else ""
            lr = f"链主{leader_return:+.1f}%" if leader_return is not None else ""
            log.info(f"  {date_str} {leader_name:>8s} {tag:>6s} {avg_ret:>+6.2f}% {bar} ({win_rate:.0f}%) {hs} {ls} {lr}")

        # ─── 汇总统计 ───
        df_events = pd.DataFrame(event_log)
        if len(df_events) == 0:
            log.info(f"\n  无有效事件 (持有{hold}天)")
            continue

        # 整体
        avg_all = df_events["avg_return"].mean()
        med_all = df_events["avg_return"].median()
        win_all = (df_events["avg_return"] > 0).mean() * 100
        total_ret = (1 + df_events["avg_return"] / 100).prod() - 1
        ret_series = df_events["avg_return"].values
        sharpe = np.mean(ret_series) / np.std(ret_series) * np.sqrt(252 / hold) if np.std(ret_series) > 0 else 0

        # 评分区分度
        hs_vals = df_events["high_score_avg"].dropna().values
        ls_vals = df_events["low_score_avg"].dropna().values
        spread = (hs_vals.mean() - ls_vals.mean()) if len(hs_vals) > 0 and len(ls_vals) > 0 else None

        # 链主自身平均回报
        leader_rets = df_events["leader_return"].dropna().values
        leader_avg = leader_rets.mean() if len(leader_rets) > 0 else None

        log.info(f"\n  {'─' * 78}")
        log.info(f"  [汇总] 持有{hold}天 | {len(df_events)}次事件")
        log.info(f"    平均收益:   {avg_all:+.2f}%  |  中位数: {med_all:+.2f}%  |  累计总收益: {total_ret*100:+.1f}%")
        log.info(f"    胜率:       {win_all:.0f}%  |  夏普: {sharpe:.3f}")
        if spread is not None:
            log.info(f"    评分区分度: 高≥6分 {hs_vals.mean():+.2f}% vs 低<6分 {ls_vals.mean():+.2f}% (差{spread:+.2f}%)")
        if leader_avg is not None:
            log.info(f"    链主自身:   {leader_avg:+.2f}%")

        # 按链主分组
        log.info(f"\n  [按链主分组]")
        leader_groups = df_events.groupby("leader")
        for ldr, sub in sorted(leader_groups, key=lambda x: x[1]["avg_return"].mean(), reverse=True):
            lr = sub["leader_return"].mean()
            lr_str = f"链主{lr:+.1f}%" if not pd.isna(lr) else ""
            log.info(f"    {ldr:>10s}: {len(sub):2d}次, {sub['avg_return'].mean():+6.2f}% (胜率{sub['win_rate'].mean():.0f}%) {lr_str}")

        # 按事件类型分组
        log.info(f"\n  [按事件类型]")
        type_groups = df_events.groupby("type")
        for tp, sub in sorted(type_groups, key=lambda x: x[1]["avg_return"].mean(), reverse=True):
            log.info(f"    {tp:>8s}: {len(sub):2d}次, {sub['avg_return'].mean():+6.2f}% (胜率{sub['win_rate'].mean():.0f}%)")

        # 按评分段分组（个股层面 - 合并所有事件）
        all_stocks = []
        for _, row in df_events.iterrows():
            # 重新生成个股级数据
            date_str = row["date"]
            intensity = row["intensity"]
            leader_id_by_date = None
            for d, _, _, lid, _ in MASTER_EVENTS:
                if d == date_str:
                    leader_id_by_date = lid
                    break
            if leader_id_by_date is None:
                continue
            suppliers = all_suppliers.get(leader_id_by_date, {})
            for code6, info in suppliers.items():
                df = akline_cache.get(code6)
                if df is None:
                    continue
                ret = calc_return(df, date_str, hold)
                if ret is not None:
                    score = calc_supplier_score(intensity, info)
                    all_stocks.append({
                        "score": score,
                        "return": ret,
                        "name": info["name"],
                        "code": code6,
                    })
        if all_stocks:
            df_stocks = pd.DataFrame(all_stocks)
            df_stocks["score_bin"] = pd.cut(df_stocks["score"], bins=[0, 3, 5, 7, 10],
                                             labels=["<3.0", "3-5", "5-7", "7-10"])
            log.info(f"\n  [按个股评分分段]")
            bins_summary = df_stocks.groupby("score_bin", observed=True).agg(
                平均收益=("return", "mean"), 个股次数=("return", "count"), 胜率=("return", lambda x: (x > 0).mean() * 100)
            ).round(2)
            for bin_label, row in bins_summary.iterrows():
                bar2 = "█" * max(min(int(abs(row["平均收益"]) * 3), 40), 0)
                log.info(f"    {bin_label}: {row['平均收益']:+6.2f}% {bar2} ({row['个股次数']}次, 胜率{row['胜率']:.0f}%)")

        # 保存
        all_results[hold] = {
            "hold_days": hold,
            "total_events": len(event_log),
            "avg_return_per_event": round(avg_all, 2),
            "median_return_per_event": round(med_all, 2),
            "accumulated_return": round(total_ret * 100, 2),
            "win_rate": round(win_all, 1),
            "sharpe": round(sharpe, 3),
            "high_score_avg": round(hs_vals.mean(), 2) if len(hs_vals) > 0 else None,
            "low_score_avg": round(ls_vals.mean(), 2) if len(ls_vals) > 0 else None,
            "score_spread": round(spread, 2) if spread is not None else None,
            "leader_avg_return": round(leader_avg, 2) if leader_avg is not None else None,
            "details": event_log,
        }

    return all_results


# ─── 保存 ────────────────────────────────────────────────────

def save_results(results: dict):
    os.makedirs(OUT_DIR, exist_ok=True)

    # JSON
    path = os.path.join(OUT_DIR, "master_backtest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "events_count": len(MASTER_EVENTS),
            "events_range": f"{MASTER_EVENTS[-1][0]} ~ {MASTER_EVENTS[0][0]}",
            "results": {str(k): v for k, v in results.items()},
        }, f, ensure_ascii=False, indent=2)
    log.info(f"\n  回测结果已保存: {path}")

    # CSV 摘要
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
            "链主均收益%": r.get("leader_avg_return", ""),
        })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "master_backtest.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"  CSV摘要已保存: {csv_path}")
    log.info(f"\n{'=' * 80}")
    log.info(f"  {df_out.to_string(index=False)}")
    log.info(f"{'=' * 80}")


if __name__ == "__main__":
    results = run_master_backtest()
    if results:
        save_results(results)
