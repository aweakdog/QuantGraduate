"""
特征工程管线 — 六维原始数据 → ML 特征矩阵

输入:
  - data/raw/kline/{code}.parquet               # 日K前复权
  - data/raw/fund_flow_full/fundflow_history.parquet  # 资金流(新: 5列, 日频)
  - data/raw/MainNetFlow/margintrade_history.parquet  # 融资融券(新: 15列, 日频)
  - data/raw/announcements/{code}.parquet       # 公告
  - data/universe/stock_list.parquet             # 基本面
  - data/raw/macro/*.parquet                     # 宏观

输出:
  - data/processed/features/{code}.parquet  # 特征矩阵(按股)
  - data/processed/training_data.parquet    # 训练集(全量)

变更历史:
  v4: +dde_net/mtss_balance/fund_flow 资金面日频补齐
      +融资融券特征(15列margin trade, 净买入/余额/动量)
      弃用旧 fund_flow/{code}.parquet (稀疏, 仅2列)
  v5: +P0-P4事件特征 (events_clean.parquet)
      +事件级别加权 (P0×10, P1×5, P2×2, P3×1)
      +事件方向硬编码 (event_type → bullish/bearish)
      +钟形时间衰减
  v8: -US ETF/期货特征(标普/道指/纳指/美股个股/板块ETF)
      MA窗口 2/5/21 → 3/13 → 5/20 (均线偏离/价格位置/滚动均值)
  v9: +全球指数期货(SP/DJ/NQ/SOX) 回归
      +A50期货(CN) 保留
      MA窗口 5/20
  v10: +汇率特征(USDINR/USDCNH/USDJPY) + MA3/13
      DXY从iFinD, USDCNH从iFinD, USDJPY从ECB
  v11: 汇率仅保留原始价格+MA3/13,去除变化率/z-score衍生
  v12: +国债指数(CNBOND Sina sh000012 + USBOND iFinD SCH018052)
       +MA3/13,仅保留原始指数值
  v13: 国债指数→国债收益率(CN2Y/CN5Y/US2Y/US5Y from akshare bond_zh_us_rate)
       2020起全覆盖,统一收益率口径,+MA3/13
  v14: +MA跨窗口交叉信号(已废弃,发现对T+1冗余)
  v15: MA窗口 3/13 → 5/20
  v16: 清理_ma2旧窗口残留,移除_cross交叉信号
       walk_forward: 恢复MA20纳入特征(之前误排除)
  v15: +央行流动性(PBOC_LIQ=SHIBOR O/N) + MA5/20
"""
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from pipeline.config import settings
from pipeline.logger import get_logger as _gl

log = _gl("feature_engine")
DATA_DIR = str(settings.DATA_DIR)

# ─── 新数据源路径 ─────────────────────────────────────────────
FUNDFLOW_PATH = os.path.join(DATA_DIR, "raw", "fund_flow_full", "fundflow_history.parquet")
MARGIN_PATH   = os.path.join(DATA_DIR, "raw", "MainNetFlow", "margintrade_history.parquet")
EVENTS_PATH   = os.path.join(DATA_DIR, "raw", "events_ifind", "events_v2.parquet")

# ─── 事件方向映射 (event_type → inherent direction) ──────
EVENT_DIR_MAP = {
    'big_contract': 1, 'buyback_done': 1, 'buyback_ongoing': 1, 'buyback_plan': 1,
    'dividend': 1, 'equity_incentive': 1, 'expansion': 1, 'increase': 1, 'unpledge': 1,
    'lawsuit': -1, 'pledge': -1, 'reduction': -1, 'regulatory': -1, 'regulatory_action': -1,
    'earnings_revise': 0, 'major_restructure': 0, 'regulatory_filing': 0,
    'routine_filing': 0, 'state_capital': 0,
}

# P-level 权重
EVENT_P_WEIGHT = {'P0': 10, 'P1': 5, 'P2': 2, 'P3': 1}

# ─── 资金面缓存(按code过滤用,避免反复读大文件)────────────
_fundflow_cache = None
_margin_cache = None
_events_cache = None

def _load_fundflow() -> pd.DataFrame:
    """加载 consolidated fundflow_history 并构建缓存"""
    global _fundflow_cache
    if _fundflow_cache is None:
        df = pd.read_parquet(FUNDFLOW_PATH)
        df["date"] = pd.to_datetime(df["date"])
        _fundflow_cache = df.sort_values(["code", "date"]).reset_index(drop=True)
    return _fundflow_cache

def _load_margin() -> pd.DataFrame:
    """加载 consolidated margintrade_history 并构建缓存"""
    global _margin_cache
    if _margin_cache is None:
        df = pd.read_parquet(MARGIN_PATH)
        df["date"] = pd.to_datetime(df["date"])
        _margin_cache = df.sort_values(["code", "date"]).reset_index(drop=True)
    return _margin_cache

# ─── 列名统一 ────────────────────────────────────────────────

KLINE_COL_MAP = {
    "时间": "date", "收盘价": "close", "开盘价": "open",
    "最高价": "high", "最低价": "low", "成交量": "volume",
    "总金额": "amount",
}

def read_kline(code: str) -> Optional[pd.DataFrame]:
    """读取日K, 统一列名, date转datetime

    Args:
        code: 6位股票代码

    Returns:
        DataFrame with columns [date, close, open, high, low, volume, amount]
        或 None(文件不存在时)
    """
    path = os.path.join(DATA_DIR, "raw", "kline", f"{code}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.rename(columns=KLINE_COL_MAP, inplace=True, errors='ignore')
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def _load_events() -> pd.DataFrame:
    """加载 events_clean (P0-P4分级) 并构建缓存"""
    global _events_cache
    if _events_cache is None:
        if not os.path.exists(EVENTS_PATH):
            log.warning("事件数据缺失，事件特征将为空: %s", EVENTS_PATH)
            _events_cache = pd.DataFrame(columns=["code", "date", "event_type", "p_level", "dir_hard", "p_w", "impact"])
            return _events_cache
        df = pd.read_parquet(EVENTS_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df["dir_hard"] = df["event_type"].map(EVENT_DIR_MAP).fillna(0)
        df["p_w"] = df["p_level"].map(EVENT_P_WEIGHT).fillna(1).astype(float)
        df["impact"] = df["dir_hard"] * df["p_w"]
        _events_cache = df.sort_values(["code", "date"]).reset_index(drop=True)
    return _events_cache

def read_events_for_stock(code6: str) -> Optional[pd.DataFrame]:
    """读取单个股票的事件数据 (events_clean)
    
    Args:
        code6: 6位股票代码

    Returns:
        DataFrame with [date, event_type, p_level, dir_hard, p_w, impact]
    """
    try:
        pool = _load_events()
        sub = pool[pool["code"].str.startswith(code6)].copy()
        if len(sub) == 0:
            return None
        return sub[["date", "event_type", "p_level", "dir_hard", "p_w", "impact"]]
    except Exception as e:
        log.debug("读取事件失败 %s: %s", code6, e)
        return None

def read_fund_flow(code: str) -> Optional[pd.DataFrame]:
    """从 consolidated fundflow_history 读取单只股票资金流

    支持5列: main_force_net, main_force_pct, dde_net, mtss_balance, fund_flow
    日频, 2020-01 ~ 2026-06

    Args:
        code: 6位股票代码

    Returns:
        DataFrame with [date, main_force_net, main_force_pct, dde_net, mtss_balance, fund_flow]
        或 None(文件不存在或无该股票数据时)
    """
    try:
        pool = _load_fundflow()
        sub = pool[pool["code"] == code].copy()
        if len(sub) == 0:
            return None
        return sub[["date", "main_force_net", "main_force_pct", "dde_net", "mtss_balance", "fund_flow"]].reset_index(drop=True)
    except Exception as e:
        log.debug("读取资金流失败 %s: %s", code, e)
        return None

def read_margin_trade(code: str) -> Optional[pd.DataFrame]:
    """从 consolidated margintrade_history 读取单只股票融资融券

    包含15列: rzye, rzmre, rzche, rzjme, rqye, rqmcl, rqchl, rqjmg, rzrqye, spj, ...

    Args:
        code: 6位股票代码

    Returns:
        DataFrame with [date, rzye, rzmre, rzche, rzjme, rqye, rqmcl, rqchl, rqjmg, rzrqye, spj, zdf, ...]
        或 None
    """
    try:
        pool = _load_margin()
        # margin 数据中 code 是纯6位数字
        sub = pool[pool["code"] == code].copy()
        if len(sub) == 0:
            return None
        return sub.reset_index(drop=True)
    except Exception as e:
        log.debug("读取融资融券失败 %s: %s", code, e)
        return None

def read_announcements(code: str) -> Optional[pd.DataFrame]:
    """读取公告数据

    Args:
        code: 6位股票代码

    Returns:
        DataFrame with [date, ...] 列,或 None(文件不存在或格式异常时)
    """
    path = os.path.join(DATA_DIR, "raw", "announcements", f"{code}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except (ValueError, TypeError, OSError, pd.errors.EmptyDataError) as e:
        log.warning("读取公告失败 %s: %s", code, e)
        return None

# ─── 技术面特征 ──────────────────────────────────────────────

def calc_technical_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """K线 → 技术面特征(17项)

    计算收益率, 均线偏离, 成交量变化, 波动率, 价格位置, RSI, MACD。

    Args:
        df: K线 DataFrame,需含 [close, high, low, volume] 列

    Returns:
        DataFrame 附加技术特征列,或 None(数据不足20日时)
    """
    if df is None or len(df) < 20:
        return None
    feats = df[["date"]].copy()
    c = df["close"]
    h, l, v = df["high"], df["low"], df["volume"]

    # 收益率 (保留原始多周期)
    feats["ret_1d"] = c.pct_change(1)
    feats["ret_2d"] = c.pct_change(2)
    feats["ret_5d"] = c.pct_change(5)
    feats["ret_21d"] = c.pct_change(21)

    # 均线偏离 (MA5/20)
    for w in [5, 20]:
        ma = c.rolling(w).mean()
        feats[f"ma{w}_pct"] = (c / ma - 1) * 100

    # 成交量变化
    feats["vol_ma5"] = v.rolling(5).mean()
    feats["vol_ma20"] = v.rolling(20).mean()   # v23: 补齐成交量20日MA(原漏写, 与vol_ma5对称)
    feats["vol_ratio"] = v / feats["vol_ma5"]
    feats["vol_change_1d"] = v.pct_change(1)

    # 波动率
    feats["atr"] = (h - l).rolling(14).mean()
    feats["atr_pct"] = feats["atr"] / c * 100

    # 价格位置 (MA5/20)
    for w in [5, 20]:
        hh = h.rolling(w).max()
        ll = l.rolling(w).min()
        feats[f"pos_{w}"] = (c - ll) / (hh - ll) * 100

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feats["macd"] = ema12 - ema26
    feats["macd_signal"] = feats["macd"].ewm(span=9).mean()
    feats["macd_hist"] = feats["macd"] - feats["macd_signal"]

    return feats

# ─── 资金面特征 ──────────────────────────────────────────────

def calc_fund_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """资金流 → 资金面特征(支持新旧两种列名,含Z-score归一化)

    计算主力净流入/流出多期移动平均, Z-score 跨市值归一化, 信号增强。

    Args:
        df: 资金流 DataFrame,需含 [date] 及资金流相关列

    Returns:
        资金特征 DataFrame,或 None(数据不足5日时)
    """
    if df is None or len(df) < 5:
        return None
    feats = df[["date"]].copy()

    col_map = {"main_force_net": "mf_net", "main_force_pct": "mf_pct",
               "dde_net": "dde_net", "mtss_balance": "mtss", "fund_flow": "fund_flow",
               "超大单净流入-净额": "super_large", "大单净流入-净额": "large_net"}

    for src, dst in col_map.items():
        if src in df.columns:
            s = df[src].fillna(0).astype(float)
            # 原始值
            feats[f"{dst}_1d"] = s
            # Z-score 归一化 (21日滚动)
            rolling_mean = s.rolling(21).mean()
            rolling_std = s.rolling(21).std().replace(0, np.nan)
            feats[f"{dst}_z"] = ((s - rolling_mean) / rolling_std).fillna(0)

    if "main_force_pct" in df.columns:
        pct = df["main_force_pct"].fillna(0).astype(float)
        feats["mf_signal"] = np.where(pct.abs() >= 1.0, pct * 2.5, 0)

    return feats

# ─── 融资融券特征 ─────────────────────────────────────────────

MARGIN_FEATURES = [
    ("rzye", "marg_bal"),      # 融资余额
    ("rzjme", "marg_netbuy"),  # 融资净买入
    ("rzmre", "marg_buy"),     # 融资买入
    ("rzche", "marg_repay"),   # 融资偿还
    ("rqye", "short_bal"),     # 融券余额
    ("rqjmg", "short_net"),    # 融券净卖出
    ("rqmcl", "short_vol"),    # 融券卖出量
    ("rzrqye", "marg_total"),  # 两融余额
]

def calc_margin_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """融资融券 → 杠杆+做空特征

    计算余额z-score, 净买入动量, 融资/融券比率, 综合信号。
    返回列以 marg_ / short_ 前缀。

    Args:
        df: margintrade DataFrame, 需含 [date] 及 MARGIN_FEATURES 中的列

    Returns:
        融资融券特征 DataFrame, 或 None(数据不足5日时)
    """
    if df is None or len(df) < 5:
        return None
    feats = df[["date"]].copy()
    avail = {src for src, _ in MARGIN_FEATURES if src in df.columns}
    if len(avail) < 3:
        return None

    for src, dst in MARGIN_FEATURES:
        if src not in df.columns:
            continue
        s = df[src].fillna(0).astype(float)
        feats[f"{dst}_1d"] = s
        feats[f"{dst}_ma5"] = s.rolling(5).mean()
        # z-score (21日滚动)
        r21 = s.rolling(21)
        feats[f"{dst}_z"] = ((s - r21.mean()) / r21.std().replace(0, np.nan)).fillna(0)
        # 日变动
        feats[f"{dst}_chg"] = s.diff(1).fillna(0)

    # 净买入动量
    if "rzjme" in avail:
        nj = df["rzjme"].fillna(0).astype(float)
        feats["marg_momentum_2d"] = nj.rolling(2).sum()
        feats["marg_momentum_5d"] = nj.rolling(5).sum()
        feats["marg_momentum_21d"] = nj.rolling(21).sum()

    # 融券净卖出动量
    if "rqjmg" in avail:
        sj = df["rqjmg"].fillna(0).astype(float)
        feats["short_momentum_2d"] = sj.rolling(2).sum()
        feats["short_momentum_5d"] = sj.rolling(5).sum()

    # 融资/两融 比率 (杠杆使用率)
    if "rzye" in avail and "rzrqye" in avail:
        bal = df["rzye"].fillna(0).astype(float)
        tot = df["rzrqye"].fillna(0).astype(float).replace(0, np.nan)
        feats["marg_usage_ratio"] = (bal / tot).fillna(0)
        feats["marg_usage_chg"] = feats["marg_usage_ratio"].diff(1).fillna(0)

    # 综合信号: 融资净买入↑ + 融券净卖出↓ = 看多
    if "rzjme" in avail and "rqjmg" in avail:
        nj_z = (df["rzjme"].fillna(0).astype(float) - df["rzjme"].rolling(21).mean()) / \
               df["rzjme"].rolling(21).std().replace(0, np.nan)
        sj_z = (df["rqjmg"].fillna(0).astype(float) - df["rqjmg"].rolling(21).mean()) / \
               df["rqjmg"].rolling(21).std().replace(0, np.nan)
        feats["margin_signal"] = (nj_z.fillna(0) - sj_z.fillna(0)).clip(-3, 3)

    return feats

# ─── P0-P4 事件特征(v5新增)──────────────────────────────────

def _bell_decay(day_offset: int) -> float:
    """钟形衰减: 事件发生后第3天影响力最大, 20天后趋近0"""
    return np.exp(-0.5 * ((day_offset - 3) / 4) ** 2)

def calc_event_v2_features(df_events: pd.DataFrame, kline_dates: pd.Series) -> Optional[pd.DataFrame]:
    """P0-P4 事件特征 — 级别加权 + 方向硬编码 + 钟形衰减

    生成特征:
      ev_p0_5d / ev_p1_5d / ev_p2_5d — P-level滚动计数
      ev_bull_5d / ev_bear_5d — 方向滚动计数
      ev_net_5d — 方向净得分 (P加权)
      ev_weighted_5d — P加权总事件量
      ev_decay_n_5d — 钟形衰减净信号
      以及 20d 版本

    Args:
        df_events: 事件 DataFrame (含 date, p_level, dir_hard, p_w, impact)
        kline_dates: 交易日序列

    Returns:
        事件特征 DataFrame, 或 None
    """
    if df_events is None or len(df_events) == 0:
        return None

    # 按天汇总
    daily = df_events.copy()
    daily["_date"] = daily["date"].dt.date
    grp = daily.groupby("_date").agg(
        p0_cnt=("p_level", lambda x: (x == "P0").sum()),
        p1_cnt=("p_level", lambda x: (x == "P1").sum()),
        p2_cnt=("p_level", lambda x: (x == "P2").sum()),
        bull_cnt=("dir_hard", lambda x: (x > 0).sum()),
        bear_cnt=("dir_hard", lambda x: (x < 0).sum()),
        net_score=("impact", "sum"),
    ).reset_index()
    grp.columns = ["date", "p0_cnt", "p1_cnt", "p2_cnt", "bull_cnt", "bear_cnt", "net_score"]
    grp["date"] = pd.to_datetime(grp["date"])

    # 对齐交易日
    result = kline_dates.to_frame("date").copy()
    result = result.merge(grp, on="date", how="left")
    for c in ["p0_cnt","p1_cnt","p2_cnt","bull_cnt","bear_cnt","net_score"]:
        result[c] = result[c].fillna(0).astype(float)

    # 钟形衰减
    result["decay_bull"] = result["bull_cnt"].rolling(20, min_periods=1).apply(
        lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)
    result["decay_bear"] = result["bear_cnt"].rolling(20, min_periods=1).apply(
        lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)

    # 输出特征 (统一MA5/21)
    for w in [5, 21]:
        result[f"ev_p0_{w}d"] = result["p0_cnt"].rolling(w, min_periods=1).sum()
        result[f"ev_p1_{w}d"] = result["p1_cnt"].rolling(w, min_periods=1).sum()
        result[f"ev_p2_{w}d"] = result["p2_cnt"].rolling(w, min_periods=1).sum()
        result[f"ev_bull_{w}d"] = result["bull_cnt"].rolling(w, min_periods=1).sum()
        result[f"ev_bear_{w}d"] = result["bear_cnt"].rolling(w, min_periods=1).sum()
        result[f"ev_net_{w}d"] = result["net_score"].rolling(w, min_periods=1).sum()
        result[f"ev_decay_n_{w}d"] = (result["decay_bull"].rolling(w, min_periods=1).sum() -
                                       result["decay_bear"].rolling(w, min_periods=1).sum()).fillna(0)

    keep = [c for c in result.columns if c.startswith("ev_") or c == "date"]
    return result[keep]

# ─── 板块级别事件特征(v5.1: 事件溢出效应)─────────────────────

_theme_map_cache = None  # stock -> theme
_theme_events_cache = None  # theme -> daily event aggregates

def _load_theme_map() -> dict:
    """加载 watchlist 中股票的 theme 归属"""
    global _theme_map_cache
    if _theme_map_cache is not None:
        return _theme_map_cache
    
    path = os.path.join(DATA_DIR, "universe", "watchlist.json")
    if not os.path.exists(path):
        _theme_map_cache = {}
        return _theme_map_cache
    
    with open(path, encoding="utf-8") as f:
        wl = json.load(f)
    
    mapping = {}
    for s in wl.get("watchlist", []):
        code6 = s["code"][:6]
        theme = s.get("theme", "未分类")
        mapping[code6] = theme
    _theme_map_cache = mapping
    return mapping

def _precompute_theme_events(kline_dates: pd.Series) -> dict:
    """预计算所有 theme 的每日事件聚合

    Returns:
        {theme_name: pd.DataFrame with tev_* columns indexed by date}
    """
    global _theme_events_cache
    if _theme_events_cache is not None:
        return _theme_events_cache
    
    theme_map = _load_theme_map()
    events_pool = _load_events()
    if len(events_pool) == 0:
        _theme_events_cache = {}
        return _theme_events_cache
    
    # 给每条事件标上 theme
    events_pool = events_pool.copy()
    events_pool["code6"] = events_pool["code"].str[:6]
    events_pool["theme"] = events_pool["code6"].map(theme_map).fillna("未分类")
    
    result = {}
    for theme_name, grp in events_pool.groupby("theme"):
        if len(grp) == 0:
            continue
        
        # 按天汇总 (同 calc_event_v2_features)
        daily = grp.copy()
        daily["_date"] = daily["date"].dt.date
        summ = daily.groupby("_date").agg(
            p0_cnt=("p_level", lambda x: (x == "P0").sum()),
            p1_cnt=("p_level", lambda x: (x == "P1").sum()),
            p2_cnt=("p_level", lambda x: (x == "P2").sum()),
            bull_cnt=("dir_hard", lambda x: (x > 0).sum()),
            bear_cnt=("dir_hard", lambda x: (x < 0).sum()),
            net_score=("impact", "sum"),
        ).reset_index()
        summ.columns = ["date", "p0_cnt", "p1_cnt", "p2_cnt", "bull_cnt", "bear_cnt", "net_score"]
        summ["date"] = pd.to_datetime(summ["date"])
        
        # 对齐交易日
        feat = kline_dates.to_frame("date").copy()
        feat = feat.merge(summ, on="date", how="left")
        for c in ["p0_cnt","p1_cnt","p2_cnt","bull_cnt","bear_cnt","net_score"]:
            feat[c] = feat[c].fillna(0).astype(float)
        
        # 滚动 + 衰减(与 calc_event_v2 一致)
        feat["tev_decay_bull"] = feat["bull_cnt"].rolling(20, min_periods=1).apply(
            lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)
        feat["tev_decay_bear"] = feat["bear_cnt"].rolling(20, min_periods=1).apply(
            lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)
        
        for w in [5, 21]:
            feat[f"tev_p0_{w}d"] = feat["p0_cnt"].rolling(w, min_periods=1).sum()
            feat[f"tev_p1_{w}d"] = feat["p1_cnt"].rolling(w, min_periods=1).sum()
            feat[f"tev_p2_{w}d"] = feat["p2_cnt"].rolling(w, min_periods=1).sum()
            feat[f"tev_bull_{w}d"] = feat["bull_cnt"].rolling(w, min_periods=1).sum()
            feat[f"tev_bear_{w}d"] = feat["bear_cnt"].rolling(w, min_periods=1).sum()
            feat[f"tev_net_{w}d"] = feat["net_score"].rolling(w, min_periods=1).sum()
            feat[f"tev_decay_n_{w}d"] = (feat["tev_decay_bull"].rolling(w, min_periods=1).sum() -
                                          feat["tev_decay_bear"].rolling(w, min_periods=1).sum()).fillna(0)
        
    # MA rolling(与其他维度一致)
        keep = [c for c in feat.columns if c.startswith("tev_") or c == "date"]
        result[theme_name] = feat[keep].copy()

    # 加一个"全市场"版: 所有股票事件聚合
    all_events_pool = _load_events().copy()
    if len(all_events_pool) > 0:
        daily_all = all_events_pool.copy()
        daily_all["_date"] = daily_all["date"].dt.date
        summ_all = daily_all.groupby("_date").agg(
            p0_cnt=("p_level", lambda x: (x == "P0").sum()),
            p1_cnt=("p_level", lambda x: (x == "P1").sum()),
            p2_cnt=("p_level", lambda x: (x == "P2").sum()),
            bull_cnt=("dir_hard", lambda x: (x > 0).sum()),
            bear_cnt=("dir_hard", lambda x: (x < 0).sum()),
            net_score=("impact", "sum"),
        ).reset_index()
        summ_all.columns = ["date", "p0_cnt", "p1_cnt", "p2_cnt", "bull_cnt", "bear_cnt", "net_score"]
        summ_all["date"] = pd.to_datetime(summ_all["date"])

        feat_all = kline_dates.to_frame("date").copy()
        feat_all = feat_all.merge(summ_all, on="date", how="left")
        for c in ["p0_cnt","p1_cnt","p2_cnt","bull_cnt","bear_cnt","net_score"]:
            feat_all[c] = feat_all[c].fillna(0).astype(float)

        feat_all["tev_all_decay_bull"] = feat_all["bull_cnt"].rolling(20, min_periods=1).apply(
            lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)
        feat_all["tev_all_decay_bear"] = feat_all["bear_cnt"].rolling(20, min_periods=1).apply(
            lambda x: sum(v * _bell_decay(i) for i, v in enumerate(x)), raw=True)

        for w in [5, 21]:
            feat_all[f"tev_all_bull_{w}d"] = feat_all["bull_cnt"].rolling(w, min_periods=1).sum()
            feat_all[f"tev_all_bear_{w}d"] = feat_all["bear_cnt"].rolling(w, min_periods=1).sum()
            feat_all[f"tev_all_net_{w}d"] = feat_all["net_score"].rolling(w, min_periods=1).sum()
            feat_all[f"tev_all_decay_n_{w}d"] = (feat_all["tev_all_decay_bull"].rolling(w, min_periods=1).sum() -
                                                   feat_all["tev_all_decay_bear"].rolling(w, min_periods=1).sum()).fillna(0)

        # MA rolling(与其他维度一致)
        keep_all = [c for c in feat_all.columns if c.startswith("tev_all_") or c == "date"]
        result["__all__"] = feat_all[keep_all].copy()

    _theme_events_cache = result
    return result


# ─── 概念板块技术面特征 (v24: 板块级别日K/资金流等) ────────
# 对于每只股票, 聚合其所属概念板块的成员股数据, 生成板块级技术面特征

_CONCEPT_FEATS_CACHE: Optional[dict] = None  # {theme: DataFrame}


def _load_watchlist_theme_map() -> dict:
    """加载 watchlist_216.json 中股票的 theme 归属 (code6 → theme)."""
    path = os.path.join(DATA_DIR, "universe", "watchlist_216.json")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "universe", "watchlist.json")
    with open(path, encoding="utf-8") as f:
        wl = json.load(f)
    items = wl.get("watchlist", wl) if isinstance(wl, dict) else wl
    mapping = {}
    for s in items:
        c6 = s["code"][:6]
        theme = s.get("theme", "未分类")
        mapping[c6] = theme
    return mapping


def _precompute_concept_features(all_dates: pd.Series) -> dict:
    """预计算所有概念板块的每日技术面聚合特征。

    对每个主题(theme), 聚合其成员股的日K+资金流数据:
      - 收益率均值, 成交量总和, 成交额总和
      - 主力净流入总和, 主力净流入占比均值, DDE总和
      - 上述的 MA5/MA10 衍生
    Returns: {theme: DataFrame (date + con_* 列)}
    """
    global _CONCEPT_FEATS_CACHE
    if _CONCEPT_FEATS_CACHE is not None:
        return _CONCEPT_FEATS_CACHE

    theme_map = _load_watchlist_theme_map()  # code6 → theme
    if not theme_map:
        _CONCEPT_FEATS_CACHE = {}
        return {}

    # 按 theme 分组代码
    theme_codes: dict[str, list[str]] = {}
    for c6, theme in theme_map.items():
        theme_codes.setdefault(theme, []).append(c6)

    log.info("概览: %d 个主题", len(theme_codes))

    # 读取所有成员股日K + 资金流
    klines = []
    fflows = []
    for theme, codes in theme_codes.items():
        for c6 in codes:
            dk = read_kline(c6)
            if dk is not None:
                tmp = dk[["date", "close", "volume", "amount"]].copy()
                tmp["_theme"] = theme
                tmp["_code"] = c6
                tmp["_ret"] = tmp["close"].pct_change(1)
                klines.append(tmp)
            ff = read_fund_flow(c6)
            if ff is not None:
                ftmp = ff[["date", "main_force_net", "main_force_pct", "dde_net"]].copy()
                ftmp["_theme"] = theme
                ftmp["_code"] = c6
                fflows.append(ftmp)

    if not klines:
        _CONCEPT_FEATS_CACHE = {}
        return {}

    kl_all = pd.concat(klines, ignore_index=True)
    if fflows:
        ff_all = pd.concat(fflows, ignore_index=True)
        # 合并资金流到 K 线
        kl_all = kl_all.merge(
            ff_all[["date", "_code", "main_force_net", "main_force_pct", "dde_net"]],
            on=["date", "_code"], how="left")

    # 按 theme + date 聚合
    grp = kl_all.groupby(["_theme", "date"])
    con = grp.agg(
        con_ret_1d=("_ret", "mean"),
        con_vol=("volume", "sum"),
        con_amount=("amount", "sum"),
    ).reset_index()
    if "main_force_net" in kl_all.columns:
        mf = grp.agg(
            con_mf_net=("main_force_net", "sum"),
            con_mf_pct=("main_force_pct", "mean"),
            con_dde_net=("dde_net", "sum"),
        ).reset_index()
        con = con.merge(mf, on=["_theme", "date"], how="left")

    con["date"] = pd.to_datetime(con["date"])
    con = con.sort_values(["date", "_theme"]).reset_index(drop=True)

    # 对齐交易日并计算 MA5/MA10 (按theme分组)
    result = {}
    for theme in con["_theme"].unique():
        tdf = con[con["_theme"] == theme].copy()
        # reindex 到全部交易日
        full = all_dates.to_frame("date").copy()
        full = full.merge(tdf.drop(columns=["_theme"]), on="date", how="left")
        val_cols = [c for c in full.columns if c.startswith("con_") and c != "date"]
        for c in val_cols:
            full[c] = full[c].ffill().fillna(0)
            full[f"{c}_ma5"] = full[c].rolling(5, min_periods=3).mean()
            full[f"{c}_ma10"] = full[c].rolling(10, min_periods=5).mean()
        result[theme] = full

    _CONCEPT_FEATS_CACHE = result
    log.info("概念特征预计算完成: %d 个主题", len(result))
    return result


def calc_concept_features(code6: str, trade_dates: pd.Series) -> Optional[pd.DataFrame]:
    """为单只股票加载其所属概念板块的技术面特征。

    Args:
        code6: 6位股票代码
        trade_dates: 交易日序列

    Returns:
        DataFrame 含 con_* 列, 或 None(无概念映射时)
    """
    theme_map = _load_watchlist_theme_map()
    theme = theme_map.get(code6)
    if not theme:
        return None
    pool = _precompute_concept_features(trade_dates)
    if not pool or theme not in pool:
        return None
    return pool[theme]

# ─── 事件/公告特征 ───────────────────────────────────────────

def calc_event_features(df: pd.DataFrame, kline_dates: pd.Series) -> Optional[pd.DataFrame]:
    """公告事件特征 — 公告密度, 最近公告天数

    Args:
        df: 公告 DataFrame,需含 [date] 列
        kline_dates: 交易日序列,用于对齐日期

    Returns:
        事件特征 DataFrame,或 None(无公告数据时)
    """
    if df is None or len(df) == 0:
        return None
    event_dates = df["date"].value_counts().reset_index()
    event_dates.columns = ["date", "ann_count"]
    event_dates = event_dates.sort_values("date")
    event_dates["date"] = pd.to_datetime(event_dates["date"])
    result = kline_dates.to_frame("date").copy()
    result = result.merge(event_dates, on="date", how="left")
    result["ann_count"] = result["ann_count"].fillna(0).astype(int)
    result["ann_5d"] = result["ann_count"].rolling(5, min_periods=1).sum()
    result["ann_21d"] = result["ann_count"].rolling(21, min_periods=1).sum()
    result["has_ann"] = (result["ann_count"] > 0).astype(int)
    result["days_since_ann"] = np.nan
    last_ann = -999
    for i in range(len(result)):
        if result.iloc[i]["has_ann"]:
            last_ann = 0
        elif last_ann >= 0:
            last_ann += 1
        result.iloc[i, result.columns.get_loc("days_since_ann")] = last_ann
    return result

# ─── 基本面特征 ──────────────────────────────────────────────

def _load_stock_list_row(code6: str) -> Optional[dict]:
    """从 stock_list 快照加载单只股票当前基本面值

    匹配股票代码前缀,映射列名到标准基本面字段(pe/pb/roe/mcap/revenue/profit/eps/bps 等)。

    Args:
        code6: 6位股票代码

    Returns:
        dict {col_name: float_value, ...},或 None(文件不存在或未匹配到时)
    """
    path = os.path.join(DATA_DIR, "universe", "stock_list.parquet")
    if not os.path.exists(path):
        return None
    df_all = pd.read_parquet(path)
    mask = df_all["股票代码"].str.startswith(code6)
    if not mask.any():
        return None
    row = df_all[mask].iloc[0]

    funda_map = [
        ("市盈率(pe)", "pe"),
        ("市净率(pb)", "pb"),
        ("净资产收益率roe", "roe"),
        ("总市值", "mcap"),
        ("营业总收入", "revenue"),
        ("营业收入", "revenue"),
        ("归属于母公司所有者的净利润", "profit"),
        ("净利润", "profit"),
        ("基本每股收益", "eps"),
        ("每股收益", "eps"),
        ("每股净资产", "bps"),
        ("每股净资产bps", "bps"),
        ("毛利率", "gross_margin"),
        ("资产负债率", "debt_ratio"),
        ("总资产", "total_assets"),
        ("资产总计", "total_assets"),
        ("现金流", "operate_cf"),
        ("经营活动现金流", "operate_cf"),
    ]
    dst_to_col = {}
    for src_pattern, dst in funda_map:
        if dst in dst_to_col:
            continue
        matched = next((col for col in df_all.columns if src_pattern in col), None)
        if matched:
            dst_to_col[dst] = matched

    out = {}
    for dst, col_name in dst_to_col.items():
        val = row.get(col_name)
        if val is not None and str(val) != "nan" and val != "":
            try:
                out[dst] = float(val)
            except (ValueError, TypeError):
                out[dst] = np.nan
        else:
            out[dst] = np.nan
    return out


def _fund_pub_date(d):
    """报告期截止日 → 保守发布日(未来泄露修复 #2)

    A股财报披露时限:
      Q1 (3/31)  → 4/30  (+30天)
      H1 (6/30)  → 8/31  (+62天)
      Q3 (9/30)  → 10/31 (+31天)
      年报 (12/31) → 次年4/30 (+120天)

    Args:
        d: 报告期截止日 (Timestamp)

    Returns:
        保守发布日 (Timestamp)
    """
    m = d.month
    if m == 3:
        return d + pd.Timedelta(days=30)
    elif m == 6:
        return d + pd.Timedelta(days=62)
    elif m == 9:
        return d + pd.Timedelta(days=31)
    elif m == 12:
        return d + pd.Timedelta(days=120)
    else:
        return d + pd.Timedelta(days=90)


def calc_fundamental_features(code6: str, trade_dates: pd.Series) -> Optional[pd.DataFrame]:
    """基本面特征 — 历史数据优先,缺列从 stock_list 快照补充

    策略：
      1. 优先加载历史基本面 parquet(来自 iFinD 采集)
      2. 过滤不合理年份(2010-2030),去重同日期行
      3. reindex 到交易日并前向填充(最多 250 天)
      4. 缺失列从 stock_list 快照的当前值填充

    Args:
        code6: 6位股票代码
        trade_dates: 交易日日期序列,用于对齐

    Returns:
        DataFrame with [date, pe, pb, mcap, revenue, profit, eps, bps, roe, ...]
    """
    result = trade_dates.to_frame("date").copy()
    core_cols = ["pe", "pb", "mcap", "revenue", "profit", "eps", "bps", "roe",
                 "total_assets", "debt_ratio", "gross_margin", "operate_cf"]
    have_from_hist = set()

    # ── 优先: 历史基本面 parquet ──
    hist_path = os.path.join(DATA_DIR, "raw", "fundamentals", f"{code6}.parquet")
    if os.path.exists(hist_path):
        try:
            hist = pd.read_parquet(hist_path)
            hist["date"] = pd.to_datetime(hist["date"], errors='coerce')
            # 过滤不合理年份(iFinD 偶发乱码年份如 9046)
            hist = hist[hist["date"].dt.year.between(2010, 2030)]
            if len(hist) > 0:
                hist = hist.sort_values("date")
                # 同日期有多行时保留首行(季度/年度报告取第一个)
                hist = hist.drop_duplicates(subset=["date"], keep="first")
                hist = hist.set_index("date")
                # 报告期截止日 → 发布日偏移(未来泄露修复 #2)
                hist.index = hist.index.map(_fund_pub_date)
                # 年报(12/31→次年4/30) 与一季报(3/31→4/30) 会撞到同一发布日,
                # 映射后必须再去重, 否则 reindex 抛 "duplicate labels" 被静默
                # 吞掉, 导致全部基本面列丢失。同日保留报告期更新的一条。
                hist = hist[~hist.index.duplicated(keep="last")].sort_index()

                for col in core_cols:
                    if col not in hist.columns:
                        continue
                    s = pd.to_numeric(hist[col], errors='coerce').sort_index()
                    # reindex 到全部交易日并用前值填充(最多 250 天 ≈ 1 年)
                    s = s.reindex(result.set_index("date").index, method='ffill', limit=250)
                    if s.notna().any():
                        result[col] = s.values
                        have_from_hist.add(col)
        except (ValueError, TypeError, KeyError) as e:
            log.debug("历史基本面加载失败 %s: %s", code6, e)

    # ── stock_list 快照不再填充历史行(未来泄露修复 #1)──
    # 2026年快照值(pe/pb/mcap等)填到2010-2025每一行 = 严重未来泄露
    # 缺历史数据的列保留 NaN,模型在回测中用 train median 填充

    return result

# ─── 商品/产品价格特征(v7新増)────────────────────────────

# CN/亚洲数据(CN交易日历,不需shift)
_CN_COMMODITY_FILES = [
    ("中国大宗商品价格指数", "cn_commodity_idx", 0, ["最新值"]),
    ("六氟化钨", "wf6_price", 0, ["最新值"]),
    ("六氟磷酸锂", "lipf6_price", 0, ["最新值"]),
    ("EVA光伏料", "eva_price", 0, ["最新值"]),
    ("磷矿石", "phos_price", 0, ["最新值"]),
    ("纯碱", "soda_price", 0, ["最新值"]),
    ("黄金", "cn_gold", 0, ["最新值"]),         # 沪金主连 CN期货
    ("白银", "cn_silver", 0, ["最新值"]),        # 沪银主连 CN期货
    ("铜", "cn_copper", 0, ["最新值"]),           # 沪铜主连 CN期货
    ("铝", "cn_aluminum", 0, ["最新值"]),
    ("锌", "cn_zinc", 0, ["最新值"]),
    ("镍", "cn_nickel", 0, ["最新值"]),
    ("锡", "cn_tin", 0, ["最新值"]),
    ("A50期货", "a50_futures", 0, ["最新值"]),    # CN股指期货
]

# 全球指数期货(US交易日历,需shift+1对齐A股)
_GLOBAL_INDEX_FILES = [
    ("标普期货", "sp_futures", 0, ["最新值"]),
    ("道指期货", "dj_futures", 0, ["最新值"]),
    ("纳指期货", "nq_futures", 0, ["最新值"]),
    ("全球半导体SOX", "sox", 0, ["最新值"]),
]

# 汇率数据(US/欧洲交易时段,需shift+1对齐A股T+1)
_FOREX_FILES = [
    ("USDIND", "usdind", 0, ["最新值"]),    # 美元指数 (iFinD DX0Y.NBT 连续)
    ("USDCNH", "usdcnh", 0, ["最新值"]),    # 美元兑离岸人民币
    ("USDJPY", "usdjpy", 0, ["最新值"]),    # 美元兑日元
]

# 国债收益率(CN同交易日历不偏移,US需shift+1)
_BOND_CN_FILES = [
    ("CN2Y", "cn2y", 0, ["最新值"]),        # 中国国债2年期收益率
    ("CN5Y", "cn5y", 0, ["最新值"]),        # 中国国债5年期收益率
]
_BOND_US_FILES = [
    ("US2Y", "us2y", 0, ["最新值"]),        # 美国国债2年期收益率
    ("US5Y", "us5y", 0, ["最新值"]),        # 美国国债5年期收益率
]

_commodity_cache = None  # precomputed commodity features

def _load_commodity_group(files: list, shift_days: int = 0) -> Optional[pd.DataFrame]:
    """加载一组 commodity 数据,可选偏移日期
    
    Args:
        files: (fname, col_prefix, _, val_cols) 元组列表
        shift_days: 日期偏移天数(US数据=1, CN数据=0)
    
    Returns:
        DataFrame, 或 None
    """
    macro_dir = os.path.join(DATA_DIR, "raw", "macro")
    series = []
    for fname, col_prefix, _, val_cols in files:
        path = os.path.join(macro_dir, f"{fname}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            date_cols = [c for c in df.columns if "日期" in c or "date" in c.lower()]
            if not date_cols: continue
            df["date"] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.dropna(subset=["date"]).sort_values("date")
            for vc in val_cols:
                if vc in df.columns:
                    s = df[["date", vc]].copy()
                    s.columns = ["date", col_prefix]
                    s[col_prefix] = pd.to_numeric(s[col_prefix], errors='coerce')
                    if shift_days > 0:
                        s["date"] = s["date"] + pd.Timedelta(days=shift_days)
                    series.append(s)
                    break
        except:
            continue
    if not series: return None
    merged = series[0]
    for s in series[1:]:
        merged = merged.merge(s, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)

commodity_cache = None

def calc_commodity_features(trade_dates: pd.Series) -> Optional[pd.DataFrame]:
    """商品/产品价格特征 — CN数据 + 全球指数期货

    CN商品(沪金/六氟化钨等)：同一交易日历,不偏移
    全球指数(SP/DJ/NQ/SOX)：US T日闭市=BJT T+1→shift+1天

    对每种商品计算:
      - 日/3日/13日变化率
      - 21日z-score
    """
    global commodity_cache
    if commodity_cache is None:
        cn = _load_commodity_group(_CN_COMMODITY_FILES, shift_days=0)
        gi = _load_commodity_group(_GLOBAL_INDEX_FILES, shift_days=1)
        fx = _load_commodity_group(_FOREX_FILES, shift_days=1)
        bd_cn = _load_commodity_group(_BOND_CN_FILES, shift_days=0)
        bd_us = _load_commodity_group(_BOND_US_FILES, shift_days=1)
        merged = cn
        for other in [gi, fx, bd_cn, bd_us]:
            if other is not None:
                if merged is not None:
                    merged = merged.merge(other, on="date", how="outer")
                else:
                    merged = other
        if merged is not None:
            commodity_cache = merged.sort_values("date").reset_index(drop=True)

    if commodity_cache is None or len(commodity_cache) < 20:
        return None

    raw = commodity_cache.copy()
    all_cols = [c for c in raw.columns if c != "date"]
    fx_cols = [c for c in all_cols if c.startswith(("usdind", "usdcnh", "usdjpy", "cn2y", "cn5y", "us2y", "us5y"))]
    deriv_cols = [c for c in all_cols if c not in fx_cols]

    feats = raw[["date"]].copy()
    # 汇率: 仅保留原始价格,不计算变化率/z-score(MA3/13由build_features_for_stock统一添加)
    for pc in fx_cols:
        feats[pc] = raw[pc].ffill()
    # 商品/指数: 计算变化率+z-score
    for pc in deriv_cols:
        s = raw[pc].ffill()
        feats[f"{pc}_chg_1d"] = s.pct_change(1) * 100
        feats[f"{pc}_chg_2d"] = s.pct_change(2) * 100
        feats[f"{pc}_chg_5d"] = s.pct_change(5) * 100
        feats[f"{pc}_chg_21d"] = s.pct_change(21) * 100
        rm = s.rolling(21).mean()
        rs = s.rolling(21).std().replace(0, np.nan)
        feats[f"{pc}_z_21d"] = ((s - rm) / rs).fillna(0)

    # 对齐交易日
    result = trade_dates.to_frame("date").copy()
    result = result.merge(feats, on="date", how="left")
    for c in result.columns:
        if c != "date":
            result[c] = result[c].ffill(limit=5)
    return result

# ─── 标签 ────────────────────────────────────────────────────

def calc_labels(df_k: pd.DataFrame) -> pd.DataFrame:
    """标签生成 — 未来 N 日收益率(用于监督学习目标)

    Args:
        df_k: K线 DataFrame,需含 [date, close] 列

    Returns:
        DataFrame with [date, fwd_{1/5/10/20}d_ret] 标签列
    """
    c = df_k["close"]
    o = df_k["open"]
    result = df_k[["date"]].copy()
    result["fwd_1d_ret"] = c.shift(-1) / c - 1
    result["fwd_2d_ret"] = c.shift(-2) / c - 1
    result["fwd_5d_ret"] = c.shift(-5) / c - 1
    result["fwd_21d_ret"] = c.shift(-21) / c - 1
    # 开盘价执行标签: T日开盘买入, T+1日开盘卖出 -> open_{t+1}/open_t - 1
    result["fwd_1d_open_ret"] = o.shift(-1) / o - 1
    result["fwd_1d_exec_ret"] = c.shift(-1) / o.shift(-1) - 1
    result["fwd_1d_t1_open_ret"] = o.shift(-2) / o.shift(-1) - 1
    return result

# ─── 主流程 ───────────────────────────────────────────────────

def build_features_for_stock(code: str, code6: str) -> Optional[pd.DataFrame]:
    """构建单只股票的完整特征矩阵(六维)

    依次加载 K线 → 技术面 → 标签 → 资金面 → 公告事件 → 基本面 → 宏观 → 链主特征。

    Args:
        code: 完整股票代码(如 600519.SH)
        code6: 6位数字代码(如 600519)

    Returns:
        特征 DataFrame(54列),或 None(K线数据不足60日时)
    """
    df_k = read_kline(code6)
    df_ff = read_fund_flow(code6)      # consolidated fundflow_history
    df_ev = read_events_for_stock(code6)  # events_clean P0-P4
    df_ann = read_announcements(code6)
    if df_k is None or len(df_k) < 60:
        return None

    tech = calc_technical_features(df_k)
    if tech is None:
        return None
    labels = calc_labels(df_k)

    # 合并基础
    result = tech.merge(labels, on="date", how="left")

    # 资金面
    if df_ff is not None:
        fund = calc_fund_features(df_ff)
        if fund is not None:
            result = result.merge(fund, on="date", how="left")

    # P0-P4 事件
    if df_ev is not None:
        ev2 = calc_event_v2_features(df_ev, result["date"])
        if ev2 is not None:
            result = result.merge(ev2, on="date", how="left")

    # 板块事件
    theme_map = _load_theme_map()
    theme_name = theme_map.get(code6)
    theme_events = _precompute_theme_events(result["date"])
    if theme_name and theme_name in theme_events:
        result = result.merge(theme_events[theme_name], on="date", how="left")
    if "__all__" in theme_events:
        result = result.merge(theme_events["__all__"], on="date", how="left")

    # 公告事件
    if df_ann is not None:
        event = calc_event_features(df_ann, result["date"])
        if event is not None:
            result = result.merge(event, on="date", how="left")

    # 基本面(日频不变)
    funda = calc_fundamental_features(code6, result["date"])
    if funda is not None:
        result = result.merge(funda, on="date", how="left")

    # 宏观
    macro_dir = os.path.join(DATA_DIR, "raw", "macro")
    pmi_path = os.path.join(macro_dir, "中国PMI.parquet")
    if os.path.exists(pmi_path):
        df_pmi = pd.read_parquet(pmi_path)
        df_pmi["date"] = pd.to_datetime(df_pmi["月份"].str.replace("年", "-").str.replace("月份", "-01"), errors='coerce')
        # CN PMI 月末发布(未来泄露修复 #3): 月初→月末+1个交易日
        df_pmi["date"] = df_pmi["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        df_pmi = df_pmi.dropna(subset=["date"])
        df_pmi_v = df_pmi[["date", "制造业-指数"]].rename(columns={"制造业-指数": "cn_pmi"})
        result = result.merge(df_pmi_v, on="date", how="left")
        result["cn_pmi"] = result["cn_pmi"].ffill()

    ism_path = os.path.join(macro_dir, "美国ISM制造业PMI.parquet")
    if os.path.exists(ism_path):
        df_ism = pd.read_parquet(ism_path)
        df_ism["date"] = pd.to_datetime(df_ism["日期"], errors='coerce')
        # US ISM 发布于美东10:00=北京22:00,A股已收盘 → 次日可用(未来泄露修复 #3)
        df_ism["date"] = df_ism["date"] + pd.Timedelta(days=1)
        df_ism = df_ism.dropna(subset=["date"])
        df_ism = df_ism[df_ism["商品"] == "美国ISM制造业PMI报告"]
        df_ism_v = df_ism[["date", "今值"]].rename(columns={"今值": "us_ism_pmi"})
        result = result.merge(df_ism_v, on="date", how="left")
        result["us_ism_pmi"] = result["us_ism_pmi"].ffill()

    # ─── 商品/产品价格特征(v7新増)────────────────────────────
    commodity_feats = calc_commodity_features(result["date"])
    if commodity_feats is not None:
        result = result.merge(commodity_feats, on="date", how="left")

    # ─── 概念板块技术面特征(v24) ───────────────────────────
    con_feats = calc_concept_features(code6, result["date"])
    if con_feats is not None:
        result = result.merge(con_feats, on="date", how="left")

    # ─── 链主特征 ────────────────────────────────────────────
    _LEADER_MAP = None
    if _LEADER_MAP is None:
        lm = {}
        sc_path = os.path.join(DATA_DIR, "universe", "supply_chain_map.json")
        if os.path.exists(sc_path):
            with open(sc_path, encoding="utf-8") as f:
                sc = json.load(f)
            ev = {"核心": 3, "高": 2, "中": 1}
            for chain in sc["chains"]:
                for link in chain["demand_links"]:
                    for s in link["a_share_suppliers"]:
                        c6 = s["code"][:6]
                        if c6 not in lm:
                            lm[c6] = {"cnt": 0, "exp": 0, "binding_sum": 0.0}
                        lm[c6]["cnt"] += 1
                        lm[c6]["exp"] = max(lm[c6]["exp"], ev.get(s["exposure"], 0))
                        lm[c6]["binding_sum"] += s.get("scoring", {}).get("binding", 0)
        _LEADER_MAP = lm
    info = _LEADER_MAP.get(code6, {})
    result["has_leader"] = 1 if info.get("cnt", 0) > 0 else 0
    result["leader_count"] = info.get("cnt", 0)
    result["leader_exp"] = info.get("exp", 0)
    result["leader_binding_sum"] = info.get("binding_sum", 0.0)

    # ─── 滚动均值特征 (MA3/13) — 让截面带历史动量 ─────
    _ROLL_COLS = [c for c in result.columns
                  if c not in ("date", "code")
                  and not c.startswith(("fwd_", "ev_", "tev_", "ann_", "has_", "leader_"))
                  and c not in ("cn_pmi", "us_ism_pmi")
                  and result[c].dtype in ("float64", "float32", "int64", "int32")]
    for w in (5, 20):
        for c in _ROLL_COLS:
            roll = result[c].rolling(w, min_periods=w//2+1).mean()
            new_name = f"{c}_ma{w}"
            if new_name not in result.columns:
                result.loc[:, new_name] = roll

    result["code"] = code
    return result

def _build_one_stock(code: str, out_dir: str, cutoff: pd.Timestamp) -> bool:
    """构建单只股票并直接写特征文件, 只返回成败

    定义在模块顶层 (而非 build_all 内的闭包) 是为了能被 ProcessPoolExecutor
    pickle; 不返回 DataFrame 是为了避开进程间传 1GB+ 数据的开销。
    """
    code6 = code[:6]
    try:
        df = build_features_for_stock(code, code6)
        if df is None:
            return False
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= cutoff].copy()
        if len(df) < 20:
            return False
        df.to_parquet(os.path.join(out_dir, f"{code6}.parquet"), index=False)
        return True
    except Exception as e:
        log.warning("  %s 失败: %s", code, e)
        return False


def build_all(incremental: bool = True, max_workers: int = 16,
              watchlist_file: Optional[str] = None,
              out_file: Optional[str] = None,
              procs: int = 0) -> None:
    """构建全部股票的特征矩阵 (六维特征 → 训练集 v15)

    流程:
      1. 预计算概念板块特征/主题事件 (单线程, 仅一次)
      2. 并行遍历 watchlist 每只股票 (ThreadPoolExecutor)
      3. 增量模式: 特征文件日期≥K线日期则跳过
      4. 合并为训练集 data/processed/training_data_v15.parquet

    Args:
        incremental: 是否启用增量跳过 (默认 True)
        max_workers: 并行线程数 (默认 16)
        watchlist_file: 股票池 json 文件名, 默认 watchlist_top120.json
        out_file: 输出训练集文件名, 默认 training_data_v15.parquet
        procs: >0 时用多进程真并行取代多线程 (绕过 pandas 的 GIL 瓶颈)
    """
    from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                    as_completed)

    if watchlist_file:
        watchlist_path = os.path.join(DATA_DIR, "universe", watchlist_file)
        if not os.path.exists(watchlist_path):
            raise FileNotFoundError(watchlist_path)
    else:
        watchlist_path = os.path.join(DATA_DIR, "universe", "watchlist_top120.json")
        if not os.path.exists(watchlist_path):
            watchlist_path = os.path.join(DATA_DIR, "universe", "watchlist.json")
    with open(watchlist_path, encoding="utf-8") as f:
        watch = json.load(f)
    stocks = watch.get("watchlist", [])
    log.info("训练特征: %d 只 (%s), incremental=%s, max_workers=%d",
             len(stocks), os.path.basename(watchlist_path), incremental, max_workers)

    out_dir = os.path.join(DATA_DIR, "processed", "features")
    os.makedirs(out_dir, exist_ok=True)
    cutoff = pd.Timestamp("2022-09-01")

    # ── 0. 单线程预热缓存: 用第1只股票日期激发概念/主题缓存, 避免多线程重复计算 ──
    t_pre = time.time()
    if stocks:
        s0 = stocks[0]
        _ = build_features_for_stock(s0["code"], s0["code"][:6])
    log.info("  缓存预热: %.1fs", time.time() - t_pre)

    # ── 1. 确定需要构建的股票 ──
    to_build = []
    skipped = 0
    for s in stocks:
        code = s["code"]
        code6 = code[:6]
        feat_path = os.path.join(out_dir, f"{code6}.parquet")
        if incremental and os.path.exists(feat_path):
            # 检查特征文件是否足够新
            try:
                feat_max = pd.read_parquet(feat_path, columns=["date"])["date"].max()
                kl_path = os.path.join(DATA_DIR, "raw", "kline", f"{code6}.parquet")
                if os.path.exists(kl_path):
                    kl_max = pd.read_parquet(kl_path, columns=["date"])["date"].max()
                    if feat_max >= kl_max:
                        skipped += 1
                        continue
            except Exception:
                pass  # 异常则重建
        to_build.append(s)
    log.info("  需构建 %d 只, 跳过 %d 只 (已最新)", len(to_build), skipped)

    # ── 2. 并行构建 ──
    #   pandas 的 rolling/merge 大量持有 GIL, 多线程只能跑满约 1 个核;
    #   procs>0 时用多进程真并行 (worker 只写盘, 不把 DataFrame pickle 回来)
    successes, build_fail = 0, 0
    t0 = time.time()

    if procs and procs > 0:
        Pool, n_par, tag = ProcessPoolExecutor, procs, f"{procs} 进程"
    else:
        Pool, n_par, tag = ThreadPoolExecutor, max_workers, f"{max_workers} 线程"
    log.info("  并行方式: %s", tag)

    with Pool(max_workers=n_par) as pool:
        fut_map = {pool.submit(_build_one_stock, s["code"], out_dir, cutoff): s
                   for s in to_build}
        for i, fut in enumerate(as_completed(fut_map)):
            s = fut_map[fut]
            try:
                ok = fut.result()
            except Exception as e:
                log.warning("  %s 异常: %s", s["code"], e)
                ok = False
            if ok:
                successes += 1
            else:
                build_fail += 1
            if (i + 1) % 50 == 0 or i == len(to_build) - 1:
                log.info("  [%d/%d] %s — 成功 %d, 失败 %d",
                         i + 1, len(to_build), s["name"], successes, build_fail)

    elapsed = time.time() - t0
    log.info("  并行构建完成: %.1fs (成功 %d, 失败 %d, 跳过 %d)",
             elapsed, successes, build_fail, skipped)

    # ── 3. 从特征文件统一读回合并 (新构建的和跳过的一视同仁) ──
    all_dfs = []
    for s in stocks:
        code6 = s["code"][:6]
        feat_path = os.path.join(out_dir, f"{code6}.parquet")
        if not os.path.exists(feat_path):
            continue
        try:
            df = pd.read_parquet(feat_path)
            df["date"] = pd.to_datetime(df["date"])
            df["code"] = df["code"].astype(str)
            all_dfs.append(df)
        except Exception as e:
            log.warning("  读取特征文件失败 %s: %s", code6, e)

    # 合并保存
    if all_dfs:
        train = pd.concat(all_dfs, ignore_index=True)
        train = train.drop_duplicates(subset=["date", "code"]).reset_index(drop=True)
        train_path = os.path.join(DATA_DIR, "processed",
                                  out_file or "training_data_v15.parquet")
        train.to_parquet(train_path, index=False)
        log.info("训练集 v15: %d 行, %d 列, %d 只股票, max=%s",
                 len(train), len(train.columns), train["code"].nunique(), train["date"].max())
    else:
        log.warning("无有效特征数据")

if __name__ == "__main__":
    import sys as _sys
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)  # quant-strategy
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", default=True,
                    help="启用增量跳过 (默认开启)")
    ap.add_argument("--no-incremental", dest="incremental", action="store_false",
                    help="禁用增量跳过")
    ap.add_argument("--workers", type=int, default=16,
                    help="并行线程数 (默认 16)")
    ap.add_argument("--procs", type=int, default=0,
                    help="多进程并行数 (>0 时绕开 pandas GIL 瓶颈, 推荐 24)")
    ap.add_argument("--watchlist", type=str, default=None,
                    help="股票池 json 文件名 (data/universe/ 下), 如 watchlist_pit.json")
    ap.add_argument("--out", type=str, default=None,
                    help="输出训练集文件名 (data/processed/ 下)")
    args = ap.parse_args()
    build_all(incremental=args.incremental, max_workers=args.workers,
              watchlist_file=args.watchlist, out_file=args.out,
              procs=args.procs)
