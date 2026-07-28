"""
app_nicegui.py — 量化策略 Web 平台 (NiceGUI 版, 替代 Streamlit app.py)

页面结构 (工作流驱动, 默认首页=训练&回测):
  1. 训练 & 回测 (默认页) : 参数面板 + 一键训练 + 结果可视化(含年化收益)
  2. 关注圈               : 关注圈(可编辑, 训练实际范围) + 自选股(可编辑)
  3. 特征                 : 增删改(默认 v23) + 缺失率 + 分布可视化
  4. 结果对比             : 版本/Run 叠加 + 参数-绩效对照 + 方向性α照妖镜

复用 backend (database/paths/trainer/backtest_results/stock_diagnostics/models),
仅替换 UI 层。图表用 plotly, 表格用 ag-grid。
"""
from __future__ import annotations

import sys
import os
import subprocess
import re
import json
import asyncio
import threading
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui, app
import time

# ── 路径 / 导入 backend ──────────────────────────────────────────────────────
WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from backend import trainer  # noqa: F401
from backend.trainer import train, TrainParams  # noqa: F401
from backend.trainer import _TrainCancelled  # noqa: F401
from backend.database import (  # noqa: F401
    get_runs, get_run_detail, delete_run, init_db,
    get_settings, save_settings,
)
from backend.paths import (  # noqa: F401
    latest_training_data, earliest_train_date, watchlist_path,
    self_selected_path, train_pool_path, load_universe_codes, processed_dir,
    stock_name, _stock_name_map,
)
from backend.backtest_results import discover_backtests, load_backtest, directional_alpha  # noqa: F401
from backend.stock_diagnostics import diagnose_self_selected  # noqa: F401

import pyarrow.parquet as pq  # noqa: F401

# ── 安全过滤 (与 trainer 默认一致) ───────────────────────────────────────────
_SKIP = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
         "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret"}
_INITIAL_CAPITAL = 2_000_000  # 初始资金200万元
_LEAK = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
# 消融实验确认负贡献 → 剔除, 但数据继续采集
_EXCLUDED = {"mtss_1d", "mtss_z", "mtss_1d_ma5", "mtss_z_ma5", "mtss_1d_ma20", "mtss_z_ma20",
             # P0: 完全重复特征 (corr=1.000 或 >0.96), 保留 mf_signal/macd 系列
             "mf_pct_1d", "mf_pct_1d_ma5", "mf_pct_1d_ma20",
             "macd_signal", "macd_signal_ma5", "macd_signal_ma20"}
# 基本面 MA5/20 衍生 (季度数据, 日频MA无意义)
_FUNDA_RAW = ["pe", "pb", "revenue", "profit", "eps", "bps", "debt_ratio",
              "gross_margin", "roe", "total_assets"]
_EXCLUDED.update(f"{r}_{s}" for r in _FUNDA_RAW for s in ("ma5", "ma20"))

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#dcdcdc"),
    margin=dict(l=40, r=20, t=36, b=20),
)
FEATURE_DIMS = {
    "技术面": ["ret_", "ma5_pct", "ma20_pct", "macd", "rsi_", "atr", "vol_", "pos_"],
    "资金流": ["mf_", "dde_", "fund_flow"],
    "融资融券": ["mtss", "fin_", "short_", "rzrq"],
    "概念板块": ["con_"],
    "事件": ["ev_", "tev", "ann_", "has_ann", "days_since_ann"],
    "基本面": ["pe", "pb", "eps", "bps", "revenue", "profit", "roe", "gross_margin", "debt_ratio", "total_assets"],
    "宏观": ["cn_pmi", "us_ism", "usdind", "usdcnh", "usdjpy", "cn2y", "cn5y", "us2y", "us5y"],
    "商品": ["cn_aluminum", "cn_copper", "cn_gold", "cn_nickel", "cn_silver", "cn_tin", "cn_zinc", "cn_commodity"],
    "外盘指数": ["a50_", "sp_futures", "dj_futures", "nq_futures", "sox_"],
    "链主": ["leader_", "has_leader"],
}
FEATURE_SEL_PATH = WEB_DIR / "data" / "feature_selection.json"


# ── 全局状态 ────────────────────────────────────────────────────────────────
STATE: dict[str, Any] = {
    "running": False,
    "progress": 0.0,
    "last": None,
    "result": None,
    "render_id": None,
    "error": None,
    "selected_features": None,
    "features_loaded": False,
    "cancelled": False,
    "pause_event": threading.Event(),
    "cancel_event": threading.Event(),
    "live": {},  # 训练中实时累积序列 (dates/ret/ic/sharpe_sofar/day_n/total_days)
    "total_cost": 0.0007,  # 默认 0.07% (买0.03+卖0.03+滑0.01)
    "initial_capital": 2_000_000,  # 默认200万, on_train 时更新
}

# 元素引用 (build 时赋值)
PROG_BAR = None
PROG_LABEL = None
BTN_TRAIN = BTN_PAUSE = BTN_CANCEL = None
RESULT_CONTAINER = None
# 结果区持久子元素 (原地更新 .figure/.text, 防抖 + 训练完图不消失)
CARD_LABELS = []        # 6 张指标卡的数值标签引用
COMBO_FIG = None        # 三合一图 (累计/回撤/每日IC) 持久 plotly
WEIGHTS_FIG = None      # 特征权重 持久 plotly
EXTRA_BOX = None        # 最终专属: 滚动Sharpe + 方向性α (仅完成时重建)
HOLDINGS_BOX = None     # 每日持仓明细表 (仅完成时重建)
HOLD_LABEL = None       # 当日持仓 (live + final, 取消按钮右侧)
REC_LABEL = None        # 次日推荐 (final, 取消按钮右侧)
INFO_LABEL = None       # 参数信息行 (final)
TABS = None
T_TRAIN = T_POOL = T_FEAT = T_CMP = T_EVENT = None

# 关注圈 / 自选股 相关引用
POOL_GRID = POOL_DIAG = None
WL_GRID = WL_DIAG = None
# 特征页引用
FEAT_GRID = FEAT_HIST = FEAT_SELECT = None
FEAT_ALL = []
FEAT_ROWS = []


# ── 工具函数 ────────────────────────────────────────────────────────────────
def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(**DARK_LAYOUT)
    return fig


def metric_card(title: str, value: str) -> None:
    with ui.card().classes("p-3 bg-slate-800 rounded-lg"):
        ui.label(title).classes("text-xs text-gray-400")
        ui.label(value).classes("text-xl font-bold text-teal-300")


# ── 代码→名称缓存 (持仓/推荐输出中文名) ──────────────────────────────────────
_NAME_CACHE: Optional[dict] = None


def code2name(code: str) -> str:
    global _NAME_CACHE
    if _NAME_CACHE is None:
        try:
            _NAME_CACHE = _stock_name_map()
        except Exception:
            _NAME_CACHE = {}
    key = str(code).strip().upper()
    return _NAME_CACHE.get(key, key)


def _make_cards(box, titles: list[str]) -> list:
    """在 box 内建 N 张持久指标卡, 返回数值标签引用列表 (原地更新)."""
    labels = []
    with box:
        for t in titles:
            with ui.card().classes("p-3 bg-slate-800 rounded-lg"):
                ui.label(t).classes("text-xs text-gray-400")
                labels.append(ui.label("—").classes("text-xl font-bold text-teal-300"))
    return labels


def _dim_of(feat: str) -> str:
    for dim, pats in FEATURE_DIMS.items():
        if any(feat.startswith(p) for p in pats):
            return dim
    return "其他"


import re as _re
# 特征含义: 基础名→中文说明; _ma5/_ma20 后缀=均值, _chg_Nd=涨跌幅 由 _meaning_of 解析
_FEATURE_BASE_MEANING = {
    # 技术面
    "ret_1d": "1日收益率", "ret_2d": "2日收益率", "ret_5d": "5日收益率",
    "ma5_pct": "收盘价/5日均线偏离%", "ma20_pct": "收盘价/20日均线偏离%",
    "macd": "MACD(DIF-DEA)", "macd_signal": "MACD信号线DEA", "macd_hist": "MACD柱状",
    "rsi_14": "14日RSI",
    "atr": "14日ATR绝对波动", "atr_pct": "ATR/收盘价%",
    "vol_ma5": "5日成交量均值", "vol_ma20": "20日成交量均值",
    "vol_ratio": "量比(当日量/5日均量)", "vol_change_1d": "成交量1日变化率",
    "pos_5": "价格在5日高低区间位置", "pos_20": "价格在20日高低区间位置",
    # 资金流
    "mf_net_1d": "主力净流入额", "mf_net_z": "主力净流入z分",
    "mf_pct_1d": "主力净流入占成交比", "mf_pct_z": "主力净流入占比z分",
    "mf_signal": "资金流合成信号",
    "dde_net_1d": "DDE大单净额", "dde_net_z": "DDE大单净额z分",
    "fund_flow_1d": "资金净流入", "fund_flow_z": "资金净流入z分",
    # 融资融券
    "mtss_1d": "融资融券余额变化", "mtss_z": "融资融券余额z分",
    # 事件
    "ev_bull_5d": "5日利好事件数", "ev_bear_5d": "5日利空事件数",
    "ev_net_5d": "5日净事件分", "ev_decay_n_5d": "5日衰减事件数",
    "ev_p0_5d": "5日P0级(重大)事件数", "ev_p1_5d": "5日P1级事件数", "ev_p2_5d": "5日P2级事件数",
    "tev_bull_5d": "板块5日利好事件数", "tev_bear_5d": "板块5日利空事件数",
    "tev_net_5d": "板块5日净事件分", "tev_decay_n_5d": "板块5日衰减事件数",
    "tev_p0_5d": "板块5日P0级事件数", "tev_p1_5d": "板块5日P1级事件数", "tev_p2_5d": "板块5日P2级事件数",
    "tev_decay_bull": "板块利好衰减", "tev_decay_bear": "板块利空衰减",
    "tev_all_bull_5d": "全板块5日利好", "tev_all_bear_5d": "全板块5日利空",
    "tev_all_net_5d": "全板块5日净事件分", "tev_all_decay_n_5d": "全板块5日衰减事件数",
    "tev_all_decay_bull": "全板块利好衰减", "tev_all_decay_bear": "全板块利空衰减",
    "ann_count": "当日公告数", "ann_5d": "5日公告数", "has_ann": "是否有公告",
    "days_since_ann": "距上次公告天数",
    # 基本面
    "pe": "市盈率", "pb": "市净率", "eps": "每股收益", "bps": "每股净资产",
    "revenue": "营业收入", "profit": "净利润", "roe": "净资产收益率",
    "gross_margin": "毛利率", "debt_ratio": "资产负债率", "total_assets": "总资产",
    # 宏观
    "cn_pmi": "中国制造业PMI", "us_ism_pmi": "美国ISM制造业PMI",
    "usdind": "美元指数", "usdcnh": "美元兑离岸人民币", "usdjpy": "美元兑日元",
    "cn2y": "中国2年期国债收益率", "cn5y": "中国5年期国债收益率",
    "us2y": "美国2年期国债收益率", "us5y": "美国5年期国债收益率",
    # 商品 (基础名, 实际特征为 xxx_chg_Nd)
    "cn_aluminum": "沪铝", "cn_copper": "沪铜", "cn_gold": "沪金",
    "cn_nickel": "沪镍", "cn_silver": "沪银", "cn_tin": "沪锡",
    "cn_zinc": "沪锌", "cn_commodity_idx": "Wind商品指数",
    # 外盘指数
    "a50_futures": "富时A50期货", "sp_futures": "标普500期货",
    "dj_futures": "道琼斯期货", "nq_futures": "纳斯达克期货", "sox": "费城半导体指数",
    # 链主
    "leader_binding_sum": "链主绑定强度", "leader_count": "链主数量",
    "leader_exp": "链主曝光度", "has_leader": "是否为链主",
    # 概念板块 (con_)
    "con_ret_1d": "概念板块日均涨跌幅", "con_vol": "概念板块总成交量",
    "con_amount": "概念板块总成交额", "con_mf_net": "概念板块主力净流入",
    "con_mf_pct": "概念板块主力净流入占比", "con_dde_net": "概念板块DDE净额",
}


def _meaning_of(feat: str) -> str:
    """特征含义: 处理 _ma5/_ma20 均值后缀 + _chg_Nd 涨跌幅.
    仅当去掉后缀后仍可解析(在字典里 或 是 chg 特征)才视作后缀,
    避免把 vol_ma5 这类"本身含_ma5的基础特征"误拆成 vol+_ma5."""
    sfx = ""
    base = feat
    for s, lbl in (("_ma20", "(20日均值)"), ("_ma5", "(5日均值)")):
        if feat.endswith(s):
            cand = feat[:-len(s)]
            if cand in _FEATURE_BASE_MEANING or _re.match(r"^(.+?)_chg_(\d)d$", cand):
                base, sfx = cand, lbl
            break
    m = _re.match(r"^(.+?)_chg_(\d)d$", base)
    if m:
        und = _FEATURE_BASE_MEANING.get(m.group(1), m.group(1))
        return f"{und}{int(m.group(2))}日涨跌幅{sfx}"
    return _FEATURE_BASE_MEANING.get(base, base) + sfx


def default_feature_set() -> list[str]:
    path = latest_training_data()
    cols = pq.read_schema(path).names
    return [
        c for c in cols
        if c not in _SKIP and "_21d" not in c
        and not c.endswith("_cross") and c not in _LEAK
        and c not in ("date", "code")
        and c not in _EXCLUDED
    ]


def load_feature_selection() -> list[str]:
    if FEATURE_SEL_PATH.exists():
        try:
            data = json.loads(FEATURE_SEL_PATH.read_text(encoding="utf-8"))
            feats = data.get("features")
            if feats:
                return list(feats)
        except Exception:
            pass
    return default_feature_set()


def save_feature_selection(feats: list[str]) -> None:
    FEATURE_SEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_SEL_PATH.write_text(
        json.dumps({"features": feats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _col_w(rows: list[dict], field: str, base: int, cap: int) -> int:
    """按字段最大内容长度估算列宽(单行完整显示, 不换行), 夹在 [base, cap] 间."""
    mx = 0
    for r in rows:
        v = r.get(field)
        if v is not None:
            mx = max(mx, len(str(v)))
    return max(base, min(cap, mx * 8 + 24))


def _clean_rows(rows: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for r in rows:
        code = str(r.get("code", "")).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append({
            "code": code,
            "name": stock_name(code, str(r.get("name", "")).strip()),
            "theme": str(r.get("theme", "未分类")).strip() or "未分类",
        })
    return out


# ── 关注圈 主题 + 日/月/年 涨幅 (展示用, 不入库) ──
_THEME_LABELS = {"AI": "AI算力", "Machine": "工业母机", "Grid": "电网设备",
                 "Space": "航天军工", "Pharma": "医药", "Robot": "机器人"}
_theme_pools_cache: dict = {}


def _theme_pools_map() -> dict:
    """code6 -> 主题(中文), 来自 universe/theme_pools.json"""
    global _theme_pools_cache
    if _theme_pools_cache:
        return _theme_pools_cache
    p = processed_dir().parent / "universe" / "theme_pools.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        for t, codes in (d.get("themes") or {}).items():
            lbl = _THEME_LABELS.get(t, t)
            for c in codes:
                _theme_pools_cache[str(c)] = lbl
    except Exception:
        pass
    return _theme_pools_cache


_pool_chg_cache: dict = {}


def _pool_chg(code6: str) -> tuple:
    """(日涨幅%, 月涨幅%, 年涨幅%) from raw kline 收盘价; 月≈20交易日, 年≈250交易日. None=数据不足."""
    if code6 in _pool_chg_cache:
        return _pool_chg_cache[code6]
    p = processed_dir().parent / "raw" / "kline" / f"{code6}.parquet"
    try:
        d = pd.read_parquet(p)
        col = "收盘价" if "收盘价" in d.columns else ("close" if "close" in d.columns else None)
        if col is None:
            res = (None, None, None)
        else:
            c = pd.to_numeric(d[col], errors="coerce").dropna().values
            if len(c) < 2:
                res = (None, None, None)
            else:
                cur = c[-1]
                def pct(ref):
                    return round((cur / ref - 1) * 100, 2) if ref else None
                res = (pct(c[-2]),
                       pct(c[-21] if len(c) >= 21 else None),
                       pct(c[-251] if len(c) >= 251 else None))
    except Exception:
        res = (None, None, None)
    _pool_chg_cache[code6] = res
    return res


def _enrich_pool_rows(rows: list[dict]) -> None:
    """原地填充 theme(未分类时从theme_pools) + 日/月/年涨幅."""
    tmap = _theme_pools_map()
    for r in rows:
        c6 = str(r["code"]).split(".")[0]
        if r.get("theme", "未分类") in ("", "未分类"):
            r["theme"] = tmap.get(c6, "未分类")
        d1, m20, y250 = _pool_chg(c6)
        r["日涨幅"], r["月涨幅"], r["年涨幅"] = d1, m20, y250


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"watchlist": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_diag(container, df: pd.DataFrame) -> None:
    container.clear()
    if df is None or len(df) == 0:
        with container:
            ui.label("无数据")
        return
    show = df[["code", "name", "数据起始", "数据密集", "建议训练起始", "状态"]]
    with container:
        ui.table(
            rows=show.to_dict("records"),
            pagination=10,
        ).classes("w-full")


# ── 结果可视化 (训练&回测 默认页 / 结果对比 共用) ───────────────────────────
def _build_combo_fig(dates, cum_acc, dd, ic, title, x_range=None):
    """三合一图 (账户金额万元 + 回撤% 左轴, 每日IC 右轴) — 持久元素原地更新用.
    cum_acc: 实际账户金额 (元), 基于初始200万元复利
    x_range: 固定 x 轴范围 (如 [训练起始, 最后交易日]), 避免训练中坐标轴缩放."""
    # 转换为万元显示
    acc_wan = cum_acc / 10_000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=acc_wan, name="账户金额(万元)",
        line=dict(color="#00d4aa", width=2), yaxis="y1"))
    fig.add_trace(go.Scatter(
        x=dates, y=dd, name="回撤%",
        line=dict(color="#f25c54", width=1), fill="tozeroy",
        fillcolor="rgba(242,92,84,0.18)", yaxis="y1"))
    fig.add_trace(go.Scatter(
        x=dates, y=ic, name="每日IC",
        mode="markers", marker=dict(color="#89b4fa", size=3), yaxis="y2"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title, height=440,
        yaxis=dict(title="账户金额(万元) / 回撤(%)", side="left"),
        yaxis2=dict(title="每日IC", side="right", overlaying="y", showgrid=False),
        showlegend=False,
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range, autorange=False)
    return style(fig).update_layout(margin=dict(l=55, r=55, t=44, b=40))


def _reset_result_view() -> None:
    """训练开始前重置持久结果视图 (旧结果不残留, 避免图消失/串味)."""
    for lbl in CARD_LABELS:
        lbl.text = "—"
    if COMBO_FIG is not None:
        COMBO_FIG.update_figure( go.Figure())
    if WEIGHTS_FIG is not None:
        WEIGHTS_FIG.update_figure( go.Figure())
    if INFO_LABEL is not None:
        INFO_LABEL.text = "训练中..."
    if HOLD_LABEL is not None:
        HOLD_LABEL.text = ""
    if REC_LABEL is not None:
        REC_LABEL.text = ""
    if EXTRA_BOX is not None:
        EXTRA_BOX.clear()


def _render_final(obj) -> None:
    """训练完成后渲染 (原地更新持久元素, 图永不消失)."""
    if not CARD_LABELS:
        return
    # 统一为 dict: TrainResult 走 model_dump, fallback 的 dict 直接用
    d = obj.model_dump() if hasattr(obj, "model_dump") else (obj if isinstance(obj, dict) else {})
    daily = d.get("daily_returns", []) or []
    fi = d.get("feature_importance", []) or []

    # 从 DB 加载时, holdings 在 daily_holdings 表中而不在 daily_returns 中, 需合并
    dh_map: dict[str, list] = {}
    for h in d.get("daily_holdings", []) or []:
        dh_map.setdefault(h["date"], []).append({
            "code": h["code"], "pred_score": h.get("pred_score", 0.0)
        })
    for dd in daily:
        if not dd.get("holdings") and dd["date"] in dh_map:
            dd["holdings"] = dh_map[dd["date"]]

    def g(key, default=0.0):
        return d.get(key, default)

    # 由日收益明细推算累计账户金额/回撤/IC均值 (与最终指标一致)
    # top_ret 为原始超额收益(不含成本), 成本已在 trainer 按精确模型扣除
    # cum_return 已是净账户金额
    if daily:
        rdf = pd.DataFrame(daily)
        capital = g("initial_capital", _INITIAL_CAPITAL)
        ic_valid = rdf["ic"].dropna()
        ic_mean = float(ic_valid.mean()) if len(ic_valid) else 0.0
        final_acc = float(rdf["cum_return"].iloc[-1]) if "cum_return" in rdf else capital
        total_cost_rmb = rdf.get("cost_rmb", pd.Series(0, index=rdf.index)).sum()
        # 回撤基于 cum_return 序列
        cum_vals = rdf["cum_return"].values
        if len(cum_vals) > 1:
            peak = np.maximum.accumulate(cum_vals)
            dd_vals = (cum_vals / peak - 1) * 100
            max_dd = float(dd_vals.min())
        else:
            dd_vals = np.array([0])
            max_dd = 0.0
        # 净收益率序列 (用于图表)
        net_ret = rdf.get("top_ret_net", rdf["top_ret"])
        cum_full = (1 + net_ret).cumprod()
        cum_acc_chart = capital * cum_full
    else:
        rdf = None
        capital = _INITIAL_CAPITAL
        final_acc = ic_mean = max_dd = total_cost_rmb = 0.0
        dd_vals = np.array([0])
        cum_acc_chart = pd.Series(dtype=float)

    # 指标卡 (原地更新) — 按实际金额显示
    CARD_LABELS[0].text = f"{g('sharpe_raw'):.2f}"
    CARD_LABELS[1].text = f"¥{final_acc/1e4:.0f}万"
    CARD_LABELS[2].text = f"{ic_mean:.4f}"
    CARD_LABELS[3].text = f"{max_dd:.1f}%"
    CARD_LABELS[4].text = f"{g('annual_return') * 100:.1f}%"
    CARD_LABELS[5].text = f"{g('win_rate') * 100:.1f}%"

    if INFO_LABEL is not None:
        INFO_LABEL.text = (
            f"Top{g('top_n')} | 特征{g('n_features')} | "
            f"训练起始 {str(g('train_start', ''))[:10]} | "
            f"回测 {str(g('test_start', ''))[:10]}~{g('test_end') or '最新'} | "
            f"成本 买{g('buy_pct', 0)}% 卖{g('sell_pct', 0)}% 滑{g('slip_pct', 0)}% "
            f"(累计¥{total_cost_rmb:,.0f}) | {g('n_days')}天"
        )

    # 三合一图 (原地更新 .figure)
    if COMBO_FIG is not None:
        if daily:
            # 过滤虚拟末日日 (超出 date_range 的), 避免污染横轴
            xr = g("date_range")
            if xr and len(xr) == 2:
                _dm = (pd.to_datetime(rdf["date"]) >= pd.Timestamp(xr[0])) & \
                      (pd.to_datetime(rdf["date"]) <= pd.Timestamp(xr[1]))
                _rdf_plot = rdf[_dm]
                _cum_plot = cum_acc_chart.values[_dm.values] if hasattr(cum_acc_chart, 'values') else \
                            cum_acc_chart[_dm.values]
                _dd_plot = dd_vals[_dm.values]
                _ic_plot = rdf["ic"].tolist()
                _ic_plot = [_ic_plot[i] for i in range(len(_ic_plot)) if _dm.iloc[i]]
            else:
                _rdf_plot = rdf
                _cum_plot = cum_acc_chart.values
                _dd_plot = dd_vals
                _ic_plot = rdf["ic"].tolist()
            COMBO_FIG.update_figure( _build_combo_fig(
                pd.to_datetime(_rdf_plot["date"]), _cum_plot, _dd_plot, _ic_plot,
                f"账户金额 / 回撤 / 每日IC (年化 {g('annual_return') * 100:.1f}%)",
                x_range=xr or None))
        else:
            COMBO_FIG.update_figure( go.Figure())

    # 特征权重 (原地更新)
    if WEIGHTS_FIG is not None:
        if fi:
            imp = pd.DataFrame(fi).head(20)
            fig_fi = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h"))
            fig_fi.update_layout(title="特征重要性 Top20", height=480)
            WEIGHTS_FIG.update_figure( style(fig_fi))
        else:
            WEIGHTS_FIG.update_figure( go.Figure())

    # 持仓 / 次日推荐 (最后交易日, 输出中文名)
    if daily and HOLD_LABEL is not None and REC_LABEL is not None:
        last = daily[-1]
        hnames = "、".join(
            f"{code2name(h['code'])}({h['pred_score']:+.3f})" for h in last.get("holdings", []))
        rnames = "、".join(
            f"{code2name(h['code'])}({h['pred_score']:+.3f})" for h in last.get("next_rec", []))
        HOLD_LABEL.text = f"📋 最后交易日持仓 ({last['date']}): {hnames}"
        if rnames:
            REC_LABEL.text = f"▶ 次日推荐 (用于下一交易日): {rnames}"
        else:
            REC_LABEL.text = "▶ 次日推荐: —"

    # 最终专属图: 滚动20日Sharpe + 方向性α (仅完成时重建)
    if EXTRA_BOX is not None:
        EXTRA_BOX.clear()
        with EXTRA_BOX:
            if daily and rdf is not None and len(rdf) >= 20:
                roll = rdf["top_ret"].rolling(20)
                rs = (roll.mean() / roll.std() * np.sqrt(252)).values
                fig_rs = go.Figure(go.Scatter(
                    x=pd.to_datetime(rdf["date"]), y=rs, name="滚动20日Sharpe",
                    line=dict(color="#f9e2af", width=2)))
                fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_rs.update_layout(title="滚动20日Sharpe", height=180)
                ui.plotly(style(fig_rs))
            render_directional_alpha(EXTRA_BOX)

    # 每日持仓明细表 (仅完成时重建)
    if HOLDINGS_BOX is not None:
        HOLDINGS_BOX.clear()
        with HOLDINGS_BOX:
            if daily:
                ui.label("📅 每日持仓明细 (T日收盘等额买入Top3, T+1收盘换仓)").classes(
                    "text-sm text-gray-300 mt-2")
                rows = []
                for d in daily:
                    htxt = "、".join(
                        f"{code2name(h['code'])}" for h in d.get("holdings", []))
                    # 前日得分 = 前一天模型对今日Top3的预测得分
                    scores = "、".join(
                        f"{h['pred_score']:+.4f}" for h in d.get("holdings", []))
                    ic_v = d.get("ic")
                    rows.append({
                        "日期": d["date"],
                        "当日持仓": htxt,
                        "前日得分": scores,
                        "当日收益%": f"{d['top_ret'] * 100:.2f}%",
                        "IC": f"{ic_v:.4f}" if ic_v is not None else "—",
                    })
                cols = [
                    {"name": "日期", "label": "日期", "field": "日期", "align": "left"},
                    {"name": "当日持仓", "label": "当日持仓", "field": "当日持仓",
                     "align": "left", "width": "240px"},
                    {"name": "前日得分", "label": "前日得分", "field": "前日得分",
                     "align": "left", "width": "200px"},
                    {"name": "当日收益%", "label": "当日收益%", "field": "当日收益%",
                     "align": "right"},
                    {"name": "IC", "label": "IC", "field": "IC", "align": "right"},
                ]
                ui.table(rows=rows, columns=cols, pagination=20).classes("w-full")


def _render_live(lv: dict) -> None:
    """训练进行中实时联动: 原地更新持久元素 (三合一图 + 权重 + 持仓名 + 指标卡)."""
    if not CARD_LABELS or COMBO_FIG is None:
        return
    dates = pd.to_datetime(lv["dates"])
    ret = pd.Series(lv["ret"])
    ic = lv["ic"]
    capital = STATE.get("initial_capital", _INITIAL_CAPITAL)
    cum_full = (1 + ret).cumprod()
    cum_acc = capital * cum_full  # 实际账户金额(元)
    dd = (cum_full / cum_full.expanding().max() - 1).values * 100
    ic_valid = [x for x in ic if x is not None]
    ic_mean = float(np.mean(ic_valid)) if ic_valid else 0.0
    final_acc = float(cum_acc.iloc[-1]) if len(cum_acc) else capital
    max_dd = float(dd.min()) if len(dd) else 0.0
    sharpe = float(lv.get("sharpe_sofar", 0.0))
    n = len(ret)
    annual = float(((1 + ret).prod()) ** (252 / n) - 1) if n > 1 else 0.0

    # 指标卡 (原地更新) — 按实际金额显示
    CARD_LABELS[0].text = f"{sharpe:.2f}"
    CARD_LABELS[1].text = f"¥{final_acc/1e4:.0f}万"
    CARD_LABELS[2].text = f"{ic_mean:.4f}"
    CARD_LABELS[3].text = f"{max_dd:.1f}%"
    CARD_LABELS[4].text = f"{annual * 100:.1f}%"
    # 训练中实时胜率
    win_cnt = int((ret > 0).sum())
    CARD_LABELS[5].text = f"{win_cnt}/{n}天 ({win_cnt/n*100:.1f}%)" if n > 0 else "—"

    # 三合一图 (原地更新 .figure)
    COMBO_FIG.update_figure( _build_combo_fig(
        dates, cum_acc.values, dd, ic,
        f"账户金额 / 回撤 / 每日IC (训练中... {lv.get('day_n', 0)}/{lv.get('total_days', 0)}天)",
        x_range=lv.get("x_range")))

    # 特征权重 (原地更新, 训练中累积 Top15)
    if WEIGHTS_FIG is not None:
        fi = lv.get("feature_importance") or []
        if fi:
            imp = pd.DataFrame(fi).head(15)
            fig_w = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h"))
            fig_w.update_layout(title="特征权重 (训练中累积 Top15)", height=420)
            WEIGHTS_FIG.update_figure( style(fig_w))
        else:
            WEIGHTS_FIG.update_figure( go.Figure())

    # 当日持仓 (输出中文名)
    if HOLD_LABEL is not None:
        hold = lv.get("holdings") or []
        names = "、".join(
            f"{code2name(h['code'])}({h['pred_score']:+.3f})" for h in hold)
        HOLD_LABEL.text = f"📋 当日持仓 ({lv.get('date')}): {names}"
    # 次日推荐 (输出中文名, 模型对下一交易日的实时预测 topN)
    if REC_LABEL is not None:
        nxt = lv.get("next_rec") or []
        if nxt:
            rnames = "、".join(
                f"{code2name(h['code'])}({h['pred_score']:+.3f})" for h in nxt)
            REC_LABEL.text = f"▶ 次日推荐 ({lv.get('date')} 信号 → 下一交易日): {rnames}"
        else:
            REC_LABEL.text = "▶ 次日推荐: 训练中补算 (下一交易日信号)"


def render_directional_alpha(container) -> None:
    recs = discover_backtests()
    alpha = directional_alpha(recs)
    if not alpha:
        return
    fig = go.Figure(go.Bar(
        x=[f"v{v}" for v in alpha.keys()],
        y=list(alpha.values()),
        marker_color=["#00d4aa" if v >= 0 else "#f25c54" for v in alpha.values()],
        text=[f"{v:+.2f}" for v in alpha.values()], textposition="outside",
    ))
    fig.update_layout(title="方向性α照妖镜 (正向−反向 Sharpe, 反向亏钱才说明正向是真α)", height=300)
    with container:
        ui.plotly(style(fig))


def _render_fallback(rec) -> None:
    """无本地训练记录时的预计算快照 (原地更新持久元素)."""
    if not CARD_LABELS:
        return
    s = rec["summary"]
    pnl_pct = s.get('cum_return_pct', 0)
    final_acc = _INITIAL_CAPITAL * (1 + pnl_pct / 100)
    CARD_LABELS[0].text = f"{s.get('sharpe', 0):.2f}"
    CARD_LABELS[1].text = f"¥{final_acc/1e4:.0f}万"
    CARD_LABELS[2].text = f"{s.get('ic_mean', 0):.4f}"
    CARD_LABELS[3].text = f"{s.get('max_dd_pct', 0):.1f}%"
    CARD_LABELS[4].text = f"{s.get('annualized_return_pct', 0):.1f}%"
    CARD_LABELS[5].text = f"{s.get('win_rate_pct', 0):.1f}%"
    if INFO_LABEL is not None:
        INFO_LABEL.text = (
            f"📭 预计算快照 | {rec['name']} | 区间 {rec['period']} | "
            f"总成本 {s.get('total_cost_est_pct', 0):.2f}%"
        )
    if COMBO_FIG is not None:
        COMBO_FIG.update_figure( go.Figure())
    if WEIGHTS_FIG is not None:
        WEIGHTS_FIG.update_figure( go.Figure())
    if HOLD_LABEL is not None:
        HOLD_LABEL.text = ""
    if REC_LABEL is not None:
        REC_LABEL.text = ""
    if EXTRA_BOX is not None:
        EXTRA_BOX.clear()
        with EXTRA_BOX:
            ui.label("📭 尚无本地训练记录 — 下方为预计算 Walk-Forward 回测快照 (运行训练&回测写入完整曲线)")
            render_directional_alpha(EXTRA_BOX)


# ── 训练执行 (线程 + 进度轮询) ───────────────────────────────────────────────
def _run_training(params: TrainParams) -> None:
    STATE["running"] = True
    STATE["progress"] = 0.0
    STATE["last"] = None
    STATE["error"] = None
    STATE["result"] = None
    STATE["cancelled"] = False
    STATE["live"] = {}

    def on_progress(d: dict) -> None:
        print(f"[ON_PROGRESS] warmup={d.get('warmup')} day_n={d.get('day_n')} dates={len(d.get('live_dates', []))}", flush=True)
        if d.get("warmup"):
            # 预热期: 只训练不输出图, 给 UI 状态提示 (回测起始日前不画图)
            STATE["progress"] = 0.0
            STATE["last"] = d
            STATE["live"] = {}
            STATE["_warmup_info"] = d
            return
        STATE["_warmup_info"] = None
        STATE["progress"] = d["progress_pct"]
        STATE["last"] = d
        STATE["live"] = {
            "dates": d.get("live_dates", []),
            "ret": d.get("live_ret", []),
            "ic": d.get("live_ic", []),
            "sharpe_sofar": d.get("sharpe_sofar", 0.0),
            "day_n": d.get("day_n", 0),
            "total_days": d.get("total_days", 0),
            "feature_importance": d.get("feature_importance", []),
            "holdings": d.get("holdings", []),
            "next_rec": d.get("next_rec", []),
            "date": d.get("date"),
            "x_range": d.get("x_range"),  # 回测终止日范围, 固定x轴
            "_rendered_day": STATE["live"].get("_rendered_day", -1),
        }

    try:
        result = train(
            params, progress_callback=on_progress,
            pause_event=STATE["pause_event"], cancel_event=STATE["cancel_event"],
        )
        if STATE["cancel_event"].is_set():
            STATE["cancelled"] = True
        else:
            STATE["result"] = result
    except _TrainCancelled as e:  # noqa: BLE001
        STATE["cancelled"] = True
        STATE["error"] = None
    except Exception as e:  # noqa: BLE001
        STATE["error"] = str(e)
        import traceback as _tb
        _tb.print_exc()
    finally:
        STATE["running"] = False  # 确保无论如何都解除锁定


def _poll_progress() -> None:
    if STATE["running"]:
        if PROG_BAR is not None:
            PROG_BAR.value = STATE["progress"] / 100.0
        if PROG_LABEL is not None and STATE["last"]:
            d = STATE["last"]
            if d.get("warmup"):
                PROG_LABEL.text = (
                    f"🟡 预热训练中：[{d.get('train_start')}]→[{d.get('test_start')}] "
                    f"共{d.get('warmup_days')}天 (回测起始后开始输出图)"
                )
            else:
                ic = d.get("ic")
                ic_s = f"{ic:.4f}" if ic is not None else "—"
                sharpe_s = f"{d['sharpe_sofar']:.2f}" if d.get('day_n', 0) >= 20 else "—"
                PROG_LABEL.text = (
                    f"{d['day_n']}/{d['total_days']}天 ({d['progress_pct']:.0f}%) | "
                    f"Sharpe(sofar)={sharpe_s} | IC={ic_s} | "
                    f"耗时{d['elapsed_seconds']:.0f}s"
                )
        # 训练中实时联动: 原地更新持久元素 (按天节流做防抖, 避免每轮询全量重绘)
        # 关键修复: 仅当 _render_live 成功才更新 _rendered_day 节流位;
        # 否则一旦首次渲染抛异常(被提前置位)会永久跳过后续联动, 表现为"图冻结".
        lv = STATE.get("live")
        if lv and lv.get("dates") and lv.get("_rendered_day") != lv.get("day_n"):
            print(f"[LIVE_RENDER] day_n={lv.get('day_n')}", flush=True)
            try:
                _render_live(lv)
                lv["_rendered_day"] = lv.get("day_n")
            except Exception as _ex:  # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                ui.notify(f"⚠️ 实时联动渲染异常: {_ex}", type="warning", timeout=5000)
    # 新结果 -> 原地渲染 (图永不消失)
    if STATE["result"] is not None and STATE.get("render_id") != id(STATE["result"]):
        STATE["render_id"] = id(STATE["result"])
        _render_final(STATE["result"])
        if PROG_BAR is not None:
            PROG_BAR.value = 1.0
        if PROG_LABEL is not None:
            PROG_LABEL.text = "✅ 训练完成"
    # 错误提示
    if STATE.get("error") and not STATE.get("_err_shown"):
        STATE["_err_shown"] = True
        ui.notify(f"❌ 训练失败: {STATE['error']}", type="negative", timeout=8000)

    # 训练结束 -> 恢复按钮 (主线程安全: _poll_progress 由 ui.timer 调度)
    if (not STATE["running"]) and STATE.get("_was_running"):
        STATE["_was_running"] = False
        if BTN_TRAIN is not None:
            BTN_TRAIN.set_visibility(True)
        if BTN_PAUSE is not None:
            BTN_PAUSE.set_visibility(False)
        if BTN_CANCEL is not None:
            BTN_CANCEL.set_visibility(False)
        if STATE.get("cancelled"):
            STATE["cancelled"] = False
            if PROG_BAR is not None:
                PROG_BAR.value = 0.0
            if PROG_LABEL is not None:
                PROG_LABEL.text = "⏹ 训练已取消"
            if HOLD_LABEL is not None:
                HOLD_LABEL.text = ""
            if REC_LABEL is not None:
                REC_LABEL.text = ""
            if RESULT_CONTAINER is not None:
                RESULT_CONTAINER.clear()
                with RESULT_CONTAINER:
                    ui.label("⏹ 训练已取消，未生成结果").classes("text-gray-400")


# ── 页面 1: 训练 & 回测 (默认) ──────────────────────────────────────────────
def build_training_page() -> None:
    global PROG_BAR, PROG_LABEL, RESULT_CONTAINER, BTN_TRAIN, BTN_PAUSE, BTN_CANCEL, HOLD_LABEL, REC_LABEL, HOLD_BOX
    tp_saved = {}
    raw = get_settings("train_params")
    if raw:
        try:
            tp_saved = json.loads(raw)
        except Exception:
            pass
    mp_saved = {}
    rawm = get_settings("model_params")
    if rawm:
        try:
            mp_saved = json.loads(rawm)
        except Exception:
            pass

    earliest = str(earliest_train_date())

    def _is_date(v: str, allow_empty: bool = False) -> bool:
        v = (v or "").strip()
        if not v:
            return allow_empty
        try:
            date.fromisoformat(v)
            return True
        except Exception:
            return False

    def _num_to_date(n: int) -> str:
        """YYYYMMDD 整数 -> YYYY-MM-DD."""
        return datetime.strptime(f"{int(n):08d}", "%Y%m%d").date().isoformat()

    def _date_to_num(s: str) -> int:
        """YYYY-MM-DD -> YYYYMMDD 整数 (直观显示真实日期, 与 TOPN 同控件)."""
        return int(datetime.strptime(s, "%Y-%m-%d").date().strftime("%Y%m%d"))

    def _date_input(label, value_str, placeholder, allow_empty) -> "ui.number":
        """日期复用 ui.number 原生 spinner 控件 (与 TOPN 完全相同), 值=YYYYMMDD 直观显示真实日期."""
        init = _date_to_num(value_str) if (value_str and _is_date(value_str, allow_empty)) else None
        return ui.number(label=label, value=init, min=20000101, max=29991231, step=1)

    def _num_input(label, value, min, max, step, style) -> "ui.number":
        """数字输入框 (浮动标签), 原生浏览器 spinner 三角 (与原来 TOPN 一致)."""
        num = ui.number(label=label, value=value, min=min, max=max, step=step)
        num.style(style)
        return num

    ui.label("🚀 训练 & 回测").classes("text-2xl font-bold mb-2")

    # ── 参数模态 (ui.dialog) ──
    param_dialog = ui.dialog()
    with param_dialog, ui.card().classes("w-[540px] p-5 bg-slate-800 shadow-lg"):
        ui.label("⚙ 参数设置").classes("text-lg font-bold mb-3")
        ui.label("回测区间").classes("text-sm font-semibold text-gray-300 mb-1")
        with ui.row().classes("gap-3 w-full items-end"):
            train_start = _date_input("训练起始", tp_saved.get("train_start", earliest), "YYYY-MM-DD", False)
            test_start = _date_input("回测起始", tp_saved.get("test_start", "2025-09-01"), "YYYY-MM-DD", False)
            test_end = _date_input("回测终止(留空=最新)", tp_saved.get("test_end") or "", "YYYY-MM-DD", True)

        ui.label("成本 & 持仓 & 采样").classes("text-sm font-semibold text-gray-300 mt-3 mb-1")
        with ui.row().classes("gap-3 w-full items-end flex-wrap"):
            buy = _num_input("买入%", tp_saved.get("buy_pct", 0.03), 0, 0.5, 0.001, "min-width:110px")
            sell = _num_input("卖出%", tp_saved.get("sell_pct", 0.03), 0, 0.5, 0.001, "min-width:110px")
            slip = _num_input("滑点%", tp_saved.get("slip_pct", 0.01), 0, 0.2, 0.001, "min-width:110px")
            capital_input = _num_input("初始资金(万)", tp_saved.get("initial_capital", 200), 10, 10000, 1, "min-width:130px")
            topn = _num_input("TopN", tp_saved.get("top_n", 3), 1, 10, 1, "min-width:100px")
            sampleint = _num_input("采样(日)", tp_saved.get("sample_interval", 5), 1, 20, 1, "min-width:100px")

        ui.label("训练集来源").classes("text-sm font-semibold text-gray-300 mt-3 mb-1")
        src_options = ["关注圈", "自选股"]
        src_val = tp_saved.get("universe_source", "关注圈")
        if src_val not in src_options:
            src_val = "关注圈"
        univ = ui.radio(src_options, value=src_val).props("inline dense")

        ui.label("模型参数").classes("text-sm font-semibold text-gray-300 mt-3 mb-1")
        with ui.row().classes("gap-3 w-full items-end"):
            nest = _num_input("n_estimators", mp_saved.get("n_estimators", 400), 50, 1000, 50, "min-width:140px")
            md = _num_input("max_depth", mp_saved.get("max_depth", 4), 2, 10, 1, "min-width:140px")
            lr = _num_input("learning_rate", mp_saved.get("learning_rate", 0.03), 0.001, 0.5, 0.005, "min-width:140px")
        with ui.row().classes("gap-3 w-full items-end"):
            nl = _num_input("num_leaves", mp_saved.get("num_leaves", 15), 7, 127, 8, "min-width:140px")
            ss = _num_input("subsample", mp_saved.get("subsample", 0.8), 0.5, 1.0, 0.05, "min-width:140px")
            cs = _num_input("colsample_bytree", mp_saved.get("colsample_bytree", 0.8), 0.5, 1.0, 0.05, "min-width:140px")
        with ui.row().classes("gap-2 justify-end mt-4"):
            ui.button("取消", on_click=param_dialog.close).props("flat")
            ui.button("确认", on_click=param_dialog.close).props("primary")

    # 特征信息 (底部小字)
    feat_info = ui.label("").classes("text-xs text-gray-500 mb-1")
    def refresh_feat_info() -> None:
        feats = STATE["selected_features"]
        n = len(feats) if feats else len(default_feature_set())
        feat_info.text = f"特征数: {n} (默认 v23 过滤集, 在『特征』页管理)"
    refresh_feat_info()

    # ── 训练按钮 ──
    def on_train() -> None:
        if STATE["running"]:
            return
        STATE["running"] = True  # 立即锁定，防止竞态双跑
        src = univ.value
        try:
            params = TrainParams(
                train_start=date.fromisoformat(_num_to_date(int(train_start.value))),
                test_start=date.fromisoformat(_num_to_date(int(test_start.value))),
                test_end=(date.fromisoformat(_num_to_date(int(test_end.value))) if test_end.value else None),
                buy_pct=float(buy.value), sell_pct=float(sell.value), slip_pct=float(slip.value),
                initial_capital=float(capital_input.value) * 10_000,  # 万元→元
                top_n=int(topn.value),
                universe_source=src,
                sample_interval=int(sampleint.value),
                n_estimators=int(nest.value), max_depth=int(md.value),
                learning_rate=float(lr.value), num_leaves=int(nl.value),
                subsample=float(ss.value), colsample_bytree=float(cs.value),
                min_child_samples=50, random_state=42, n_jobs=32,
                features=STATE["selected_features"],
            )
            STATE["total_cost"] = params.buy_pct / 100 + params.sell_pct / 100 + params.slip_pct / 100
            STATE["initial_capital"] = params.initial_capital
        except Exception as e:  # noqa: BLE001
            ui.notify(f"参数错误: {e}", type="negative")
            return
        # 持久化
        save_settings("train_params", json.dumps({
            "train_start": _num_to_date(int(train_start.value)), "test_start": _num_to_date(int(test_start.value)),
            "test_end": _num_to_date(int(test_end.value)) if test_end.value else "", "buy_pct": float(buy.value),
            "sell_pct": float(sell.value), "slip_pct": float(slip.value),
            "initial_capital": float(capital_input.value),
            "top_n": int(topn.value),
            "sample_interval": int(sampleint.value), "universe_source": src,
        }, ensure_ascii=False))
        save_settings("model_params", json.dumps({
            "n_estimators": int(nest.value), "max_depth": int(md.value),
            "learning_rate": float(lr.value), "num_leaves": int(nl.value),
            "subsample": float(ss.value), "colsample_bytree": float(cs.value),
            "min_child_samples": 50, "random_state": 42, "n_jobs": 32,
        }, ensure_ascii=False))
        STATE["_err_shown"] = False
        STATE["pause_event"].clear()
        STATE["cancel_event"].clear()
        STATE["_was_running"] = True
        if BTN_TRAIN is not None:
            BTN_TRAIN.set_visibility(False)
        if BTN_PAUSE is not None:
            BTN_PAUSE.set_visibility(True)
            BTN_PAUSE.text = "⏸ 暂停"
            BTN_PAUSE.props(remove="outline")
        if BTN_CANCEL is not None:
            BTN_CANCEL.set_visibility(True)
        _reset_result_view()  # 重置持久结果视图, 避免旧图残留/消失
        threading.Thread(target=_run_training, args=(params,), daemon=True).start()

    def on_pause() -> None:
        if not STATE["running"]:
            return
        if STATE["pause_event"].is_set():
            STATE["pause_event"].clear()
            if BTN_PAUSE is not None:
                BTN_PAUSE.text = "⏸ 暂停"
        else:
            STATE["pause_event"].set()
            if BTN_PAUSE is not None:
                BTN_PAUSE.text = "▶ 继续"

    def on_cancel() -> None:
        if not STATE["running"]:
            return
        STATE["cancel_event"].set()
        STATE["pause_event"].clear()  # 解除暂停, 让循环立即检查取消信号
        if BTN_PAUSE is not None:
            BTN_PAUSE.set_visibility(False)
        if BTN_CANCEL is not None:
            BTN_CANCEL.set_visibility(False)

    with ui.row().classes("items-center gap-3 w-full"):
        BTN_TRAIN = ui.button("▶️ 开始训练", on_click=on_train).props("primary")
        with ui.row().classes("gap-2"):
            BTN_PAUSE = ui.button("⏸ 暂停", on_click=on_pause).props("outline").set_visibility(False)
            BTN_CANCEL = ui.button("⏹ 取消", on_click=on_cancel).props("outline").set_visibility(False)
        # 持仓/推荐: 取消按钮右侧, 两行 (上为当日持仓, 下为次日推荐)
        HOLD_BOX = ui.column().classes("gap-0 ml-1")
        with HOLD_BOX:
            HOLD_LABEL = ui.label("").classes("text-sm text-gray-200")
            REC_LABEL = ui.label("").classes("text-sm text-amber-300 font-semibold")
        ui.space()
        ui.button("📊 特征管理 →", on_click=lambda: TABS.set_value(T_FEAT)).props("outline dense")
        ui.button("⚙ 参数设置", on_click=param_dialog.open).props("outline dense")

    PROG_BAR = ui.linear_progress(value=0).classes("w-full mt-2")
    PROG_LABEL = ui.label("").classes("text-sm text-gray-400")

    # ── 结果区 (持久子元素, 原地更新 .figure/.text: 防抖 + 训练完图不消失) ──
    global RESULT_CONTAINER, CARD_LABELS, COMBO_FIG, WEIGHTS_FIG, EXTRA_BOX, INFO_LABEL, HOLDINGS_BOX
    ui.separator()
    RESULT_CONTAINER = ui.column().classes("w-full")
    with RESULT_CONTAINER:
        CARDS_BOX = ui.grid(columns=6).classes("gap-3 w-full")
        CARD_LABELS = _make_cards(CARDS_BOX, [
            "Sharpe", "账户金额", "日均IC", "最大回撤%", "年化收益%", "胜率/天数",
        ])
        INFO_LABEL = ui.label("").classes("text-sm text-gray-400")
        COMBO_FIG = ui.plotly(go.Figure()).classes("w-full")
        WEIGHTS_FIG = ui.plotly(go.Figure()).classes("w-full")
        EXTRA_BOX = ui.column().classes("w-full")
        HOLDINGS_BOX = ui.column().classes("w-full")

    # 初始: 最近一次训练记录 或 预计算兜底
    runs = get_runs(limit=5)
    if runs:
        detail = get_run_detail(runs[0]["id"])
        if detail:
            STATE["result"] = detail
            _render_final(detail)
            STATE["render_id"] = id(detail)
    else:
        recs = discover_backtests()
        fwd = [r for r in recs if r["direction"] == "正向" and r.get("canonical", True)]
        real = [r for r in fwd if r.get("is_web") is not True and r["version"] <= 23]
        pick = real if real else fwd
        if pick:
            latest = max(pick, key=lambda r: r["version"])
            _render_fallback(latest)


# ── 页面 2: 关注圈 ───────────────────────────────────────────────────────────
def _build_pool_section(title: str, path: Path, circle_set: Optional[set], editable: bool,
                        grid_holder, diag_holder, default_df: pd.DataFrame) -> None:
    global POOL_GRID, POOL_DIAG, WL_GRID, WL_DIAG
    ui.label(title).classes("text-lg font-bold mt-3")
    rows = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("watchlist", [])
            rows = _clean_rows(data)
        except Exception:
            rows = []
    if not rows and len(default_df):
        rows = _clean_rows(default_df.to_dict("records"))
    if rows:
        _enrich_pool_rows(rows)

    _fmt_pct = "params.value==null?'—':(params.value>0?'+':'')+Number(params.value).toFixed(2)+'%'"
    col_defs = [
        {"headerName": "代码", "field": "code", "editable": editable, "width": 110},
        {"headerName": "名称", "field": "name", "editable": editable,
         "width": _col_w(rows, "name", 140, 360)},
        {"headerName": "主题", "field": "theme", "editable": editable,
         "width": _col_w(rows, "theme", 140, 320)},
        {"headerName": "日涨幅%", "field": "日涨幅", "editable": False, "width": 95,
         "type": "numericColumn", "valueFormatter": _fmt_pct},
        {"headerName": "月涨幅%", "field": "月涨幅", "editable": False, "width": 95,
         "type": "numericColumn", "valueFormatter": _fmt_pct},
        {"headerName": "年涨幅%", "field": "年涨幅", "editable": False, "width": 95,
         "type": "numericColumn", "valueFormatter": _fmt_pct},
    ]
    options = {
        "columnDefs": col_defs,
        "rowData": rows,
        "defaultColDef": {"resizable": True, "sortable": True},
        "rowHeight": 32,
    }
    grid = ui.aggrid(options, theme="alpine").classes("w-full h-64")
    if editable and title.startswith("🎯"):
        POOL_GRID = grid
    elif editable:
        WL_GRID = grid

    with ui.row().classes("gap-2"):
        if editable:
            ui.button("💾 保存", on_click=lambda: _save_pool(grid, path, diag_holder, circle_set, title)).props("primary")
            ui.button("↺ 重置默认", on_click=lambda: _reset_pool(grid, default_df, path, diag_holder, circle_set, title))

    diag = ui.column().classes("w-full")
    if editable and title.startswith("🎯"):
        POOL_DIAG = diag
    elif editable:
        WL_DIAG = diag
    _render_diag(diag, diagnose_self_selected(pd.DataFrame(rows)))


async def _save_pool(grid, path: Path, diag_holder, circle_set, title: str) -> None:
    rows = await grid.get_client_data()
    clean = _clean_rows(rows)
    if circle_set is not None:
        dropped = [r["code"] for r in clean if r["code"] not in circle_set]
        clean = [r for r in clean if r["code"] in circle_set]
    else:
        dropped = []
    _write_json(path, clean)
    msg = f"已保存 {len(clean)} 只"
    if dropped:
        msg += f"; 剔除圈外 {len(dropped)} 只"
    ui.notify(msg, type="positive")
    _render_diag(diag_holder, diagnose_self_selected(pd.DataFrame(clean)))


def _reset_pool(grid, default_df, path, diag_holder, circle_set, title) -> None:
    rows = _clean_rows(default_df.to_dict("records"))
    grid.options["rowData"] = rows
    ui.timer(0.01, lambda: asyncio.create_task(grid.load_client_data()), once=True)
    _write_json(path, rows)
    _render_diag(diag_holder, diagnose_self_selected(pd.DataFrame(rows)))
    ui.notify("已重置为默认", type="info")

def _on_fetch_data() -> None:
    """增量数据获取: 日K(iFinD) + 资金流(iFinD) + 概念板块."""
    start_d = (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_d = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    _pool_progress("🔍 检查现有数据…")
    py_managed = r"C:\Users\admin\.workbuddy\binaries\python\envs\quant\Scripts\python.exe"
    web_dir = str(WEB_DIR.parent)

    def _run_sub(cmd, label) -> bool:
        """运行子进程, 解析 PROGRESS 行, 返回 True=成功."""
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            line = line.strip()
            if line.startswith("PROGRESS:"):
                parts = line[len("PROGRESS:"):].split("/")
                if len(parts) == 2:
                    done_n, total_n = int(parts[0]), int(parts[1])
                    pct = done_n / total_n if total_n > 0 else 0
                    _pool_progress_update(pct)
                    _POOL_STATE["msg"] = f"{label}: {done_n}/{total_n}只"
        p.wait()
        return p.returncode == 0

    def _run():
        t0 = time.time()
        # 1. 日K
        _POOL_STATE["msg"] = "🔄 日K获取中…"
        kline_ok = _run_sub(
            [py_managed, str(WEB_DIR.parent / "scripts" / "backfill_ifind.py"),
             "--start", start_d, "--end", end_d, "--progress"],
            "🔄 日K"
        )
        if kline_ok:
            _POOL_STATE["msg"] = "✅ 日K完成"
            _pool_progress_update(0.5)
        else:
            _POOL_STATE["msg"] = "⚠️ 日K部分失败, 继续资金流…"

        # 2. 资金流 (iFinD)
        time.sleep(0.5)
        _POOL_STATE["msg"] = "🔄 资金流获取中 (iFinD)…"
        _pool_progress_update(0)
        ff_ok = _run_sub(
            [py_managed, str(WEB_DIR.parent / "scripts" / "backfill_fundflow_ifind.py"),
             "--progress"],
            "🔄 资金流"
        )
        elapsed = time.time() - t0
        if ff_ok:
            _POOL_STATE["msg"] = f"✅ 数据获取完成 (日K+资金流, {int(elapsed)}s)"
        else:
            _POOL_STATE["msg"] = f"⚠️ 日K{'✅' if kline_ok else '❌'} 资金流{'❌' if not ff_ok else '✅'} ({int(elapsed)}s)"
        _pool_progress_done_proper()
    threading.Thread(target=_run, daemon=True).start()

    def _run():
        py = r"C:\Users\admin\.workbuddy\binaries\python\envs\quant\Scripts\python.exe"
        script = str(WEB_DIR.parent / "scripts" / "backfill_ifind.py")
        cmd = [py, script, "--start", start_d, "--end", end_d, "--progress"]
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
        for line in p.stdout:
            line = line.strip()
            if line.startswith("PROGRESS:"):
                parts = line[len("PROGRESS:"):].split("/")
                if len(parts) == 2:
                    done_n, total_n = int(parts[0]), int(parts[1])
                    pct = done_n / total_n if total_n > 0 else 0
                    _pool_progress_update(pct)
                    _POOL_STATE["msg"] = f"🔄 数据获取: {done_n}/{total_n}只"
        p.wait()
        elapsed = time.time() - t0
        if p.returncode == 0:
            _POOL_STATE["msg"] = f"✅ 数据获取完成 ({int(elapsed)}s)"
        else:
            _POOL_STATE["msg"] = f"❌ 数据获取失败 rc={p.returncode}"
        _pool_progress_done_proper()
    threading.Thread(target=_run, daemon=True).start()


def _on_rebuild_features() -> None:
    """特征入库: 重建 v23 特征 (build_all → v22 → v23)."""
    _pool_progress("📦 特征入库中")

    step_labels = {0: "1/3 构建特征矩阵", 1: "2/3 复制 v22", 2: "3/3 构建 v23"}

    def _run():
        py = r"C:\Users\admin\.workbuddy\binaries\python\versions\3.13.12\python.exe"
        script = str(WEB_DIR.parent / "scripts" / "rebuild_features.py")
        cmd = [py, script, "--progress"]
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=str(WEB_DIR.parent),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
        for line in p.stdout:
            line = line.strip()
            if line.startswith("PROGRESS:"):
                parts = line[len("PROGRESS:"):].split("/")
                if len(parts) == 2:
                    step_n, total_n = int(parts[0]), int(parts[1])
                    pct = step_n / total_n if total_n > 0 else 0
                    _pool_progress_update(pct)
                    _POOL_STATE["msg"] = f"📦 {step_labels.get(step_n, f'特征入库 {step_n}/{total_n}')}"
        p.wait()
        elapsed = time.time() - t0
        if p.returncode == 0:
            _POOL_STATE["msg"] = f"✅ 特征入库完成 ({int(elapsed)}s)"
        else:
            _POOL_STATE["msg"] = f"❌ 特征入库失败 rc={p.returncode}"
        _pool_progress_done_proper()
    threading.Thread(target=_run, daemon=True).start()


# ── 池页进度条辅助 (线程安全: 用 _POOL_STATE + ui.timer 轮询, 不直接操作UI) ──
_POOL_PROG: Any = None
_POOL_LABEL: Any = None
_POOL_STATE: dict[str, Any] = {"msg": "", "pct": 0, "active": False, "_done_ts": 0}


def _pool_progress(msg: str) -> None:
    _POOL_STATE["msg"] = msg
    _POOL_STATE["active"] = True
    _POOL_STATE["pct"] = 0


def _pool_progress_update(pct: float) -> None:
    while pct > 1.0:
        pct /= 100.0
    _POOL_STATE["pct"] = pct


def _pool_progress_done() -> None:
    _POOL_STATE["msg"] = ""
    _POOL_STATE["active"] = False
    _POOL_STATE["pct"] = 0
    _POOL_STATE["_done_ts"] = 0


def _pool_progress_done_proper() -> None:
    """完成后保留消息 5 秒再清空."""
    _POOL_STATE["pct"] = 0
    _POOL_STATE["active"] = False
    _POOL_STATE["_done_ts"] = time.time()


def _pool_progress_poll() -> None:
    """由 ui.timer 定期轮询, 主线程安全."""
    global _POOL_PROG, _POOL_LABEL
    s = _POOL_STATE
    # 完成状态保留5秒
    if not s["active"] and s.get("_done_ts", 0) and time.time() - s["_done_ts"] > 5:
        s["_done_ts"] = 0
        s["msg"] = ""
    if s["active"] or s.get("_done_ts", 0):
        if _POOL_PROG is not None:
            PROG_BAR_PROPS = {prop.strip() for prop in (_POOL_PROG._props.get("class", "") + " " + _POOL_PROG._props.get("style", "")).split()}
            if s["pct"] <= 0:
                _POOL_PROG.props("indeterminate")
            else:
                _POOL_PROG.props(remove="indeterminate")
                _POOL_PROG.value = s["pct"]
            _POOL_PROG.set_visibility(True)
        if _POOL_LABEL is not None:
            _POOL_LABEL.text = s["msg"] + (f" ({s['pct']*100:.0f}%)" if s["pct"] > 0 else "")
            _POOL_LABEL.set_visibility(True)
    else:
        if _POOL_PROG is not None:
            _POOL_PROG.set_visibility(False)
            _POOL_PROG.value = 0
            _POOL_PROG.props(remove="indeterminate")
        if _POOL_LABEL is not None:
            _POOL_LABEL.text = s["msg"]
            _POOL_LABEL.set_visibility(True if s["msg"] else False)


def _get_last_update() -> str:
    """获取 v23 特征文件的最新修改时间 (BJT)."""
    import os
    v23 = str(latest_training_data())
    if not os.path.exists(v23):
        return "暂无特征数据"
    mtime = os.path.getmtime(v23)
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc) + timedelta(hours=8)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_kline_last_date() -> str:
    """获取 raw/kline 216只股票中的最晚交易日."""
    import os, pandas as pd
    kl = str(processed_dir().parent / "raw" / "kline")
    if not os.path.isdir(kl):
        return "无kline数据"
    # 从池股票读 (不在池的不会被 auto_wf 更新, 日期滞后)
    from backend.paths import load_universe_codes
    pool_codes = load_universe_codes("关注圈")
    max_d = pd.Timestamp(0)
    count = 0
    for code in pool_codes[:20]:  # 扫前20只足够
        fn = f"{code[:6]}.parquet"
        fp = os.path.join(kl, fn)
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp)
            # 兼容中文/英文列名
            date_col = "时间" if "时间" in df.columns else "date"
            mx = pd.to_datetime(df[date_col]).max()
            if mx > max_d:
                max_d = mx
            count += 1
        except Exception:
            continue
    return max_d.strftime("%Y-%m-%d") if max_d.year > 2020 else "未知"


def _get_event_last_date() -> str:
    """获取 events 中最晚的事件日期 (排除远期排期)."""
    import os, pandas as pd, glob
    # 优先 ths_news_clean (有精确时间)
    try:
        daily_dir = str(WEB_DIR.parent / "data" / "raw" / "events_daily")
        clean_files = sorted(glob.glob(os.path.join(daily_dir, "ths_news_clean_*.parquet")), reverse=True)
        for cf in clean_files[:3]:
            tmp = pd.read_parquet(cf, columns=["date", "time"])
            if "time" in tmp.columns and tmp["time"].notna().any():
                dt = pd.to_datetime(tmp["date"] + " " + tmp["time"], errors="coerce").max()
                if pd.notna(dt):
                    return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    # 回退 events_v2 (仅日期)
    if not os.path.exists(_EVENT_PATH):
        return "无事件数据"
    try:
        df = pd.read_parquet(_EVENT_PATH, columns=["date"])
        today = pd.Timestamp.now()
        ok = df[pd.to_datetime(df["date"]) <= today + pd.Timedelta(days=3)]
        mx = pd.to_datetime(ok["date"]).max() if len(ok) > 0 else None
        return pd.to_datetime(mx).strftime("%Y-%m-%d") if pd.notna(mx) else "未知"
    except Exception:
        return "未知"


def _get_event_feature_update() -> str:
    """获取事件特征 (events_v2) 文件修改时间 (BJT)."""
    import os
    if not os.path.exists(_EVENT_PATH):
        return "无事件特征"
    try:
        mtime = os.path.getmtime(_EVENT_PATH)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc) + timedelta(hours=8)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "未知"


def build_pool_page() -> None:
    global _POOL_PROG, _POOL_LABEL
    ui.label("🎯 关注圈 & 自选股").classes("text-2xl font-bold mb-2")
    with ui.row().classes("gap-2 mb-1"):
        ui.button("🔄 数据获取 (iFinD kline)", on_click=_on_fetch_data).props("dense")
        ui.button("📦 特征入库 (rebuild v23)", on_click=_on_rebuild_features).props("dense")
    # 最后更新时间 (读 v23 文件 mtime)
    last_up = _get_last_update()
    kline_max = _get_kline_last_date()
    ui.label(f"📅 K线最新日: {kline_max} | 特征最后更新: {last_up}").classes(
        "text-xs text-gray-500 mb-1")
    _POOL_PROG = ui.linear_progress(value=0).classes("w-full mt-1")
    _POOL_LABEL = ui.label("").classes("text-sm text-gray-400")
    _POOL_PROG.set_visibility(False)
    _POOL_LABEL.set_visibility(False)
    ui.timer(0.5, _pool_progress_poll, active=True)
    ui.label("关注圈 = 模型训练/回测的实际股票范围 (可编辑, 默认=216只全池); 自选股 = 独立维度。编辑后点保存生效。")

    # 默认全集 (universe/watchlist_216.json, 216 只) — 编辑后另存为 Web/data/train_pool.json
    full_df = pd.DataFrame()
    wp = watchlist_path()
    if wp.exists():
        try:
            full_df = pd.DataFrame(json.loads(wp.read_text(encoding="utf-8")).get("watchlist", []))
        except Exception:
            pass

    _build_pool_section("🎯 关注圈 (可编辑, 训练实际范围)", train_pool_path(), None, True, None, None, full_df)
    _build_pool_section("⭐ 自选股 (可编辑)", self_selected_path(), None, True, None, None, full_df)


# ── 页面 3: 特征 (增删改, 默认 v23) ─────────────────────────────────────────
def _build_feat_rows(selected: set) -> list[dict]:
    return [
        {"feature": f, "维度": _dim_of(f), "含义": _meaning_of(f), "缺失率": "—", "纳入": f in selected}
        for f in FEAT_ALL
    ]


async def _compute_missing() -> None:
    global FEAT_ROWS
    path = latest_training_data()
    df = pd.read_parquet(path, columns=FEAT_ALL)
    miss = df.isnull().mean()
    rate_map = {c: round(float(miss.get(c, 0.0)) * 100, 1) for c in FEAT_ALL}
    FEAT_ROWS = [
        {**r, "缺失率": rate_map.get(r["feature"], 0.0)} for r in FEAT_ROWS
    ]
    if FEAT_GRID is not None:
        FEAT_GRID.options["rowData"] = FEAT_ROWS
        await FEAT_GRID.load_client_data()
    STATE["features_loaded"] = True


def _ensure_features() -> None:
    if not STATE["features_loaded"]:
        STATE["features_loaded"] = True
        try:
            asyncio.create_task(_compute_missing())
        except RuntimeError:
            pass  # 无事件循环时不加载


async def _save_feats() -> None:
    rows = await FEAT_GRID.get_client_data()
    sel = [r["feature"] for r in rows if r.get("纳入")]
    save_feature_selection(sel)
    STATE["selected_features"] = sel
    ui.notify(f"已保存特征选择: {len(sel)} 个 (默认 v23 过滤集 {len(default_feature_set())})", type="positive")


def _set_all(val: bool) -> None:
    for r in FEAT_ROWS:
        r["纳入"] = val
    if FEAT_GRID is not None:
        FEAT_GRID.options["rowData"] = FEAT_ROWS
        ui.timer(0.01, lambda: asyncio.create_task(FEAT_GRID.load_client_data()), once=True)


def _reset_default() -> None:
    global FEAT_ROWS
    default = set(default_feature_set())
    FEAT_ROWS = _build_feat_rows(default)
    if FEAT_GRID is not None:
        FEAT_GRID.options["rowData"] = FEAT_ROWS
        ui.timer(0.01, lambda: asyncio.create_task(FEAT_GRID.load_client_data()), once=True)
    STATE["selected_features"] = list(default)
    ui.notify("已重置为 v23 默认过滤集", type="info")


async def _show_hist() -> None:
    feat = FEAT_SELECT.value
    if not feat:
        return
    path = latest_training_data()
    ser = pd.read_parquet(path, columns=[feat])[feat].dropna()
    fig = go.Figure(go.Histogram(x=ser, nbinsx=50, marker_color="#378ADD"))
    fig.update_layout(title=f"{feat} 分布 (n={len(ser)})", height=300)
    FEAT_HIST.clear()
    with FEAT_HIST:
        ui.plotly(style(fig))


# ── 事件 Tab ────────────────────────────────────────────────────────────────
_EVENT_DIR = str(WEB_DIR.parent / "data" / "raw" / "events_ifind")
_EVENT_PATH = str(WEB_DIR.parent / "data" / "raw" / "events_ifind" / "events_v2.parquet")
# 事件方向中文映射
_EVENT_DIR_LABEL = {
    1: ("🔵 利好", "bg-teal-500"),
    -1: ("🔴 利空", "bg-red-500"),
    0: ("⚪ 中性", "bg-blue-300"),
}
# P级别颜色 (CSS类)
_P_CLASSES = {"P0": "bg-red-600 text-white px-1 rounded", "P1": "bg-yellow-400 text-black px-1 rounded",
             "P2": "bg-blue-400 text-white px-1 rounded", "P3": "bg-green-400 text-white px-1 rounded"}
# 事件类型中文映射
_EVENT_TYPE_CN = {
    "lawsuit": "诉讼", "dividend": "分红", "reduction": "减持", "pledge": "质押",
    "buyback_plan": "回购计划", "buyback_ongoing": "回购中", "buyback_done": "回购完成",
    "equity_incentive": "股权激励", "regulatory_action": "监管处罚", "increase": "增持",
    "earnings_revise": "业绩修正", "big_contract": "重大合同", "state_capital": "国资入驻",
    "regulatory_filing": "监管函", "regulatory": "监管", "unpledge": "解质押",
    "routine_filing": "常规公告", "expansion": "扩产/投资", "major_restructure": "重大重组",
    "unlock": "解禁", "delist_risk": "退市风险", "research_upgrade": "研报上调",
    "research_downgrade": "研报下调", "profit_warn": "业绩预警", "profit_forecast": "业绩预告",
    "oil_crash": "原油波动",
}


# ── 特征 Tab (特征选择器) ──────────────────────────────────────────────────
FEAT_SORTED: list[str] = []
FEAT_ALL: list[dict] = []
FEAT_ROWS: list[dict] = []
FEAT_GRID: Any = None
FEAT_HIST: Any = None
FEAT_SELECT: Any = None


def build_features_page() -> None:
    global FEAT_SORTED, FEAT_ALL, FEAT_ROWS, FEAT_GRID, FEAT_HIST, FEAT_SELECT
    ui.label("🧬 特征选择").classes("text-2xl font-bold mb-2")
    ui.label("勾选需纳入训练/回测的特征。默认 = 过滤后的规范集 (排除标签/leak/_21d/_cross/已剔除的)").classes("text-xs text-gray-400 mb-1")
    with ui.row().classes("gap-1 mb-1"):
        ui.button("全选", on_click=lambda: _set_all(True)).props("dense")
        ui.button("全清", on_click=lambda: _set_all(False)).props("dense")
        ui.button("恢复默认", on_click=_reset_default).props("dense")
        ui.button("保存", on_click=_save_feats).props("dense outline")
        FEAT_SELECT = ui.select(FEAT_SORTED, label="查看分布", with_input=True,
                                on_change=_show_hist).classes("w-64")
    FEAT_HIST = ui.plotly(go.Figure()).classes("w-full")
    FEAT_GRID = ui.aggrid({
        "columnDefs": [
            {"headerName":"维度","field":"dim","width":90,"filter":True},
            {"headerName":"特征","field":"feature","width":260,"filter":True},
            {"headerName":"含义","field":"meaning","width":220,"filter":True},
            {"headerName":"纳入","field":"纳入","width":80,"editable":True,
             "cellRenderer":"agCheckboxCellRenderer"},
        ],
        "rowData": FEAT_ROWS,
        "defaultColDef": {"sortable": True, "resizable": True},
        "rowHeight": 28,
        "stopEditingWhenCellsLoseFocus": True,
        "enableCellTextSelection": True,
    }, html_columns=[], theme="balham-dark").classes("w-full")
    FEAT_GRID.on("cellValueChanged", lambda e: None)
    # 延迟加载特征 (异步需在事件循环中执行)
    ui.timer(0.1, lambda: _ensure_features(), once=True)


# ── 事件Tab 辅助函数 ────────────────────────────────────────────────────────
_EVENT_PBAR: Any = None
_EVENT_PLABEL: Any = None
_EVENT_PSTATE: dict = {"msg": "", "active": False, "_ts": 0}
_EVENT_TABLE: Any = None  # 事件列表容器


def _event_progress(msg: str) -> None:
    _EVENT_PSTATE["msg"] = msg
    _EVENT_PSTATE["active"] = True
    _EVENT_PSTATE["_ts"] = time.time()


def _event_done() -> None:
    _EVENT_PSTATE["active"] = False
    _EVENT_PSTATE["_ts"] = time.time()


def _event_poll() -> None:
    s = _EVENT_PSTATE
    if not s["active"] and time.time() - s["_ts"] > 5:
        s["msg"] = ""
    show = s["active"] or (s["_ts"] > 0 and time.time() - s["_ts"] < 5)
    if _EVENT_PBAR is not None:
        _EVENT_PBAR.set_visibility(show)
        if show:
            _EVENT_PBAR.props("indeterminate" if s["active"] else "indeterminate")
    if _EVENT_PLABEL is not None:
        _EVENT_PLABEL.text = s["msg"]
        _EVENT_PLABEL.set_visibility(show)
    if not show and _EVENT_PBAR is not None:
        _EVENT_PBAR.set_visibility(False)
    if not show and _EVENT_PLABEL is not None:
        _EVENT_PLABEL.set_visibility(False)


def _on_fetch_events() -> None:
    """事件获取: 同花顺7x24快讯 + 盘后公告."""
    _event_progress("🔄 事件获取中 (同花顺快讯)…")
    py = r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe"
    script = str(WEB_DIR.parent / "pipeline" / "process_ths_news.py")

    def _run():
        t0 = time.time()
        r = subprocess.run([py, script], capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        if r.returncode == 0:
            _event_progress(f"✅ 事件获取完成 ({int(elapsed)}s)")
        else:
            _event_progress(f"❌ 事件获取失败 rc={r.returncode}")
        _event_done()
    threading.Thread(target=_run, daemon=True).start()


def _on_rebuild_events() -> None:
    """事件特征入库: clean_events → expand_events。"""
    _event_progress("📦 事件特征入库中…")
    py = r"C:\Users\admin\.workbuddy\binaries\python\versions\3.13.12\python.exe"

    def _run():
        t0 = time.time()
        steps = [("clean_events", ["-m", "pipeline.clean_events"]),
                 ("expand_events", ["-m", "pipeline.expand_events"])]
        for sn, cmd_args in steps:
            _event_progress(f"📦 {sn}…")
            r = subprocess.run([py] + cmd_args, cwd=str(WEB_DIR.parent),
                              capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                _event_progress(f"❌ {sn} 失败 rc={r.returncode}")
                _event_done()
                return
        elapsed = time.time() - t0
        _event_progress(f"✅ 事件特征入库完成 ({int(elapsed)}s)")
        _event_done()
    threading.Thread(target=_run, daemon=True).start()


def build_event_page() -> None:
    global _EVENT_PBAR, _EVENT_PLABEL, _EVENT_TABLE
    ui.label("📰 事件列表 & 管理").classes("text-2xl font-bold mb-2")

    # 最新事件日期 + 事件特征更新时间 (events_v2 文件时间)
    ev_latest = _get_event_last_date()
    feat_up = _get_event_feature_update()
    ui.label(f"📅 事件最新日: {ev_latest} | 事件特征更新: {feat_up}").classes(
        "text-xs text-gray-500 mb-1")
    with ui.row().classes("gap-2 mb-1"):
        ui.button("🔄 事件获取 (同花顺快讯)", on_click=_on_fetch_events).props("dense")
        ui.button("📦 事件特征入库", on_click=_on_rebuild_events).props("dense")
    # 进度条
    _EVENT_PBAR = ui.linear_progress(value=0).classes("w-full mt-1")
    _EVENT_PLABEL = ui.label("").classes("text-sm text-gray-400")
    _EVENT_PBAR.set_visibility(False)
    _EVENT_PLABEL.set_visibility(False)
    ui.timer(0.5, _event_poll, active=True)

    import os, pandas as pd, json, glob
    # 加载事件数据: ths_news_clean (含 time) + events_v2 合并
    df = pd.DataFrame()
    try:
        daily_dir = str(WEB_DIR.parent / "data" / "raw" / "events_daily")
        clean_files = sorted(glob.glob(os.path.join(daily_dir, "ths_news_clean_*.parquet")), reverse=True)
        for cf in clean_files[:7]:
            tmp = pd.read_parquet(cf)
            if "time" in tmp.columns:
                tmp["datetime"] = tmp["date"] + " " + tmp["time"]
            else:
                tmp["datetime"] = tmp["date"]
            df = pd.concat([df, tmp], ignore_index=True)
    except Exception:
        pass

    # 补 events_v2 (更完整的历史)
    if os.path.exists(_EVENT_PATH):
        try:
            recent = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
            ev2 = pd.read_parquet(_EVENT_PATH, filters=[("date", ">=", recent)])
            ev2["datetime"] = ev2["date"]
            df = pd.concat([df, ev2], ignore_index=True).drop_duplicates(
                subset=["datetime", "code", "event_type", "title"], keep="first")
        except Exception:
            pass

    # 加载 stock→name 映射
    name_map = _load_event_names()

    # 日期筛选栏
    with ui.row().classes("gap-2 mb-1 items-center"):
        date_from = ui.input("起始日期", value="2026-07-01").classes("w-32")
        date_to = ui.input("截止日期", value="").classes("w-32")
        p_filter = ui.select(["全部","P0","P1","P2","P3"], value="全部").classes("w-24")
        dir_filter = ui.select(["全部","利好","利空","中性"], value="全部").classes("w-24")
        ui.button("筛选", on_click=lambda: _render_event_table(
            df, date_from.value, date_to.value, p_filter.value, dir_filter.value)).props("dense")
        ui.button("重置", on_click=lambda: _render_event_table(
            df, "", "", "全部", "全部")).props("dense")

    _EVENT_TABLE = ui.column().classes("w-full")
    _render_event_table(df)


def _load_event_names() -> dict:
    name_map = {}
    try:
        wp = str(WEB_DIR.parent / "data" / "universe" / "watchlist_216.json")
        if os.path.exists(wp):
            with open(wp, encoding="utf-8") as f:
                wl = json.load(f)
            for s in (wl.get("watchlist", []) if isinstance(wl, dict) else wl):
                name_map[s["code"][:6]] = s.get("name", s["code"][:6])
    except Exception:
        pass
    return name_map


def _render_event_table(df, date_from="", date_to="", p_sel="全部", dir_sel="全部"):
    _EVENT_TABLE.clear()
    with _EVENT_TABLE:
        if len(df) == 0:
            ui.label("暂无事件数据").classes("text-gray-400")
            return
        raw = df.sort_values("datetime", ascending=False)
        # 排除远期事件 (>30天后)
        today_str = (pd.Timestamp.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        raw = raw[raw["datetime"] <= today_str]
        if date_from:
            raw = raw[raw["datetime"] >= date_from]
        if date_to:
            raw = raw[raw["datetime"] <= date_to]
        if p_sel != "全部":
            # 兼容 Pn 和 n 两种格式
            p_int = p_sel[1]  # "P0" -> "0"
            raw = raw[raw["p_level"].astype(str).str.replace("P", "").str.strip() == p_int]
        if dir_sel != "全部":
            dm = {"利好": 1, "利空": -1, "中性": 0}
            raw = raw[raw["direction"] == dm[dir_sel]]

        show = raw.head(500)
        name_map = _load_event_names()
        rows = []
        for _, r in show.iterrows():
            code_col = "stock_code" if "stock_code" in r else "code"
            c6 = str(r[code_col])[:6]
            c6 = "" if c6 in ("nan", "None", "__") else c6
            sname = name_map.get(c6, str(r.get("name", c6)))
            sname = "" if sname in ("nan", "None") else sname
            # 方向
            direction = r.get("direction", 0)
            if isinstance(direction, str):
                dir_map = {"bullish": 1, "bearish": -1, "neutral": 0}
                direction = dir_map.get(direction, 0)
            else:
                try:
                    direction = int(float(direction)) if not (isinstance(direction, float) and np.isnan(direction)) else 0
                except (ValueError, TypeError):
                    direction = 0
            dlabel, dcolor = _EVENT_DIR_LABEL.get(direction, ("⚪ 未知", ""))
            # 级别: 转 Pn 格式
            p_raw = r.get("p_level", "")
            try:
                p_str = f"P{int(p_raw)}"
            except (ValueError, TypeError):
                p_str = str(p_raw)
            pclass = _P_CLASSES.get(p_str, "")
            # 标题: 兼容 title/reason 列
            title = str(r.get("title") or r.get("reason") or "")[:120]
            # 类别: 英文→中文
            ev_type = str(r.get("event_type", ""))
            if ev_type in ("nan", "None", ""):
                ev_type = "—"
            else:
                ev_type = _EVENT_TYPE_CN.get(ev_type, ev_type)
            rows.append({
                "date": str(r.get("datetime", r.get("date", "")))[:19],
                "stock": (f"{c6} {sname}").strip() if c6 else sname,
                "type": ev_type,
                "level": p_str,
                "direction": dlabel,
                "title": title,
            })
        cols = [
            {"name":"date","label":"DateTime","field":"date","align":"left"},
            {"name":"stock","label":"股票","field":"stock","align":"left"},
            {"name":"type","label":"类别","field":"type","align":"left"},
            {"name":"level","label":"级别","field":"level","align":"center"},
            {"name":"direction","label":"方向","field":"direction","align":"center"},
            {"name":"title","label":"标题","field":"title","align":"left"},
        ]
        if rows:
            ui.table(rows=rows, columns=cols, pagination=50).classes("w-full")
            ui.label(f"显示第 1-{len(rows)} 条 / 筛选后 {len(raw):,} 条").classes("text-xs text-gray-400")
        else:
            ui.label("无匹配事件").classes("text-gray-400")


def build_compare_page() -> None:
    ui.label("📈 结果对比").classes("text-2xl font-bold mb-2")
    registry: dict[str, dict] = {}

    runs = get_runs(limit=50)
    for r in runs:
        registry[f"Run#{r['id']} {str(r.get('created_at',''))[:10]}"] = {
            "kind": "run", "id": r["id"],
            "sharpe": r.get("sharpe_raw"), "annual": (r.get("annual_return") or 0) * 100,
            "ic": r.get("ic_mean"), "maxdd": (r.get("max_dd") or 0) * 100,
            "win": (r.get("win_rate") or 0) * 100, "topn": r.get("top_n"),
            "nfeat": r.get("n_features"),
        }
    recs = discover_backtests()
    for rec in recs:
        if rec["direction"] != "正向":
            continue
        s = rec["summary"]
        registry[f"{rec['name']}"] = {
            "kind": "pre", "path": rec["path"],
            "sharpe": s.get("sharpe"), "annual": s.get("annualized_return_pct"),
            "ic": s.get("ic_mean"), "maxdd": s.get("max_dd_pct"),
            "win": s.get("win_rate_pct"), "topn": None, "nfeat": rec.get("features"),
        }

    if not registry:
        ui.label("暂无可对比的结果")
        return

    sel = ui.select(list(registry.keys()), label="选择 2-4 项对比", value=[], multiple=True).classes("w-full")
    overlay = ui.column().classes("w-full")
    table_box = ui.column().classes("w-full")

    def on_compare() -> None:
        chosen = sel.value or []
        if len(chosen) < 2:
            ui.notify("请至少选择 2 项", type="warning")
            return
        overlay.clear()
        table_box.clear()
        table_rows = []
        colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f25c54"]
        fig = go.Figure()
        for i, name in enumerate(chosen):
            meta = registry[name]
            table_rows.append({
                "名称": name, "Sharpe": meta["sharpe"], "年化%": meta["annual"],
                "IC": meta["ic"], "最大回撤%": meta["maxdd"], "胜率%": meta["win"],
                "TopN": meta["topn"], "特征数": meta["nfeat"],
            })
            # 叠加累计收益
            daily = None
            if meta["kind"] == "run":
                d = get_run_detail(meta["id"])
                daily = d.get("daily_returns", []) if d else []
            else:
                pb = load_backtest(meta["path"])
                daily = pb.get("daily", []) if pb else []
            if daily:
                # cum_return 新格式为实际账户金额(元), 旧格式为倍率; 统一转万元
                sample = daily[0].get("cum_return", 1.0)
                if sample > 10_000:
                    # 新格式: 实际金额(元)
                    cum = [x.get("cum_return", _INITIAL_CAPITAL) / 10_000 for x in daily]
                else:
                    # 旧格式: 倍率
                    cum = [x.get("cum_return", 1.0) * _INITIAL_CAPITAL / 10_000 for x in daily]
                dates = [x["date"] for x in daily]
                fig.add_trace(go.Scatter(
                    x=dates, y=cum, name=name,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
        fig.add_hline(y=200, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                       annotation_text="初始200万")
        fig.update_layout(title="累计收益对比 (账户金额万元)", height=360, yaxis_title="账户金额(万元)")
        with overlay:
            ui.plotly(style(fig))
        with table_box:
            cmp_cols = [
                {"name": "名称", "label": "名称", "field": "名称", "width": "380px"},
                {"name": "Sharpe", "label": "Sharpe", "field": "Sharpe"},
                {"name": "年化%", "label": "年化%", "field": "年化%"},
                {"name": "IC", "label": "IC", "field": "IC"},
                {"name": "最大回撤%", "label": "最大回撤%", "field": "最大回撤%"},
                {"name": "胜率%", "label": "胜率%", "field": "胜率%"},
                {"name": "TopN", "label": "TopN", "field": "TopN"},
                {"name": "特征数", "label": "特征数", "field": "特征数"},
            ]
            ui.table(rows=table_rows, columns=cmp_cols, pagination=20).classes("w-full")
        render_directional_alpha(table_box)

    ui.button("对比", on_click=on_compare).props("primary")


# ── 主入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    global TABS, T_TRAIN, T_POOL, T_FEAT, T_CMP, T_EVENT
    init_db()
    STATE["selected_features"] = load_feature_selection()

    ui.dark_mode(True)
    # 全局样式: 关闭 Quasar 所有 ellipsis(表格单元格/表头/下拉选项/标签/多选chip/页签),
    # 长文本(如 run 名/特征名)不换行、不被 "..." 截断; overflow:hidden 防溢出重叠(列宽已给足)
    ui.add_head_html("""
    <style>
    .ellipsis, .q-table__td, .q-table__th, .q-item__label,
    .q-chip__label, .q-tab__label, .q-field__label {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: clip !important;
    }
    </style>
    """)
    app.add_static_files("/data", str(WEB_DIR / "data"))

    # 顶部状态栏
    with ui.header().classes("bg-slate-800 text-white items-center"):
        with ui.row().classes("w-full items-center gap-4"):
            ui.label("📊 量化策略平台 (NiceGUI)").classes("text-lg font-bold")
            try:
                v = latest_training_data().stem.replace("training_data_", "")
                ui.label(f"{v}").classes("text-sm text-gray-300 bg-slate-700 px-2 py-0.5 rounded")
            except Exception:
                pass
            ui.space()
            try:
                n_pool = len(load_universe_codes("关注圈"))
                ui.label(f"关注圈 {n_pool}只").classes("text-sm text-gray-400")
            except Exception:
                pass

    with ui.tabs() as TABS:
        T_TRAIN = ui.tab("训练 & 回测")
        T_POOL = ui.tab("关注圈")
        T_FEAT = ui.tab("特征")
        T_CMP = ui.tab("结果对比")
        T_EVENT = ui.tab("事件")

    with ui.tab_panels(TABS, value=T_TRAIN).classes("w-full"):
        with ui.tab_panel(T_TRAIN):
            build_training_page()
        with ui.tab_panel(T_POOL):
            build_pool_page()
        with ui.tab_panel(T_FEAT):
            build_features_page()
        with ui.tab_panel(T_CMP):
            build_compare_page()
        with ui.tab_panel(T_EVENT):
            build_event_page()

    ui.timer(1.0, _poll_progress)


if __name__ == "__main__":
    main()
    ui.run(title="量化策略平台 (NiceGUI)", port=8502, reload=False, show=False)
