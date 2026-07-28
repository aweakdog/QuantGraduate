"""
Training service extracted from app.py.

Call  train(params: TrainParams) -> TrainResult  to run an expanding-window
LightGBM backtest.  Results are persisted to SQLite via database.py.

Usage::

    from backend.trainer import train, TrainParams
    result = train(TrainParams(train_start="2023-01-01", ...))
    print(result.sharpe_raw, result.run_id)
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import date as DateType
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

# ── Ensure the project root is on sys.path so we can import from sibling
#    packages (e.g. engine/, pipeline/) if needed later. ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # .../quant-strategy
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Local imports ─────────────────────────────────────────────────────────
from backend.database import init_db, save_run, save_last_train_params
from backend.models import TrainParams, TrainResult

# NumPy 2.x compat
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

# ── Paths ─────────────────────────────────────────────────────────────────
# 训练数据路径由 backend.paths 统一解析 (自动选最新 training_data_vXX, 当前 v23)
from backend.paths import latest_training_data, load_universe_codes

_TRAIN_DATA_PATH = latest_training_data()
_DEFAULT_CAPITAL = 2_000_000  # 默认初始资金(元), 可由 TrainParams.initial_capital 覆盖

# ── Column exclusion rules (mirror app.py) ────────────────────────────────
_SKIP = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret"}
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


# ── 中断信号 ────────────────────────────────────────────────────────────────
class _TrainCancelled(Exception):
    """用户取消训练时由训练循环抛出。"""


# ── Public API ────────────────────────────────────────────────────────────

def train(
    params: Union[TrainParams, dict[str, Any]],
    progress_callback: Optional[Callable[..., None]] = None,
    pause_event: Optional[threading.Event] = None,
    cancel_event: Optional[threading.Event] = None,
) -> TrainResult:
    """
    Run an expanding-window LightGBM training + backtest.

    Parameters
    ----------
    params : TrainParams or dict
        See TrainParams for fields.  If a dict is passed it will be
        converted automatically.
    progress_callback : callable or None
        Optional callback invoked after each prediction day.
        Receives a dict with keys: date, ic, top_ret, cum_return, progress_pct,
        elapsed_seconds, sharpe_sofar.

    Returns
    -------
    TrainResult
    """
    # ── Normalise input ───────────────────────────────────────────────────
    if isinstance(params, dict):
        params = TrainParams(**params)

    # Ensure DB is ready
    init_db()

    t_start = time.time()

    # ── 1. Load + preprocess data ────────────────────────────────────────
    df = pd.read_parquet(_TRAIN_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Handle inf
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # ── close 基执行: 模型目标/IC/收益用 fwd_1d_excess (截面demean, 纯T+1超额)
    # fwd_1d_ret=收盘价收益率 → fwd_1d_excess=去均值(截面demean), 排除市场beta
    # (注: excess=raw−截面均值, 减常数不改变排序 → IC_rank 不变; 但收益更干净)
    df["fwd_1d_excess"] = df.groupby("date")["fwd_1d_exec_ret"].transform(lambda x: x - x.mean())

    # ── 1.5 训练集来源过滤 (关注圈 / 自选股 / 自定义) ──
    # 新股<2年替换规则已在特征构建期落地 (parquet 内新股已替换), 此处仅按来源过滤行, 不影响该铁律。
    codes = load_universe_codes(params.universe_source, params.custom_codes)
    if not codes:
        raise ValueError(
            f"训练集来源『{params.universe_source}』未解析到任何股票代码 — "
            f"自定义来源请先填写代码; 自选股/关注圈请检查股票池文件"
        )
    _all = set(df["code"].astype(str).str.upper())
    _sel = set(codes)
    _excluded = _sel - _all
    if _excluded:
        print(
            f"[UNIVERSE] 来源『{params.universe_source}』: {len(_excluded)}/{len(_sel)} 只"
            f"不在训练数据集中, 已跳过: {sorted(_excluded)[:10]}"
            f"{'...' if len(_excluded) > 10 else ''}"
        )
    df = df[df["code"].astype(str).str.upper().isin(_sel)]
    if len(df) == 0:
        raise ValueError(
            f"训练集来源『{params.universe_source}』解析后无有效股票 — "
            f"请确认关注圈/自选股代码在默认池 (universe/watchlist*.json) 内"
        )

    # ── 2. Feature selection ─────────────────────────────────────────────
    # 默认过滤 (排除标签/leak/_21d长周期/_cross/_EXCLUDED), 与 v23 行为一致 (236 特征)
    _default_feats = [
        c for c in df.columns
        if c not in _SKIP and "_21d" not in c
        and not c.endswith("_cross") and c not in _LEAK
        and c not in _EXCLUDED
    ]
    if params.features:
        # 自定义特征集: 仅保留数据集中存在且安全的列 (标签/leak/_21d/_cross 强制排除, 防泄漏)
        _unsafe = set(_SKIP) | set(_LEAK) | _EXCLUDED
        features = [
            c for c in params.features
            if c in df.columns and c not in _unsafe
            and "_21d" not in c and not c.endswith("_cross")
        ]
        _dropped = sorted(set(params.features) - set(features))
        if _dropped:
            print(f"[FEATURES] 已剔除 {len(_dropped)} 个不安全特征(标签/leak/_21d/_cross): "
                  f"{_dropped[:10]}{'...' if len(_dropped) > 10 else ''}")
    else:
        features = _default_feats

    # ── 2.5 (close基不需要特征前移; open基需要shift(1)防泄漏, close基无此问题) ──

    # df_full = 全量行(含末日NaN标签, 供 next_rec 预测 + all_dates 找下一交易日);
    # df = dropna 后(标签有效, 供训练/评估/选股)
    df_full = df
    df = df.dropna(subset=["fwd_1d_exec_ret"])

    # ── 3. Build LightGBM params dict ────────────────────────────────────
    lgb_params = {
        "n_estimators": params.n_estimators,
        "max_depth": params.max_depth,
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "subsample": params.subsample,
        "colsample_bytree": params.colsample_bytree,
        "min_child_samples": params.min_child_samples,
        "random_state": params.random_state,
        "n_jobs": 32,
        "verbosity": -1,
    }

    # ── 4. Expanding-window loop ─────────────────────────────────────────
    import lightgbm as lgb
    from scipy.stats import spearmanr

    train_start_ts = pd.Timestamp(params.train_start)
    test_start_ts = pd.Timestamp(params.test_start)
    test_end_ts = pd.Timestamp(params.test_end) if params.test_end else None
    _date_mask = df["date"] >= test_start_ts
    if test_end_ts is not None:
        _date_mask &= df["date"] <= test_end_ts
    dates = sorted(df[_date_mask]["date"].unique())
    if not dates:
        raise ValueError(
            f"回测日期区间无有效数据: train_start={params.train_start}, "
            f"test_start={params.test_start}, test_end={params.test_end}. "
            f"可能 test_start > test_end 或数据不足"
        )
    all_dates = sorted(df_full["date"].unique())  # 全量日期(含末日NaN标签日), 用于 next_rec 找下一交易日
    # dates 已来自 df(dropna后)=标签有效日, 末日(标签NaN)天然不在评估日里;
    # 末日特征保留在 df_full 供 next_rec 预测, 故无需 reservation 让出评估日.
    # 最后一日若无下一交易日(all_dates末位), 则用当日特征 fallback 预测 (用户要求).

    # ── 成本费率 (买卖各扣滑点, 精细模型) ──
    buy_rate = params.buy_pct / 100.0
    sell_rate = params.sell_pct / 100.0
    slip_rate = params.slip_pct / 100.0
    # 买入预留: buy_amt * (1 + buy_rate + slip_rate) ≤ capital → buy_amt = capital / (1+buy_rate+slip_rate)
    buy_factor = 1.0 / (1.0 + buy_rate + slip_rate)
    # 卖出到手: sell_gross * (1 - sell_rate - slip_rate)

    daily: list[dict[str, Any]] = []
    feature_gain_totals: dict[str, float] = {}

    # 实时 next_rec 前移回填用的状态 + 固定 x 轴范围 (含 warm-up 区间)
    prev_state: dict[str, Any] | None = None
    x_range = [str(test_start_ts.date()), str(dates[-1].date())]

    # ── 预热进度回调: 回测起始日前只训练不输出图, 但给 UI 明确状态提示 ──
    # 用户定义: 训练起始→回测起始 间不输出/不联动, 故仅告知"预热中"而非画图;
    # 回测起始日之后, 主循环每次迭代才推送真实输出并开始联动.
    if progress_callback is not None and len(dates) > 0:
        _warmup_days = int((test_start_ts - train_start_ts).days)
        progress_callback({
            "warmup": True,
            "day_n": 0,
            "total_days": len(dates),
            "train_start": str(train_start_ts.date()),
            "test_start": str(test_start_ts.date()),
            "warmup_days": _warmup_days,
            "x_range": x_range,
        })

    for i, dt in enumerate(dates):
        # ── 暂停 / 取消 信号 (每天迭代初检查) ──
        if cancel_event is not None and cancel_event.is_set():
            raise _TrainCancelled("用户取消训练")
        if pause_event is not None and pause_event.is_set():
            while pause_event.is_set() and not (cancel_event is not None and cancel_event.is_set()):
                time.sleep(0.3)
        train_df = df[(df["date"] < dt) & (df["date"] >= train_start_ts)]
        if len(train_df) == 0:
            continue
        test_df = df[df["date"] == dt]

        # Prepare train set
        X_tr = train_df[features].copy()
        X_tr = X_tr.groupby(train_df["code"]).ffill().fillna(0)
        y_tr = train_df["fwd_1d_excess"]

        # Fit model
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr, y_tr)

        # Accumulate feature importance (average across days)
        if hasattr(model, "feature_importances_"):
            for feat, gain in zip(features, model.feature_importances_):
                feature_gain_totals[feat] = feature_gain_totals.get(feat, 0.0) + gain

        # Prepare test set
        X_te = test_df[features].copy()
        codes_te = test_df["code"].values
        X_te["_c"] = codes_te
        for c in features:
            if X_te[c].isna().any():
                X_te[c] = X_te.groupby("_c")[c].ffill().fillna(0)
        preds = model.predict(X_te[features])

        # Metrics
        if len(preds) > 5:
            ic, _ = spearmanr(preds, test_df["fwd_1d_excess"])
            # 退化预测 (spearmanr 返回 nan) → None, summary 阶段剔除 (与 v23 一致, 不兜底成 0)
            ic = float(ic) if (ic is not None and ic == ic) else None
        else:
            ic = None  # 测试股票数 <=5, 不足以估计 IC

        top_idx = np.argsort(preds)[-params.top_n :][::-1]  # 降序: 评分最高在前
        # top_ret 存原始超额收益(不含成本), 成本在结果汇总时按实际金额逐日计算扣除
        top_ret = float(test_df.iloc[top_idx]["fwd_1d_excess"].mean())

        # ── Extract top holdings: stock codes + prediction scores ──
        top_codes = [str(c) for c in test_df.iloc[top_idx]["code"].values]
        top_preds = [round(float(p), 6) for p in preds[top_idx]]

        # ── 当日持仓 (model(<T) 预测 T) ──
        cur_holdings = [{"code": c, "pred_score": s} for c, s in zip(top_codes, top_preds)]

        daily.append({
            "date": str(dt.date()),
            "ic": round(ic, 6) if ic is not None else None,
            "top_ret": round(top_ret, 6),
            "holdings": cur_holdings,
            "next_rec": [],  # 由下一轮迭代回填 (实时) / 4.5 节补算最后一天
        })

        # ── 进度指标 (每次迭代算一次, 供回填与推送复用) ──
        progress_pct = round((i + 1) / len(dates) * 100, 1)
        # 精确净收益: (1+raw)*(1−sell−slip)/(1+buy+slip)−1
        _net_factor = (1 - sell_rate - slip_rate) / (1 + buy_rate + slip_rate)
        done_ret = [(1 + d.get("top_ret", 0)) * _net_factor - 1 for d in daily]
        if len(done_ret) >= 20:  # 最少 20 天(≈1个月)才计算Sharpe, 避免小样本爆炸
            _std_r = float(np.std(done_ret))
            sharpe_sofar = float(np.mean(done_ret) / _std_r * np.sqrt(252)) if _std_r > 0 else 0.0
        else:
            sharpe_sofar = 0.0
        fi_live = sorted(feature_gain_totals.items(), key=lambda x: -x[1])[:15]
        fi_live = [{"feature": f, "gain": round(float(v), 6)} for f, v in fi_live]

        # ── 实时 next_rec 前移回填 (零额外训练, 严格满足 next_rec(T)=holdings(T+1)) ──
        # 第 T+1 次迭代算出 holdings(T+1), 恰好等于 model(≤T) 对 T+1 的预测 ≡ 次日推荐(T).
        # 故在迭代 T+1 时回填 daily[T].next_rec 并推送 T 天的完整进度, 实现逐日联动且零成本.
        if progress_callback is not None:
            if prev_state is not None:
                daily[prev_state["idx"]]["next_rec"] = cur_holdings
                progress_callback({
                    "date": str(prev_state["dt"].date()),
                    "ic": round(prev_state["ic"], 4) if prev_state["ic"] is not None else None,
                    "top_ret": round((1 + prev_state["top_ret"]) * _net_factor - 1, 6),  # 精确净收益
                    "progress_pct": prev_state["progress_pct"],
                    "elapsed_seconds": round(time.time() - t_start, 2),
                    "sharpe_sofar": round(sharpe_sofar, 2),
                    "day_n": prev_state["day_n"],
                    "total_days": len(dates),
                    "live_dates": [dd["date"] for dd in daily],
                    "live_ret": [(1 + dd.get("top_ret", 0)) * _net_factor - 1 for dd in daily],
                    "live_ic": [dd["ic"] for dd in daily],
                    "feature_importance": fi_live,
                    "holdings": prev_state["holdings"],
                    "next_rec": cur_holdings,  # 昨天的 next_rec = 今天的 holdings
                    "x_range": x_range,
                })
            # 保存当天状态, 供下一轮回填 next_rec
            d_idx = len(daily) - 1  # 实际 daily 索引, 非循环 i (skip 会导致 i > len(daily)-1)
            prev_state = {
                "idx": d_idx, "dt": dt, "ic": ic, "top_ret": top_ret,
                "progress_pct": progress_pct, "day_n": d_idx + 1,
                "holdings": cur_holdings,
            }

    # ── 推送最后一天进度 (next_rec 待 4.5 补算, 最终由 _render_final 显示) ──
    if progress_callback is not None and prev_state is not None:
        progress_callback({
            "date": str(prev_state["dt"].date()),
            "ic": round(prev_state["ic"], 4) if prev_state["ic"] is not None else None,
            "top_ret": round((1 + prev_state["top_ret"]) * _net_factor - 1, 6),  # 精确净收益
            "progress_pct": 100.0,
            "elapsed_seconds": round(time.time() - t_start, 2),
            "sharpe_sofar": round(sharpe_sofar, 2),
            "day_n": prev_state["day_n"],
            "total_days": len(dates),
            "live_dates": [dd["date"] for dd in daily],
            "live_ret": [(1 + dd.get("top_ret", 0)) * _net_factor - 1 for dd in daily],
            "live_ic": [dd["ic"] for dd in daily],
            "feature_importance": fi_live,
            "holdings": prev_state["holdings"],
            "next_rec": [],
            "x_range": x_range,
        })

    # ── 4.5 补算 next_rec ──
    # 严格时序: 次日推荐(T) = model(≤T) 预测 T+1 ≡ 下一轮当日持仓(T+1)
    # 训练中跳过此步避免双 fit; 全部循环结束后统一补算, 写回 daily[].next_rec
    if params.skip_next_rec:
        print("[NEXT_REC] 跳过 (skip_next_rec=True)", flush=True)
    else:
        print("[NEXT_REC] 开始补算次日推荐 (仅未回填天, 用全量日期找下一交易日)...")
        t_rec = time.time()
        for idx, d in enumerate(daily):
            # 实时前移回填已写入的跳过 (零成本, 与 4.5 计算结果一致)
            if d["next_rec"]:
                continue
            if cancel_event is not None and cancel_event.is_set():
                print("[NEXT_REC] 取消信号, 中断补算")
                break
            dt = pd.Timestamp(d["date"])
            try:
                di = all_dates.index(dt)
            except ValueError:
                d["next_rec"] = []
                continue
            if di + 1 >= len(all_dates):
                # 最后一日无下一交易日 → 用当日特征预测 (用户要求: 不排除, 预测下一日TopN)
                next_dt = dt
            else:
                next_dt = all_dates[di + 1]
            next_df = df_full[df_full["date"] == next_dt]
            train_df2 = df[(df["date"] <= dt) & (df["date"] >= train_start_ts)]
            if len(next_df) == 0 or len(train_df2) == 0:
                d["next_rec"] = []
                continue
            X_tr2 = train_df2[features].copy()
            X_tr2 = X_tr2.groupby(train_df2["code"]).ffill().fillna(0)
            y_tr2 = train_df2["fwd_1d_excess"]
            model_rec = lgb.LGBMRegressor(**lgb_params)
            model_rec.fit(X_tr2, y_tr2)
            X_ne = next_df[features].copy()
            codes_ne = next_df["code"].values
            X_ne["_c"] = codes_ne
            for c in features:
                if X_ne[c].isna().any():
                    X_ne[c] = X_ne.groupby("_c")[c].ffill().fillna(0)
            preds_ne = model_rec.predict(X_ne[features])
            top_ne = np.argsort(preds_ne)[-params.top_n:][::-1]  # 降序
            top_codes_ne = [str(c) for c in next_df.iloc[top_ne]["code"].values]
            top_preds_ne = [round(float(p), 6) for p in preds_ne[top_ne]]
            d["next_rec"] = [{"code": c, "pred_score": s} for c, s in zip(top_codes_ne, top_preds_ne)]
        print(f"[NEXT_REC] 补算完成, 耗时 {time.time()-t_rec:.1f}s ({len(daily)} 天)")

    # ── 虚拟最后一日: 今日(无标签)也入日记, 持仓=上交易日 next_rec (先卖后买) ──
    _today = all_dates[-1]
    if len(dates) == 0 or _today > dates[-1]:
        _prev_next = daily[-1].get("next_rec", []) if daily else []
        daily.append({
            "date": str(_today.date()),
            "holdings": list(_prev_next),
            "next_rec": [],
            "ic": None,
            "top_ret": 0.0,
            "cum_return": None,
        })
        dates_list = list(dates) + [_today]
        print(f"  [VIRTUAL] 末日 {_today.date()}: 持仓=上日next_rec {[h['code'] for h in _prev_next]}", flush=True)
        # 补算末日 next_rec (用全量模型预测下一交易日, 因07-14无特征, 用07-13特征fallback)
        if not params.skip_next_rec and len(daily) > 1:
            _prev = daily[-2]
            _pdt = pd.Timestamp(_prev["date"])
            _train_v = df[(df["date"] <= _pdt) & (df["date"] >= train_start_ts)]
            if len(_train_v) > 0 and _today in df_full["date"].values:
                _Xv = _train_v[features].copy().groupby(_train_v["code"]).ffill().fillna(0)
                _Xnv = df_full[df_full["date"] == _today][features].copy()
                # Bug2修复: 检查末日特征NaN比例, 超过70%则放弃预测 (fillna(0)=垃圾预测)
                _nan_ratio = _Xnv.isna().mean().mean()
                if _nan_ratio > 0.7:
                    print(f"  [VIRTUAL] 末日特征NaN比例过高({_nan_ratio:.0%}), 放弃next_rec预测", flush=True)
                else:
                    for c in features:
                        if _Xnv[c].isna().any():
                            _Xnv[c] = _Xnv[c].fillna(0)
                    try:  # Bug3修复: LightGBM fit 容错
                        _mv = lgb.LGBMRegressor(**lgb_params)
                        _mv.fit(_Xv, _train_v["fwd_1d_excess"])
                        _p = _mv.predict(_Xnv)
                        _tp = np.argsort(_p)[-params.top_n:][::-1]
                        _cnv = df_full[df_full["date"] == _today]["code"].values
                        daily[-1]["next_rec"] = [{"code": str(_cnv[i]), "pred_score": round(float(_p[i]), 6)} for i in _tp]
                        print(f"  [VIRTUAL] 末日next_rec: {[h['code'] for h in daily[-1]['next_rec']]}", flush=True)
                    except Exception as _e:
                        print(f"  [VIRTUAL] 末日next_rec预测失败: {_e}", flush=True)
            elif _today not in df_full["date"].values:
                # Bug1修复: _today 不在 df_full 中 → 管线尚未拉取今日数据
                print(f"  [VIRTUAL] 末日 {_today.date()} 不在训练集中, 跳过next_rec预测", flush=True)
    else:
        dates_list = list(dates)
    x_range = [str(test_start_ts.date()), str(dates[-1].date())] if len(dates) > 0 else \
              [str(test_start_ts.date()), str(test_start_ts.date())]  # Bug4 安全守卫: dates 空不崩溃
    # ── 5. Compile results ───────────────────────────────────────────────
    n_days = len(daily)
    rdf = pd.DataFrame(daily)

    # Guard: no valid prediction days → return zeros
    if n_days == 0:
        max_dd = 0.0
        win_rate = 0.0
        ic_mean = 0.0
        annual_return = 0.0
        sharpe_raw = 0.0
        sharpe_sampled = 0.0
        feature_importance: list[dict] = []
    else:
        # ── 累计账户金额 (精确成本模型: 先卖后买, 买卖各扣滑点) ──
        # 回测起始日: buy_amt = capital / (1+buy_rate+slip_rate), cost_buy = capital - buy_amt
        # 后续每日: sell → gross*(1−sell−slip) → buy → net/(1+buy+slip), cost=sell_cost+buy_cost
        capital = params.initial_capital
        acc = capital / (1 + buy_rate + slip_rate)  # 起始日买入后实际持仓金额
        for i, d in enumerate(daily):
            raw_ret = float(rdf["top_ret"].iloc[i])
            # 按收盘价卖出: gross = acc * (1+raw_ret)
            gross_sell = acc * (1.0 + raw_ret)
            cost_sell = gross_sell * (sell_rate + slip_rate)
            net_sell = gross_sell - cost_sell
            # 按收盘价买入次日Top3: buy_amt = net_sell / (1+buy_rate+slip_rate)
            buy_amt = net_sell / (1.0 + buy_rate + slip_rate)
            cost_buy = net_sell - buy_amt
            cost_total = cost_sell + cost_buy
            acc = buy_amt
            d["cum_return"] = round(acc, 2)
            d["cost_rmb"] = round(cost_total, 2)
            # 单日净收益率(供Sharpe等指标)
            d["top_ret_net"] = round((1.0 + raw_ret) * (1.0 - sell_rate - slip_rate) / (1.0 + buy_rate + slip_rate) - 1.0, 8)

        # 总成本统计
        total_cost_rmb = sum(d.get("cost_rmb", 0) for d in daily)

        # Sharpe (基于净收益率 net=raw−cost)
        net_ret = pd.Series([d.get("top_ret_net", d.get("top_ret", 0) - buy_rate - sell_rate - slip_rate) for d in daily])
        sharpe_raw = float(
            net_ret.mean() / net_ret.std() * np.sqrt(252)
            if net_ret.std() > 0 else 0.0
        )

        # Sharpe (sampled — 非重叠采样: 每 sample_interval 日取一个, 与 v23/canonical 一致)
        net_s = net_ret.iloc[:: params.sample_interval]
        _std_s = net_s.std()
        sharpe_sampled = float(
            net_s.mean() / _std_s * np.sqrt(252 / params.sample_interval)
        ) if _std_s and _std_s > 0 else 0.0

        # Max drawdown (基于净收益)
        cum_net = (1 + net_ret).cumprod()
        max_dd = float((cum_net / cum_net.expanding().max() - 1).min())

        # Win rate (原始超额, 与成本无关)
        win_rate = float((rdf["top_ret"] > 0).mean())

        # IC mean (剔除退化日 ic=None, 避免被 0 拉低)
        _ic_valid = rdf["ic"].dropna()
        ic_mean = float(_ic_valid.mean()) if len(_ic_valid) > 0 else 0.0

        # Annual return (净收益)
        annual_return = float(
            (1 + net_ret).prod() ** (252 / n_days) - 1 if n_days > 0 else 0.0
        )

    elapsed = round(time.time() - t_start, 2)

    # Average feature importance across training days
    n_trained_days = max(len(dates), 1)
    feature_importance = [
        {"feature": f, "gain": round(v / n_trained_days, 8)}
        for f, v in sorted(feature_gain_totals.items(), key=lambda x: -x[1])
    ]

    # ── 6. Persist to SQLite ─────────────────────────────────────────────
    run_name = params.run_name or f"run_{DateType.today().isoformat()}_{int(time.time())}"
    run_params = {
        "name": run_name,
        "train_start": str(params.train_start),
        "test_start": str(params.test_start),
        "test_end": str(params.test_end) if params.test_end else None,
        "universe_source": params.universe_source,
        "buy_pct": params.buy_pct,
        "sell_pct": params.sell_pct,
        "slip_pct": params.slip_pct,
        "top_n": params.top_n,
        "n_features": len(features),
        "min_train_days": params.min_train_days,
        "sample_interval": params.sample_interval,
        "model_params": lgb_params,
    }
    run_results = {
        "n_days": n_days,
        "sharpe_raw": sharpe_raw,
        "sharpe_sampled": sharpe_sampled,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "ic_mean": ic_mean,
        "annual_return": annual_return,
        "elapsed_seconds": elapsed,
        "daily_returns": daily,
        "feature_importance": feature_importance,
    }
    run_id = save_run(run_params, run_results)

    # Also persist the full TrainParams as last settings
    save_last_train_params(params.model_dump(mode="json"))

    return TrainResult(
        run_id=run_id,
        name=run_name,
        n_days=n_days,
        n_features=len(features),
        sharpe_raw=sharpe_raw,
        sharpe_sampled=sharpe_sampled,
        max_dd=max_dd,
        win_rate=win_rate,
        ic_mean=ic_mean,
        annual_return=annual_return,
        elapsed_seconds=elapsed,
        daily_returns=daily,
        feature_importance=feature_importance,
        model_params=lgb_params,
        date_range=x_range,
        top_n=params.top_n,
        train_start=str(params.train_start),
        test_start=str(params.test_start),
        test_end=str(params.test_end) if params.test_end else None,
        buy_pct=params.buy_pct,
        sell_pct=params.sell_pct,
        slip_pct=params.slip_pct,
        initial_capital=params.initial_capital,
    )


# ── Quick smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print(f"DB ready at: {_WEB_DIR / 'data' / 'web.db'}")
    print(f"Training data: {_TRAIN_DATA_PATH}  (exists={_TRAIN_DATA_PATH.exists()})")

    # Quick check of existing runs
    from backend.database import get_runs
    runs = get_runs(limit=5)
    print(f"Existing runs: {len(runs)}")
    for r in runs:
        print(f"  [{r['id']}] {r['name']}  sharpe={r['sharpe_raw']}  status={r['status']}")
