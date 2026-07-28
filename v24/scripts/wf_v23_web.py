"""
Walk-Forward 每日扩展回测 — Web 版 benchmark (wf_v23_web)

相对 wf_v23.py 的 Web 化改造:
  - 所有路径经 backend.paths 单一真相源解析 (训练数据=latest_training_data(),
    输出目录=processed_dir()), 不再自行推导绝对路径, 与 Web 永远一致
  - 训练集来源默认 '训练池' (Web/data/train_pool.json, 关注圈的用户子集);
    支持 训练池/关注圈/自选股/自定义, 统一经 backend.paths.load_universe_codes 解析
    → 用户在 Web「关注圈/自选股」Tab 里指定的训练池, 此脚本直接复用, 不需重复传参
  - 暴露 run_benchmark(params: dict) -> dict 可导入函数, Web 后端可直接调用
    (非仅 CLI); CLI 仅是其薄包装
  - 输出 data/processed/wf_v23_web[_<opts>][_reverse].json,
    结构兼容 backend.backtest_results.discover_backtests 的发现机制 (首页/回测结果 Tab 可读)
  - 模型 n_jobs 固定 32 (Web 惯例, 纯内部不暴露)

明确不改动: wf_v23.py / wf_daily_expanding.py / training_data_vXX.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── 定位 Web 后端 (backend.paths 单一真相源) ───────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent            # .../quant-strategy/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                         # .../quant-strategy
WEB_DIR = PROJECT_ROOT / "Web"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from backend.paths import (                              # noqa: E402
    latest_training_data,
    processed_dir,
    load_universe_codes,
)

# ── 列排除规则 (与 canonical / app.py / wf_v23.py 一致) ──
LABEL_RAW = "fwd_1d_exec_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {
    "date", "code",
    "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
}

# 默认参数 (Web 基准, 与 Web 训练页 model_params 默认值一致)
DEFAULTS = dict(
    train_data=None,            # None → latest_training_data()
    train_start=None,
    test_start="2025-09-01",
    test_end=None,
    top_n=3,
    cost_bps=6.0,
    min_train_days=250,
    sample_interval=5,
    reverse=False,
    universe="训练池",          # Web 默认: 用户指定的训练池
    custom_codes=None,
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    random_state=42,
    n_jobs=32,                  # Web 惯例: 固定 32, 不暴露
)


def _is_valid_feat(f: str) -> bool:
    return "_21d" not in f and not f.endswith("_cross")


def _lgb_params(p: dict) -> dict:
    return dict(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        min_child_samples=p["min_child_samples"],
        random_state=p["random_state"],
        n_jobs=p["n_jobs"],
        verbosity=-1,
    )


def _param_consistency_check(df: pd.DataFrame, features: list[str], p: dict) -> None:
    """跑前一致性校验: 实际 fit 一次(最后90天窗口), 断言 实际参数==设定值,
    防止配置静默失效 (参数未正确传入模型)。非取值锁 — 任意超参均可自由设置。"""
    print("[check] 参数一致性校验 (最后90天窗口实际训练)...")
    all_d = sorted(df["date"].unique())
    cut = all_d[-90]
    sub = df[df["date"] >= cut].copy()
    Xs = sub.groupby("code")[features].transform(lambda s: s.ffill().fillna(0))
    ys = sub[LABEL]
    m = lgb.LGBMRegressor(**_lgb_params(p))
    m.fit(Xs, ys)
    pp = m.get_params()
    print(f"[check] 实际训练参数: n_estimators={pp['n_estimators']} "
          f"max_depth={pp['max_depth']} lr={pp['learning_rate']}")
    assert pp["n_estimators"] == p["n_estimators"] and \
        pp["max_depth"] == p["max_depth"] and \
        abs(pp["learning_rate"] - p["learning_rate"]) < 1e-9, \
        "❌ 参数未正确传入模型 (配置静默失效), 中止以防重跑浪费"
    print("[check] ✅ 参数一致")


def run_benchmark(params: dict | None = None) -> dict:
    """Web 化基准回测入口 (可导入).

    params: 覆盖默认值的字典 (键见 DEFAULTS)。例如:
        run_benchmark({"universe": "训练池", "top_n": 5, "reverse": False})
    返回输出 dict (同时写入 data/processed/wf_v23_web_*.json)。
    """
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})

    train_path = latest_training_data() if not p["train_data"] \
        else (processed_dir() / p["train_data"])
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    cost = p["cost_bps"] / 10000.0

    print("Loading...")
    df = pd.read_parquet(train_path)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[LABEL_RAW])
    df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

    if p["train_start"]:
        ts = pd.Timestamp(p["train_start"])
        df = df[df["date"] >= ts].copy()
        print(f"  [train-start] 过滤训练数据 >= {ts.date()} → {len(df):,} rows")

    # ── 训练集来源过滤 (经 backend.paths.load_universe_codes 统一解析) ──
    codes = load_universe_codes(p["universe"], p["custom_codes"])
    if not codes:
        raise SystemExit(f"❌ 训练集来源『{p['universe']}』未解析到任何代码")
    _all = set(df["code"].astype(str).str.upper())
    _sel = set(codes)
    _excl = _sel - _all
    if _excl:
        print(f"[UNIVERSE] 来源『{p['universe']}』: {len(_excl)}/{len(_sel)} 只"
              f"不在训练数据集中, 已跳过: {sorted(_excl)[:10]}"
              f"{'...' if len(_excl) > 10 else ''}")
    df = df[df["code"].astype(str).str.upper().isin(_sel)]
    if len(df) == 0:
        raise SystemExit("❌ 过滤后无有效股票 — 请确认来源代码在训练池内")

    all_cols = [c for c in df.columns if c not in SKIP_COLS and _is_valid_feat(c)]
    features = [f for f in all_cols if f not in LEAKAGE_FEATS]
    print(f"  {len(df):,} rows, {df['code'].nunique()} codes, {len(features)} features")
    print(f"  Dates: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # ── Walk-Forward 预测日 ──
    _mask = df["date"] >= pd.Timestamp(p["test_start"])
    if p["test_end"]:
        _mask &= df["date"] <= pd.Timestamp(p["test_end"])
    dates = sorted(df[_mask]["date"].unique())
    if not dates:
        raise SystemExit(f"ERROR: 无数据 after {p['test_start']}")

    top_n = p["top_n"]
    print(f"\nWalk-Forward: {len(dates)} prediction days "
          f"({dates[0].date()} ~ {dates[-1].date()})")
    print(f"Model: n={p['n_estimators']} d={p['max_depth']} lr={p['learning_rate']}, "
          f"MIN_TRAIN={p['min_train_days']}, cost={p['cost_bps']}bps, "
          f"universe={p['universe']}, "
          f"{'REVERSE Bottom'+str(top_n) if p['reverse'] else 'Top'+str(top_n)}")

    _param_consistency_check(df, features, p)

    daily_results: list[dict] = []
    t0 = datetime.now()

    for day_idx, pred_date in enumerate(dates):
        train_df = df[df["date"] < pred_date]
        if train_df["date"].nunique() < p["min_train_days"]:
            continue

        X_train = train_df.groupby("code")[features].transform(
            lambda s: s.ffill().fillna(0))
        y_train = train_df[LABEL]

        model = lgb.LGBMRegressor(**_lgb_params(p))
        model.fit(X_train, y_train)

        test_mask = df["date"] == pred_date
        X_test = df.loc[test_mask, features].copy()
        y_test = df.loc[test_mask, LABEL].copy()
        codes_test = df.loc[test_mask, "code"].values
        for c in features:
            if X_test[c].isna().any():
                X_test[c] = X_test[c].fillna(0)
        preds = model.predict(X_test)

        if len(preds) > 5:
            ic, _ = spearmanr(preds, y_test)
            ic = np.nan if (ic is None or np.isnan(ic)) else float(ic)
        else:
            ic = np.nan

        test_df = pd.DataFrame({
            "code": codes_test, "pred": preds, "label": y_test.values})
        n_test = len(test_df)
        k = min(top_n, n_test)
        if p["reverse"]:
            sel = test_df.nsmallest(k, "pred")
        else:
            sel = test_df.nlargest(k, "pred")
        top_ret = float(sel["label"].mean() - cost)

        actual_topk = set(np.argsort(y_test)[-k:])
        pred_topk = set(np.argsort(preds)[-k:])
        hit = len(actual_topk & pred_topk) / k if k > 0 else 0.0

        daily_results.append({
            "date": str(pred_date.date()),
            "n_train": len(X_train),
            "n_test": n_test,
            "ic": (round(float(ic), 4) if not np.isnan(ic) else None),
            "top_ret": round(top_ret, 6),
            "hit_rate": round(hit, 3),
        })
        if day_idx % 20 == 0 or day_idx == len(dates) - 1:
            el = (datetime.now() - t0).total_seconds()
            print(f"  [{day_idx+1}/{len(dates)}] {pred_date.date()} "
                  f"IC={ic:.4f} top{top_n}={top_ret:+.4f} "
                  f"train={len(X_train):,d} ({el:.0f}s)")

    # ── Summary ──
    if not daily_results:
        print("⚠️ 无有效预测日 (窗口/最短训练天数过滤后为空)")
        return {}
    rdf = pd.DataFrame(daily_results)
    rdf["ic"] = rdf["ic"].astype(float)
    valid = rdf.dropna(subset=["ic"])
    if valid.empty:
        print("⚠️ 所有预测日的 IC 均为 NaN, 无可汇总指标")
        return {}

    pf = pd.DataFrame(rdf[["date", "top_ret"]].copy())
    pf["date"] = pd.to_datetime(pf["date"])
    pf = pf.sort_values("date")
    pf["cum_raw"] = (1 + pf["top_ret"]).cumprod() - 1

    pf_sampled = pf.iloc[::p["sample_interval"]]
    sharpe = float(pf_sampled["top_ret"].mean() / pf_sampled["top_ret"].std()
                   * np.sqrt(252 / p["sample_interval"])) \
        if pf_sampled["top_ret"].std() > 0 else 0.0
    sharpe_raw = float(pf["top_ret"].mean() / pf["top_ret"].std() * np.sqrt(252)) \
        if pf["top_ret"].std() > 0 else 0.0

    cum_series = (1 + pf["top_ret"]).cumprod()
    max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
    win_rate = float((pf["top_ret"] > 0).mean())
    total_cost = len(rdf) * cost
    n_days = len(rdf)
    ann_ret = float((1 + pf["top_ret"]).prod() ** (252 / n_days) - 1) if n_days > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD RESULTS (wf_v23_web, "
          f"{'REVERSE Bottom'+str(top_n) if p['reverse'] else 'Top'+str(top_n)}, "
          f"{p['test_start']} ~ {p['test_end'] or 'latest'}, universe={p['universe']})")
    print(f"{'='*60}")
    print(f"  Days: {n_days}")
    print(f"  IC: mean={valid['ic'].mean():.4f} std={valid['ic'].std():.4f}")
    print(f"  Top{top_n} excess: mean={rdf['top_ret'].mean():+.6f}")
    print(f"  Cum return (raw): {pf['cum_raw'].iloc[-1]*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f} (raw={sharpe_raw:.2f})")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Hit rate: {rdf['hit_rate'].mean():.3f}")
    print(f"  Est total cost: {total_cost*100:.2f}%")

    rdf["month"] = rdf["date"].str[:7]
    monthly_ic = rdf.groupby("month")["ic"].mean()
    print("\n  Monthly IC:")
    for m, v in monthly_ic.items():
        if pd.isna(v):
            continue
        bar = "+" * max(1, int(abs(v) * 1000)) if v > 0 else "-" * max(1, int(abs(v) * 1000))
        print(f"    {m}: {v:+.4f} {bar}")

    print(f"\n  Annualized return: {ann_ret*100:.1f}%")
    print(f"  Total elapsed: {(datetime.now()-t0).total_seconds():.0f}s")

    # ── 输出 (wf_v23_web 专属命名, 兼容 discover_backtests) ──
    suffix = ""
    if p["train_start"]:
        suffix += f"_tr{p['train_start']}"
    if p["test_start"] != "2025-09-01":
        suffix += f"_ts{p['test_start']}"
    if p["test_end"]:
        suffix += f"_te{p['test_end']}"
    if top_n != 3:
        suffix += f"_top{top_n}"
    if p["universe"] != "训练池":
        suffix += f"_{p['universe']}"
    if abs(p["cost_bps"] - 6.0) > 1e-9:
        suffix += f"_c{p['cost_bps']:.0f}"
    tag = Path(train_path).name.replace("training_data_", "").replace(".parquet", "")
    out_name = (f"wf_v23_web_{tag}_reverse{suffix}.json" if p["reverse"]
                else f"wf_v23_web_{tag}{suffix}.json")
    out_path = processed_dir() / out_name
    output = {
        "version": "wf_v23_web",
        "label": LABEL,
        "model": f"LightGBM Regression n={p['n_estimators']} d={p['max_depth']} "
                 f"lr={p['learning_rate']}, {p['n_jobs']}线程CPU, daily expanding, "
                 f"{'REVERSE Bottom'+str(top_n) if p['reverse'] else 'Top'+str(top_n)}",
        "features": len(features),
        "universe": p["universe"],
        "period": f"{rdf['date'].iloc[0]} ~ {rdf['date'].iloc[-1]}",
        "n_prediction_days": n_days,
        "summary": {
            "ic_mean": round(valid["ic"].mean(), 4),
            "ic_std": round(valid["ic"].std(), 4),
            "top_excess_mean": round(float(rdf["top_ret"].mean()), 6),
            "top3_excess_mean": round(float(rdf["top_ret"].mean()), 6),
            "cum_return_pct": round(float(pf["cum_raw"].iloc[-1]) * 100, 1),
            "annualized_return_pct": round(ann_ret * 100, 1),
            "sharpe": round(sharpe, 2),
            "sharpe_raw": round(sharpe_raw, 2),
            "max_dd_pct": round(max_dd * 100, 1),
            "win_rate_pct": round(win_rate * 100, 1),
            "hit_rate": round(float(rdf["hit_rate"].mean()), 3),
            "total_cost_est_pct": round(total_cost * 100, 2),
        },
        "monthly_ic": {str(k): round(float(v), 4) for k, v in monthly_ic.items()},
        "daily": daily_results[-10:],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return output


def _parse_args() -> argparse.Namespace:
    d = DEFAULTS
    p = argparse.ArgumentParser(
        description="Walk-Forward 每日扩展回测 (Web 版 benchmark, 经 backend.paths)")
    p.add_argument("--train-data", default=None,
                   help="训练集parquet (位于 data/processed); 默认=最新版")
    p.add_argument("--train-start", default=None, help="训练数据起点(含); 默认全历史")
    p.add_argument("--test-start", default=d["test_start"], help="回测起点(含)")
    p.add_argument("--test-end", default=None, help="回测终点(含); 默认最新日")
    p.add_argument("--top-n", type=int, default=d["top_n"], help="每日持仓数")
    p.add_argument("--cost-bps", type=float, default=d["cost_bps"], help="单边交易成本(bps,仅卖出)")
    p.add_argument("--min-train-days", type=int, default=d["min_train_days"])
    p.add_argument("--sample-interval", type=int, default=d["sample_interval"])
    p.add_argument("--reverse", action="store_true", help="反向: 选预测分最低")
    p.add_argument("--universe", default=d["universe"],
                   choices=["训练池", "关注圈", "自选股", "自定义"],
                   help="训练集来源; 默认 训练池(Web/data/train_pool.json)")
    p.add_argument("--custom-codes", default=None,
                   help="自定义代码: 文件路径或逗号串 (universe=自定义时生效)")
    p.add_argument("--n-estimators", type=int, default=d["n_estimators"])
    p.add_argument("--max-depth", type=int, default=d["max_depth"])
    p.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    p.add_argument("--num-leaves", type=int, default=d["num_leaves"])
    p.add_argument("--subsample", type=float, default=d["subsample"])
    p.add_argument("--colsample-bytree", type=float, default=d["colsample_bytree"])
    p.add_argument("--min-child-samples", type=int, default=d["min_child_samples"])
    p.add_argument("--random-state", type=int, default=d["random_state"])
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    params = dict(
        train_data=a.train_data,
        train_start=a.train_start,
        test_start=a.test_start,
        test_end=a.test_end,
        top_n=a.top_n,
        cost_bps=a.cost_bps,
        min_train_days=a.min_train_days,
        sample_interval=a.sample_interval,
        reverse=a.reverse,
        universe=a.universe,
        custom_codes=a.custom_codes,
        n_estimators=a.n_estimators,
        max_depth=a.max_depth,
        learning_rate=a.learning_rate,
        num_leaves=a.num_leaves,
        subsample=a.subsample,
        colsample_bytree=a.colsample_bytree,
        min_child_samples=a.min_child_samples,
        random_state=a.random_state,
    )
    run_benchmark(params)


if __name__ == "__main__":
    main()
