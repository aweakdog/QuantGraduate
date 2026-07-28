"""
Walk-Forward 每日扩展回测 — v23 重写版 (独立脚本, 不改动原 v22/v23)

相对 wf_daily_expanding.py 的演进:
  - 默认训练集 = training_data_v23.parquet (v23 原生, 281 特征含 vol_ma20)
  - 所有参数均可命令行设置 (训练起始/预测起始/预测终点/TopN/成本/最短训练天数/
    非重叠采样间隔/反向/训练集来源/模型超参)
  - 训练集来源: 关注圈(默认池 watchlist*.json) / 自选股(Web/data/self_selected.json) /
                自定义(--custom-codes 文件或逗号串)
  - 输出独立文件 wf_v23_*.json, 不污染原 wf_daily_vXX 输出
  - 模型超参全部可自由设置 (默认基准 n=400 d=4 lr=0.03, 仅作基线非锁定);
             新股<2年替换(数据层已落地, 此处仅按来源过滤);
             跑前参数一致性校验 (防配置静默失效, 非取值锁)

明确不改动: wf_daily_expanding.py / build_v23.py / training_data_vXX.parquet
(如需基于 v22 对比, 用 --train-data training_data_v22.parquet 即可, 只读不写)
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── 路径 (从本文件位置推导, 不写死绝对路径) ──
SCRIPT_DIR = Path(__file__).resolve().parent          # .../quant-strategy/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../quant-strategy
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
UNIVERSE_DIR = DATA_DIR / "universe"
WEB_DATA_DIR = PROJECT_ROOT / "Web" / "data"

# ── 列排除规则 (与 canonical / app.py 一致) ──
LABEL_RAW = "fwd_1d_exec_ret"
LABEL = "fwd_1d_excess"
LEAKAGE_FEATS = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
SKIP_COLS = {
    "date", "code",
    "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret", "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret", "fwd_1d_exec_ret",
}


def _is_valid_feat(f: str) -> bool:
    return "_21d" not in f and not f.endswith("_cross")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-Forward 每日扩展回测 (v23 重写版, 全参数可设)")
    # ── 数据 ──
    p.add_argument("--train-data", default="training_data_v23.parquet",
                   help="训练集parquet (位于 data/processed); 默认 v23")
    # ── 时间窗口 (训练起始等, 都要允许设置) ──
    p.add_argument("--train-start", default=None,
                   help="训练数据起点(含); 默认=None=全历史(数据集最早日)")
    p.add_argument("--test-start", default="2025-09-01",
                   help="回测起点(含); 默认 2025-09-01")
    p.add_argument("--test-end", default=None,
                   help="回测终点(含); 默认=None=最新日")
    # ── 组合 / 成本 ──
    p.add_argument("--top-n", type=int, default=3, help="每日持仓数; 默认 3")
    p.add_argument("--cost-bps", type=float, default=6.0,
                   help="单边交易成本 (bps, 仅卖出计); 默认 6=0.06% (v23 成本模型)")
    p.add_argument("--min-train-days", type=int, default=250,
                   help="最短训练天数(窗口内不足则跳过该日); 默认 250")
    p.add_argument("--sample-interval", type=int, default=5,
                   help="非重叠 Sharpe 采样间隔(日); 默认 5")
    p.add_argument("--reverse", action="store_true",
                   help="反向策略: 选预测分最低的 TopN 买入")
    # ── 训练集来源 ──
    p.add_argument("--universe", default="关注圈",
                   choices=["关注圈", "自选股", "自定义"],
                   help="训练集来源; 默认 关注圈(默认池)")
    p.add_argument("--custom-codes", default=None,
                   help="自定义代码: 文件路径(每行一个) 或 逗号分隔(000049.SZ,600000.SH)")
    # ── 模型超参 (默认基准值, 可自由设置; 400/4/0.03 仅为基线) ──
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def _resolve_universe(args: argparse.Namespace) -> list[str]:
    """返回大写代码列表.

    关注圈 = universe/watchlist*.json (规范名优先, 回退 watchlist_216.json)
    自选股 = Web/data/self_selected.json (不存在则回退关注圈)
    自定义 = --custom-codes (文件或逗号串)
    """
    def _read_json_codes(path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            data = json.load(open(path, encoding="utf-8")).get("watchlist", [])
        except Exception:
            return []
        return [str(r.get("code", "")).strip().upper()
                for r in data if str(r.get("code", "")).strip()]

    def _watchlist_codes() -> list[str]:
        canonical = UNIVERSE_DIR / "watchlist.json"
        p = canonical if canonical.exists() else UNIVERSE_DIR / "watchlist_216.json"
        return _read_json_codes(p)

    if args.universe == "自选股":
        codes = _read_json_codes(WEB_DATA_DIR / "self_selected.json")
        return codes if codes else _watchlist_codes()
    if args.universe == "自定义":
        raw: list[str] = []
        if args.custom_codes:
            fp = Path(args.custom_codes)
            if fp.exists():
                # 兼容两种格式: 每行一个 或 单行逗号分隔 (或混合)
                for ln in fp.read_text(encoding="utf-8").splitlines():
                    raw.extend(t.strip() for t in ln.split(",") if t.strip())
            else:
                raw = [c.strip() for c in args.custom_codes.split(",") if c.strip()]
        return [c.upper() for c in raw]
    return _watchlist_codes()


def _param_consistency_check(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> None:
    """跑前一致性校验: 实际 fit 一次(最后90天窗口), 断言 实际参数==设定值,
    防止配置静默失效 (参数未正确传入模型)。非取值锁 — 任意超参均可自由设置。"""
    print("[check] 参数一致性校验 (最后90天窗口实际训练)...")
    all_d = sorted(df["date"].unique())
    cut = all_d[-90]
    sub = df[df["date"] >= cut].copy()
    Xs = sub.groupby("code")[features].transform(lambda s: s.ffill().fillna(0))
    ys = sub[LABEL]
    m = lgb.LGBMRegressor(**_lgb_params(args))
    m.fit(Xs, ys)
    pp = m.get_params()
    print(f"[check] 实际训练参数: n_estimators={pp['n_estimators']} "
          f"max_depth={pp['max_depth']} lr={pp['learning_rate']}")
    assert pp["n_estimators"] == args.n_estimators and \
        pp["max_depth"] == args.max_depth and \
        abs(pp["learning_rate"] - args.learning_rate) < 1e-9, \
        "❌ 参数未正确传入模型 (配置静默失效), 中止以防重跑浪费"
    print("[check] ✅ 参数一致")


def _lgb_params(args: argparse.Namespace) -> dict:
    return dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def main() -> None:
    args = _parse_args()
    train_path = PROCESSED_DIR / args.train_data
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")
    cost = args.cost_bps / 10000.0   # bps → decimal (仅卖出计)
    lgb_params = _lgb_params(args)

    print("Loading...")
    df = pd.read_parquet(train_path)
    df["date"] = pd.to_datetime(df["date"])
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[LABEL_RAW])
    df[LABEL] = df.groupby("date")[LABEL_RAW].transform(lambda x: x - x.mean())

    if args.train_start:
        ts = pd.Timestamp(args.train_start)
        df = df[df["date"] >= ts].copy()
        print(f"  [train-start] 过滤训练数据 >= {ts.date()} → {len(df):,} rows")

    # ── 训练集来源过滤 (关注圈 / 自选股 / 自定义) ──
    # 新股<2年替换已落地于 parquet 数据层, 此处仅按来源过滤行, 不影响该铁律。
    codes = _resolve_universe(args)
    if not codes:
        raise SystemExit("❌ 训练集来源未解析到任何代码 (自定义请检查 --custom-codes)")
    _all = set(df["code"].astype(str).str.upper())
    _sel = set(codes)
    _excl = _sel - _all
    if _excl:
        print(f"[UNIVERSE] 来源『{args.universe}』: {len(_excl)}/{len(_sel)} 只"
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
    _mask = df["date"] >= pd.Timestamp(args.test_start)
    if args.test_end:
        _mask &= df["date"] <= pd.Timestamp(args.test_end)
    dates = sorted(df[_mask]["date"].unique())
    if not dates:
        raise SystemExit(f"ERROR: 无数据 after {args.test_start}")

    top_n = args.top_n
    print(f"\nWalk-Forward: {len(dates)} prediction days "
          f"({dates[0].date()} ~ {dates[-1].date()})")
    print(f"Model: n={args.n_estimators} d={args.max_depth} lr={args.learning_rate}, "
          f"MIN_TRAIN={args.min_train_days}, cost={args.cost_bps}bps, "
          f"universe={args.universe}, "
          f"{'REVERSE Bottom'+str(top_n) if args.reverse else 'Top'+str(top_n)}")

    # ── 跑前参数一致性校验 (非锁, 任意超参可设) ──
    _param_consistency_check(df, features, args)

    daily_results: list[dict] = []
    t0 = datetime.now()

    for day_idx, pred_date in enumerate(dates):
        train_df = df[df["date"] < pred_date]
        if train_df["date"].nunique() < args.min_train_days:
            continue

        X_train = train_df.groupby("code")[features].transform(
            lambda s: s.ffill().fillna(0))
        y_train = train_df[LABEL]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)

        test_mask = df["date"] == pred_date
        X_test = df.loc[test_mask, features].copy()
        y_test = df.loc[test_mask, LABEL].copy()
        codes_test = df.loc[test_mask, "code"].values
        for c in features:
            if X_test[c].isna().any():
                X_test[c] = X_test[c].fillna(0)
        preds = model.predict(X_test)

        # IC (退化预测 spearmanr 返回 nan → 保留 nan, summary 阶段 dropna 剔除,
        #      不兜底成 0 以免拉低 IC 均值; 与 canonical wf_daily_expanding.py 一致)
        if len(preds) > 5:
            ic, _ = spearmanr(preds, y_test)
            ic = np.nan if (ic is None or np.isnan(ic)) else float(ic)
        else:
            ic = np.nan

        test_df = pd.DataFrame({
            "code": codes_test, "pred": preds, "label": y_test.values})
        n_test = len(test_df)
        k = min(top_n, n_test)
        if args.reverse:
            sel = test_df.nsmallest(k, "pred")     # 反向: 买预测分最低
        else:
            sel = test_df.nlargest(k, "pred")
        top_ret = float(sel["label"].mean() - cost)  # 仅卖出成本

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
    rdf = pd.DataFrame(daily_results)
    if rdf.empty:
        print("⚠️ 无有效预测日 (窗口/最短训练天数过滤后为空)")
        return
    rdf["ic"] = rdf["ic"].astype(float)
    valid = rdf.dropna(subset=["ic"])
    if valid.empty:
        print("⚠️ 所有预测日的 IC 均为 NaN (测试特征缺失/退化), 无可汇总指标。"
              "请检查测试窗口是否落在数据有效区间内")
        return

    pf = pd.DataFrame(rdf[["date", "top_ret"]].copy())
    pf["date"] = pd.to_datetime(pf["date"])
    pf = pf.sort_values("date")
    pf["cum_raw"] = (1 + pf["top_ret"]).cumprod() - 1

    # 非重叠 Sharpe (每 sample_interval 日采样)
    pf_sampled = pf.iloc[::args.sample_interval]
    sharpe = float(pf_sampled["top_ret"].mean() / pf_sampled["top_ret"].std()
                   * np.sqrt(252 / args.sample_interval)) \
        if pf_sampled["top_ret"].std() > 0 else 0.0
    # 重叠 Sharpe (raw)
    sharpe_raw = float(pf["top_ret"].mean() / pf["top_ret"].std() * np.sqrt(252)) \
        if pf["top_ret"].std() > 0 else 0.0

    cum_series = (1 + pf["top_ret"]).cumprod()
    max_dd = float((cum_series / cum_series.expanding().max() - 1).min())
    win_rate = float((pf["top_ret"] > 0).mean())
    total_cost = len(rdf) * cost
    n_days = len(rdf)
    ann_ret = float((1 + pf["top_ret"]).prod() ** (252 / n_days) - 1) if n_days > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD RESULTS (v23, "
          f"{'REVERSE Bottom'+str(top_n) if args.reverse else 'Top'+str(top_n)}, "
          f"{args.test_start} ~ {args.test_end or 'latest'}, universe={args.universe})")
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

    # ── 输出 (v23 专属命名, 不污染 wf_daily_vXX) ──
    suffix = ""
    if args.train_start:
        suffix += f"_tr{args.train_start}"
    if args.test_start != "2025-09-01":
        suffix += f"_ts{args.test_start}"
    if args.test_end:
        suffix += f"_te{args.test_end}"
    if top_n != 3:
        suffix += f"_top{top_n}"
    if args.universe != "关注圈":
        suffix += f"_{args.universe}"
    if abs(args.cost_bps - 6.0) > 1e-9:
        suffix += f"_c{args.cost_bps:.0f}"
    tag = args.train_data.replace("training_data_", "").replace(".parquet", "")
    out_name = (f"wf_v23_{tag}_reverse{suffix}.json" if args.reverse
                else f"wf_v23_{tag}{suffix}.json")
    out_path = PROCESSED_DIR / out_name
    output = {
        "version": "v23-rewrite",
        "label": LABEL,
        "model": f"LightGBM Regression n={args.n_estimators} d={args.max_depth} "
                 f"lr={args.learning_rate}, {args.n_jobs}线程CPU, daily expanding, "
                 f"{'REVERSE Bottom'+str(top_n) if args.reverse else 'Top'+str(top_n)}",
        "features": len(features),
        "universe": args.universe,
        "period": f"{rdf['date'].iloc[0]} ~ {rdf['date'].iloc[-1]}",
        "n_prediction_days": n_days,
        "summary": {
            "ic_mean": round(valid["ic"].mean(), 4),
            "ic_std": round(valid["ic"].std(), 4),
            "top_excess_mean": round(float(rdf["top_ret"].mean()), 6),
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


if __name__ == "__main__":
    main()
