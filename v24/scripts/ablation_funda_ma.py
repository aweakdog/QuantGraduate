"""
基本面 MA5/20 衍生消融实验
基本面季度变化, MA5/20 无意义 → 验证剔除后效果

3个区间 × 有/无 = 6次回测
"""
import sys, time, json, os
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Web"))

import pandas as pd
import numpy as np

from backend.trainer import train, TrainParams
from backend.paths import latest_training_data
from backend.database import init_db

_INITIAL = 2_000_000

TEST_PERIODS = [
    ("2025H1", date(2025, 1, 2), date(2025, 6, 30)),
    ("2025H2", date(2025, 7, 1), date(2025, 12, 31)),
    ("2026Q2", date(2026, 4, 1), date(2026, 6, 30)),
]

BASE_PARAMS = dict(
    train_start=date(2023, 1, 1),
    buy_pct=0.03, sell_pct=0.03, slip_pct=0.01,
    top_n=3, sample_interval=5,
    n_estimators=400, max_depth=4, learning_rate=0.03,
    initial_capital=_INITIAL,
    universe_source="关注圈",
)

# 基本面原始列
_FUNDA_RAW = ['pe', 'pb', 'revenue', 'profit', 'eps', 'bps', 'debt_ratio',
              'gross_margin', 'roe', 'total_assets']
# 基本面 MA5/20 (20个)
_FUNDA_MA = [f'{r}_{s}' for r in _FUNDA_RAW for s in ('ma5', 'ma20')]

print(f"剔除基本面MA5/20特征: {len(_FUNDA_MA)} 个")


def _default_feats(exclude_funda_ma: bool) -> list[str]:
    cols = pd.read_parquet(latest_training_data()).columns.tolist()
    _SKIP = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
             "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret"}
    _LEAK = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
    _EXCLUDED = {"mtss_1d", "mtss_z", "mtss_1d_ma5", "mtss_z_ma5",
                 "mtss_1d_ma20", "mtss_z_ma20"}
    feats = [c for c in cols if c not in _SKIP and "_21d" not in c
             and not c.endswith("_cross") and c not in _LEAK
             and c not in _EXCLUDED]
    if exclude_funda_ma:
        feats = [c for c in feats if c not in _FUNDA_MA]
    return feats


def run_period(name: str, test_start: date, test_end: date, exclude_funda_ma: bool) -> dict:
    feats = _default_feats(exclude_funda_ma)
    params = TrainParams(
        **BASE_PARAMS,
        test_start=test_start,
        test_end=test_end,
        features=feats,
    )
    t0 = time.time()
    result = train(params)
    elapsed = time.time() - t0

    daily = result.daily_returns or []
    capital = float(_INITIAL)
    final_acc = float(daily[-1].get("cum_return", capital)) if daily else capital
    total_cost = sum(d.get("cost_rmb", 0) for d in daily) if daily else 0.0

    return {
        "period": name,
        "exclude_funda_ma": exclude_funda_ma,
        "n_feats": result.n_features,
        "final_acc": round(final_acc, 2),
        "total_return_pct": round((final_acc / capital - 1) * 100, 2),
        "sharpe": round(result.sharpe_raw, 4),
        "sharpe_sampled": round(result.sharpe_sampled, 4),
        "max_dd_pct": round(result.max_dd * 100, 2),
        "win_rate_pct": round(result.win_rate * 100, 1),
        "ic_mean": round(result.ic_mean, 4),
        "annual_return_pct": round(result.annual_return * 100, 2),
        "n_days": result.n_days,
        "total_cost_rmb": round(total_cost, 0),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    init_db()
    all_results = []
    for period_name, ts, te in TEST_PERIODS:
        print(f"\n{'='*60}")
        print(f"  {period_name}  ({ts} ~ {te})")
        print(f"{'='*60}")
        for label, excl in [("含全部", False), ("去基本面MA", True)]:
            print(f"  [{label}]...", end=" ", flush=True)
            r = run_period(period_name, ts, te, excl)
            print(f"Sharpe={r['sharpe']:.3f}  终值={r['final_acc']/1e4:.0f}万  (n_feats={r['n_feats']})")
            all_results.append(r)

    # 汇总
    print(f"\n\n{'='*80}")
    print(f"  基本面 MA5/20 衍生消融")
    print(f"{'='*80}")
    import json as j
    headers = ["区间", "特征集", "特征数", "终值(万)", "总收益%", "Sharpe", "样本Sharpe",
               "最大回撤%", "胜率%", "日均IC", "年化%", "累计成本", "天数"]
    print("  ".join(f"{h:>10}" for h in headers))
    for r in all_results:
        label = "去MA5/20" if r["exclude_funda_ma"] else "含全部"
        print("  ".join(f"{str(r.get(k, '')):>10}" for k in
                       ["period", "label", "n_feats", "final_acc//1e4",
                        "total_return_pct", "sharpe", "sharpe_sampled",
                        "max_dd_pct", "win_rate_pct", "ic_mean",
                        "annual_return_pct", "total_cost_rmb", "n_days"]))

    print(f"\n消融影响 (有→去MA5/20):")
    for period_name, _, _ in TEST_PERIODS:
        a = next(r for r in all_results if r["period"] == period_name and not r["exclude_funda_ma"])
        b = next(r for r in all_results if r["period"] == period_name and r["exclude_funda_ma"])
        print(f"  {period_name:>8s}: Sharpe {a['sharpe']:.3f}→{b['sharpe']:.3f} ({b['sharpe']-a['sharpe']:+.4f})  "
              f"终值 {a['final_acc']/1e4:.0f}万→{b['final_acc']/1e4:.0f}万  "
              f"收益 {a['total_return_pct']:.1f}%→{b['total_return_pct']:.1f}%")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / "ablation_funda_ma_results.json"
    with open(out, "w", encoding="utf-8") as f:
        j.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果保存: {out}")


if __name__ == "__main__":
    main()
