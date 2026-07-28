"""
融资融券消融实验: 5个回测区间 × 有/无两融特征

用法: python scripts/ablation_margin.py
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
from backend.models import TrainResult
from backend.database import init_db

# ── 配置 ──────────────────────────────────────────────────────
_INITIAL = 2_000_000
TEST_PERIODS = [
    ("2025H1", date(2025, 1, 2), date(2025, 6, 30)),
    ("2025H2", date(2025, 7, 1), date(2025, 12, 31)),
    ("2026Q1", date(2026, 1, 5), date(2026, 3, 31)),
    ("2026Q2", date(2026, 4, 1), date(2026, 6, 30)),
    ("full",  date(2025, 9, 1), date(2026, 7, 10)),
]

# 两融特征 (mtss_相关, mtss = margin trade 融资融券)
MARGIN_FEATS = [c for c in pd.read_parquet(latest_training_data()).columns
                if c.startswith(("mtss_", "fin_balance")) and "_21d" not in c]
print(f"两融特征: {len(MARGIN_FEATS)} 个: {MARGIN_FEATS}")

BASE_PARAMS = dict(
    train_start=date(2023, 1, 1),
    buy_pct=0.03, sell_pct=0.03, slip_pct=0.01,
    top_n=3, sample_interval=5,
    n_estimators=400, max_depth=4, learning_rate=0.03,
    initial_capital=_INITIAL,
    universe_source="关注圈",
)


def run_period(name: str, test_start: date, test_end: date, exclude_margin: bool) -> dict:
    """运行一次回测, 返回汇总指标."""
    features = None  # 默认=全部特征
    if exclude_margin:
        all_feats = pd.read_parquet(latest_training_data()).columns.tolist()
        _SKIP_CORE = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
                      "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret"}
        default_feats = [c for c in all_feats
                         if c not in _SKIP_CORE and "_21d" not in c
                         and not c.endswith("_cross")
                         and c not in ("ret_1d", "ret_2d", "ret_5d", "ret_21d")]
        features = [c for c in default_feats if c not in MARGIN_FEATS]

    params = TrainParams(
        **BASE_PARAMS,
        test_start=test_start,
        test_end=test_end,
        features=features,
    )

    t0 = time.time()
    result = train(params)
    elapsed = time.time() - t0

    # 净收益
    capital = float(_INITIAL)
    daily = result.daily_returns
    if daily:
        final_acc = float(daily[-1].get("cum_return", capital))
        acc_return = final_acc / capital - 1
    else:
        final_acc = capital
        acc_return = 0.0

    total_cost = sum(d.get("cost_rmb", 0) for d in daily) if daily else 0.0

    return {
        "period": name,
        "exclude_margin": exclude_margin,
        "n_feats": result.n_features,
        "final_acc": round(final_acc, 2),
        "total_return_pct": round(acc_return * 100, 2),
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

        # 带两融特征
        print("  [A] 含全部特征...", end=" ", flush=True)
        r1 = run_period(period_name, ts, te, exclude_margin=False)
        print(f"Sharpe={r1['sharpe']:.3f}  终值={r1['final_acc']/1e4:.0f}万")
        all_results.append(r1)

        # 无两融特征
        print("  [B] 剔除两融...", end=" ", flush=True)
        r2 = run_period(period_name, ts, te, exclude_margin=True)
        print(f"Sharpe={r2['sharpe']:.3f}  终值={r2['final_acc']/1e4:.0f}万")
        all_results.append(r2)

    # ── 汇总 ──
    print(f"\n\n{'='*80}")
    print(f"{'报告':^78}")
    print(f"{'='*80}")
    rows = []
    for r in all_results:
        rows.append({
            "区间": r["period"],
            "两融特征": "❌ 无" if r["exclude_margin"] else "✅ 有",
            "入模特征": r["n_feats"],
            "终值(万)": f"{r['final_acc']/1e4:.1f}",
            "总收益%": r["total_return_pct"],
            "Sharpe": r["sharpe"],
            "样本Sharpe": r["sharpe_sampled"],
            "最大回撤%": r["max_dd_pct"],
            "胜率%": r["win_rate_pct"],
            "日均IC": f"{r['ic_mean']:.4f}",
            "年化%": r["annual_return_pct"],
            "累计成本(元)": f"{r['total_cost_rmb']:,.0f}",
            "天数": r["n_days"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 方向性α (纵向对比: 有→无 的差异)
    print(f"\n\n{'='*80}")
    print("  消融影响 (有→无两融):")
    for period_name, _, _ in TEST_PERIODS:
        a = next(r for r in all_results if r["period"] == period_name and not r["exclude_margin"])
        b = next(r for r in all_results if r["period"] == period_name and r["exclude_margin"])
        delta_sharpe = b["sharpe"] - a["sharpe"]
        delta_ret = b["total_return_pct"] - a["total_return_pct"]
        delta_dd = b["max_dd_pct"] - a["max_dd_pct"]
        print(f"  {period_name:>8s}: Sharpe {a['sharpe']:.3f}→{b['sharpe']:.3f} ({delta_sharpe:+.4f})  "
              f"收益 {a['total_return_pct']:.1f}%→{b['total_return_pct']:.1f}% ({delta_ret:+.1f}%)  "
              f"回撤 {a['max_dd_pct']:.1f}%→{b['max_dd_pct']:.1f}% ({delta_dd:+.1f}%)")

    # 保存
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "ablation_margin_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
