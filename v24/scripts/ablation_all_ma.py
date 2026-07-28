"""
全部 MA5/20 衍生消融: 226 个 MA 特征 保留/剔除

4区间 × 有/无 = 8 次回测
"""
import sys, time, json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Web"))

import pandas as pd
from backend.trainer import train, TrainParams
from backend.paths import latest_training_data
from backend.database import init_db

_INITIAL = 2_000_000

TEST_PERIODS = [
    ("2025H1", date(2025, 1, 2), date(2025, 6, 30)),
    ("2025H2", date(2025, 7, 1), date(2025, 12, 31)),
    ("2026Q1", date(2026, 1, 5), date(2026, 3, 31)),
    ("2026Q2", date(2026, 4, 1), date(2026, 6, 30)),
]

BASE_PARAMS = dict(
    train_start=date(2023, 1, 1),
    buy_pct=0.03, sell_pct=0.03, slip_pct=0.01,
    top_n=3, sample_interval=5,
    n_estimators=400, max_depth=4, learning_rate=0.03,
    initial_capital=_INITIAL,
    universe_source="关注圈",
    skip_next_rec=True,  # 消融实验加速
)

def get_feats(exclude_all_ma: bool) -> list[str]:
    cols = pd.read_parquet(latest_training_data()).columns.tolist()
    _SKIP = {"date", "code", "fwd_1d_ret", "fwd_2d_ret", "fwd_5d_ret",
             "fwd_21d_ret", "fwd_1d_excess", "fwd_1d_open_ret"}
    _LEAK = {"ret_1d", "ret_2d", "ret_5d", "ret_21d"}
    _EXCLUDED = {"mtss_1d", "mtss_z", "mtss_1d_ma5", "mtss_z_ma5",
                 "mtss_1d_ma20", "mtss_z_ma20",
                 "pe_ma5","pe_ma20","pb_ma5","pb_ma20",
                 "revenue_ma5","revenue_ma20","profit_ma5","profit_ma20",
                 "eps_ma5","eps_ma20","bps_ma5","bps_ma20",
                 "debt_ratio_ma5","debt_ratio_ma20",
                 "gross_margin_ma5","gross_margin_ma20",
                 "roe_ma5","roe_ma20","total_assets_ma5","total_assets_ma20"}

    feats = [c for c in cols if c not in _SKIP and "_21d" not in c
             and not c.endswith("_cross") and c not in _LEAK
             and c not in _EXCLUDED]
    if exclude_all_ma:
        ma_set = {c for c in feats if c.endswith(("_ma5", "_ma20"))}
        print(f"    剔除 {len(ma_set)} 个 MA5/20 特征 (剩余 {len(feats)-len(ma_set)})", flush=True)
        feats = [c for c in feats if c not in ma_set]
    return feats


def run_period(name: str, ts: date, te: date, exclude_all_ma: bool) -> dict:
    feats = get_feats(exclude_all_ma)
    params = TrainParams(**BASE_PARAMS, test_start=ts, test_end=te, features=feats)
    t0 = time.time()
    result = train(params)
    elapsed = time.time() - t0
    daily = result.daily_returns or []
    capital = float(_INITIAL)
    final_acc = float(daily[-1].get("cum_return", capital)) if daily else capital
    total_cost = sum(d.get("cost_rmb", 0) for d in daily) if daily else 0.0
    return dict(period=name, exclude_ma=exclude_all_ma, n_feats=result.n_features,
                final_acc=round(final_acc,2),
                ret_pct=round((final_acc/capital-1)*100,2),
                sharpe=round(result.sharpe_raw,4),
                sharpe_s=round(result.sharpe_sampled,4),
                maxdd=round(result.max_dd*100,2),
                win=round(result.win_rate*100,1),
                ic=round(result.ic_mean,4),
                ann=round(result.annual_return*100,2),
                cost=round(total_cost,0), days=result.n_days,
                elp=round(elapsed,1))


def main():
    init_db()
    all_r = []
    for pn, ts, te in TEST_PERIODS:
        print(f"\n{'='*60}\n  {pn}  ({ts} ~ {te})\n{'='*60}")
        for lbl, excl in [("含MA", False), ("去MA", True)]:
            print(f"  [{lbl}]...", end=" ", flush=True)
            r = run_period(pn, ts, te, excl)
            print(f"Sharpe={r['sharpe']:.3f}  终值={r['final_acc']/1e4:.0f}万  (n={r['n_feats']})")
            all_r.append(r)

    print(f"\n\n{'='*80}")
    print(f"  MA5/20 全部消融 (226个)")
    print(f"{'='*80}")
    print(f"{'区间':>8s} {'MA':>4s} {'特':>4s} {'终值万':>8s} {'收益%':>8s} {'Sharpe':>8s} "
          f"{'SharpeS':>8s} {'回撤%':>8s} {'胜率%':>6s} {'IC':>8s} {'年化%':>10s} {'成本':>8s} {'天':>5s}")
    for r in all_r:
        print(f"{r['period']:>8s} {'无' if r['exclude_ma'] else '有'} "
              f"{r['n_feats']:4d} {r['final_acc']/1e4:8.1f} {r['ret_pct']:8.2f} {r['sharpe']:8.4f} "
              f"{r['sharpe_s']:8.4f} {r['maxdd']:8.2f} {r['win']:6.1f} {r['ic']:8.4f} "
              f"{r['ann']:10.2f} {r['cost']:8.0f} {r['days']:5d}")

    print(f"\n消融影响 (有MA→全去MA):")
    for pn, _, _ in TEST_PERIODS:
        a = next(r for r in all_r if r["period"]==pn and not r["exclude_ma"])
        b = next(r for r in all_r if r["period"]==pn and r["exclude_ma"])
        print(f"  {pn:>8s}: Sharpe {a['sharpe']:.3f}→{b['sharpe']:.3f} ({b['sharpe']-a['sharpe']:+.4f})  "
              f"终值 {a['final_acc']/1e4:.0f}→{b['final_acc']/1e4:.0f}  "
              f"收益 {a['ret_pct']:.1f}%→{b['ret_pct']:.1f}%  "
              f"回撤 {a['maxdd']:.1f}%→{b['maxdd']:.1f}%")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / "ablation_all_ma_results.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {out}")

if __name__ == "__main__":
    main()
