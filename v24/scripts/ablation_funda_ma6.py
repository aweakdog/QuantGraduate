"""
基本面 MA5/20 衍生消融 (6区间)
注意: 由于 trainer.py 已将基本面MA加入 _EXCLUDED, 脚本运行时会临时移除该拦截
"""
import sys, time, json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Web"))

import pandas as pd
import backend.trainer as bt
from backend.trainer import train, TrainParams
from backend.paths import latest_training_data
from backend.database import init_db

_INITIAL = 2_000_000

TEST_PERIODS = [
    ("2025Q1", date(2025, 1, 2), date(2025, 3, 31)),
    ("2025Q2", date(2025, 4, 1), date(2025, 6, 30)),
    ("2025Q3", date(2025, 7, 1), date(2025, 9, 30)),
    ("2025Q4", date(2025, 10, 1), date(2025, 12, 31)),
    ("2026Q1", date(2026, 1, 5), date(2026, 3, 31)),
    ("2026Q2", date(2026, 4, 1), date(2026, 6, 30)),
]

BASE_PARAMS = dict(
    train_start=date(2023, 1, 1),
    buy_pct=0.03, sell_pct=0.03, slip_pct=0.01,
    top_n=3, sample_interval=5,
    n_estimators=400, max_depth=4, learning_rate=0.03,
    initial_capital=_INITIAL, universe_source="关注圈",
    skip_next_rec=True,
)

_FUNDA_MA = {"pe_ma5","pe_ma20","pb_ma5","pb_ma20",
             "revenue_ma5","revenue_ma20","profit_ma5","profit_ma20",
             "eps_ma5","eps_ma20","bps_ma5","bps_ma20",
             "debt_ratio_ma5","debt_ratio_ma20",
             "gross_margin_ma5","gross_margin_ma20",
             "roe_ma5","roe_ma20","total_assets_ma5","total_assets_ma20"}

print(f"基本面 MA5/20: {len(_FUNDA_MA)} 个")

# 临时从 trainer._EXCLUDED 中移除基本面MA, 让 ablation 能传回去
_removed = {f for f in _FUNDA_MA if f in bt._EXCLUDED}
bt._EXCLUDED -= _removed
print(f"已从 trainer._EXCLUDED 临时移除 {len(_removed)} 个 (恢复: {len(_FUNDA_MA - _removed)} 本来就不在)")

def get_feats(exclude_funda: bool) -> list[str]:
    cols = pd.read_parquet(latest_training_data()).columns.tolist()
    _SKIP = {"date","code","fwd_1d_ret","fwd_2d_ret","fwd_5d_ret",
             "fwd_21d_ret","fwd_1d_excess","fwd_1d_open_ret"}
    _LEAK = {"ret_1d","ret_2d","ret_5d","ret_21d"}
    feats = [c for c in cols if c not in _SKIP and "_21d" not in c
             and not c.endswith("_cross") and c not in _LEAK
             and c not in bt._EXCLUDED]  # 不重复排除 funda_ma
    if exclude_funda:
        feats = [c for c in feats if c not in _FUNDA_MA]
    return feats

def run_one(name, ts, te, excl):
    params = TrainParams(**BASE_PARAMS, test_start=ts, test_end=te, features=get_feats(excl))
    t0 = time.time()
    r = train(params)
    elp = time.time() - t0
    daily = r.daily_returns or []
    cap = float(_INITIAL)
    fa = float(daily[-1].get("cum_return", cap)) if daily else cap
    return dict(period=name, exclude=excl, n_feats=r.n_features,
                final=round(fa,2), ret=round((fa/cap-1)*100,2),
                sharpe=round(r.sharpe_raw,4), sharpe_s=round(r.sharpe_sampled,4),
                maxdd=round(r.max_dd*100,2), win=round(r.win_rate*100,1),
                ic=round(r.ic_mean,4), ann=round(r.annual_return*100,2),
                cost=sum(d.get("cost_rmb",0) for d in daily),
                days=r.n_days, elp=round(elp,1))

def main():
    init_db()
    all_r = []
    for pn, ts, te in TEST_PERIODS:
        print(f"\n{'='*55}\n  {pn}  ({ts}~{te})\n{'='*55}")
        for lbl, excl in [("含全部", False), ("去基本面MA", True)]:
            print(f"  [{lbl}]...", end=" ", flush=True)
            r = run_one(pn, ts, te, excl)
            print(f"Sharpe={r['sharpe']:.3f}  终值={r['final']/1e4:.0f}万  (n={r['n_feats']})")
            all_r.append(r)

    print(f"\n\n{'='*90}")
    print(f"  基本面 MA5/20 消融 (20个)")
    print(f"{'='*90}")
    hdrs = ["区间","基本面MA","特征","终值万","收益%","Sharpe","SSharpe","回撤%","胜率%","IC","年化%","成本","天"]
    print("  ".join(f"{h:>9}" for h in hdrs))
    for r in all_r:
        lbl = "无" if r["exclude"] else "有"
        print(f"{r['period']:>7s} {lbl:>8s} {r['n_feats']:4d} {r['final']/1e4:8.1f} "
              f"{r['ret']:8.2f} {r['sharpe']:8.4f} {r['sharpe_s']:8.4f} "
              f"{r['maxdd']:8.2f} {r['win']:6.1f} {r['ic']:8.4f} {r['ann']:10.2f} {r['cost']:8.0f} {r['days']:4d}")

    print(f"\n消融影响 (有→去基本面MA):")
    for pn, _, _ in TEST_PERIODS:
        a = next(r for r in all_r if r["period"]==pn and not r["exclude"])
        b = next(r for r in all_r if r["period"]==pn and r["exclude"])
        ds = b['sharpe']-a['sharpe']
        dr = b['ret']-a['ret']
        print(f"  {pn:>7s}: Sharpe {a['sharpe']:.3f}→{b['sharpe']:.3f} ({ds:+.4f})  "
              f"收益 {a['ret']:.1f}%→{b['ret']:.1f}% ({dr:+.1f}%)  "
              f"终值 {a['final']/1e4:.0f}→{b['final']/1e4:.0f}")

    out = Path(__file__).resolve().parent.parent / "data" / "processed" / "ablation_funda_ma6_results.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {out}")

if __name__ == "__main__":
    main()
