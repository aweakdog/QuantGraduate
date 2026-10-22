"""网格交易研究 G1/G2/G3 (2026-10-22, 预注册见 docs/experiment_board.md)。在研究根目录跑:

  python scripts/grid_study.py --root /home/yliog/quant-research-20260912-n2 --procs 8

G1 单独用: 7 只 ETF, 主配置 + 参数扫描, 2016-2020 样本内 / 2021-2026.10 样本外, 对照买入持有、
   50% 恒定比例(20 日再平衡)、货币基金。
G2 空仓期闲钱: 只在现行策略空仓的日子跑网格 (2017-2026.10), 对照同日子的货币基金/50% 持有/满仓持有,
   并逐段统计每次空仓的盈亏。
G3 结合现行策略: qmt10w 配置 (N7_Q5, 40 种子, 2023-09-21~2026-09-11) 的逐日收益 +
   (a) 空仓日叠加网格 (b) 空仓日只买货币基金 (c) 静态 80/20、70/30 与常开网格并行。
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from grid_backtest import GridParams, constant_mix, run_grid

ETFS = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000",
        "159915": "创业板", "510880": "红利", "518880": "黄金"}
T0 = {"518880"}
PRIMARY = GridParams(step=0.02, levels=10, base=0.5, center="fixed", reset=True, path="ohlc",
                     fee_rate=1e-4, min_fee=5.0, slippage=5e-4, cash_rate=0.015)
PERIODS = {"IS": ("2016-01-01", "2020-12-31"), "OOS": ("2021-01-01", "2026-10-21"), "FULL": ("2016-01-01", "2026-10-21")}
G2_HALVES = {"H1": ("2017-01-01", "2021-12-31"), "H2": ("2022-01-01", "2026-10-21")}
G3_ETF = "510300"
ROOT = None


def load_etf(code):
    return pd.read_parquet(ROOT / "data" / "grid" / f"etf_{code}.parquet")


def params_for(code, **kw):
    return replace(PRIMARY, t0=code in T0, **kw)


# ── G1 ───────────────────────────────────────────────────────────
def _g1_job(args):
    code, label, kw, period = args
    df = load_etf(code)
    s, e = PERIODS[period]
    if label == "buyhold":
        _, r = constant_mix(df, 1.0, start=s, end=e, p=params_for(code))
    elif label == "mix50":
        _, r = constant_mix(df, 0.5, rebalance_days=20, start=s, end=e, p=params_for(code))
    elif label == "cash":
        _, r = constant_mix(df, 0.0, start=s, end=e, p=params_for(code))
    else:
        _, r = run_grid(df, params_for(code, **kw), start=s, end=e)
    return {"code": code, "label": label, "kw": kw, "period": period, **r}


def g1_jobs():
    jobs = []
    for code in ETFS:
        for period in PERIODS:
            for label in ("buyhold", "mix50", "cash"):
                jobs.append((code, label, {}, period))
            jobs.append((code, "primary", {}, period))
            jobs.append((code, "primary_closepath", {"path": "close"}, period))
            jobs.append((code, "primary_minfee0.1", {"min_fee": 0.1}, period))
            for step in (0.01, 0.015, 0.02, 0.03, 0.05):
                for levels in (5, 10, 20):
                    for cname, ckw in (("fixed_reset", {}), ("fixed_noreset", {"reset": False}),
                                       ("ma60", {"center": "ma"})):
                        jobs.append((code, f"scan_{cname}", {"step": step, "levels": levels, **ckw}, period))
    return jobs


# ── G2 ───────────────────────────────────────────────────────────
def _episodes(mask):
    eps, start = [], None
    for d, v in mask.items():
        if v and start is None:
            start = d
        elif not v and start is not None:
            eps.append((start, d))
            start = None
    if start is not None:
        eps.append((start, mask.index[-1]))
    return eps


def _g2_job(args):
    code, label, kw, half = args
    df = load_etf(code)
    reg = pd.read_parquet(ROOT / "data" / "grid" / "regime.parquet")
    act = reg["cash_day"].reindex(df.index).fillna(False).astype(bool)
    s, e = G2_HALVES[half] if half != "ALL" else ("2017-01-01", "2026-10-21")
    p = params_for(code, **kw)
    if label == "mmf":
        out, r = constant_mix(df, 0.0, start=s, end=e, p=p)
    elif label == "hold100":
        out, r = constant_mix(df, 1.0, active=act, start=s, end=e, p=p)
    elif label == "mix50":
        out, r = constant_mix(df, 0.5, rebalance_days=20, active=act, start=s, end=e, p=p)
    else:
        out, r = run_grid(df, p, active=act, start=s, end=e)
    res = {"code": code, "label": label, "kw": kw, "half": half, **r}
    if half == "ALL":
        a = act.reindex(out.index).fillna(False)
        eq = out["equity"]
        rets = []
        for st, en in _episodes(a):
            i0 = eq.index.get_loc(st)
            base = eq.iloc[i0 - 1] if i0 > 0 else eq.iloc[i0]
            rets.append(float(eq.loc[en] / base - 1))
        res["episodes"] = {"n": len(rets), "median_pct": round(float(np.median(rets)) * 100, 2) if rets else None,
                           "win_rate": round(float(np.mean([x > 0 for x in rets])), 3) if rets else None,
                           "worst_pct": round(float(min(rets)) * 100, 2) if rets else None,
                           "best_pct": round(float(max(rets)) * 100, 2) if rets else None}
        res["cash_day_share"] = round(float(a.mean()), 3)
    return res


def g2_jobs():
    jobs = []
    for code in ETFS:
        for half in ("ALL", "H1", "H2"):
            for label in ("mmf", "hold100", "mix50", "primary"):
                jobs.append((code, label, {}, half))
            for step in (0.01, 0.015, 0.03, 0.05):
                jobs.append((code, "scan_step", {"step": step}, half))
    return jobs


# ── G3 ───────────────────────────────────────────────────────────
def _metrics(ret):
    eq = (1 + ret).cumprod()
    tail = eq.iloc[-126:]
    return {"total_pct": float((eq.iloc[-1] - 1) * 100), "dd_pct": float((eq / eq.cummax() - 1).min() * 100),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(244)) if ret.std() > 0 else 0.0,
            "recent126_pct": float((tail.iloc[-1] / tail.iloc[0] - 1) * 100)}


def g3(etf_codes):
    files = sorted(glob.glob(str(ROOT / "data" / "processed" / "wf_daily_N7_Q5_s*_ts2023-09-20_te2026-09-11_cap100000.json")))
    seeds = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        day = pd.DataFrame(d["daily"])
        day["date"] = pd.to_datetime(day["date"])
        seeds[Path(f).name.split("_s")[1].split("_")[0]] = day.set_index("date")
    if not seeds:
        return {"error": "没有 N7_Q5 产物"}
    any_day = next(iter(seeds.values()))
    in_cash = any_day["in_cash"].astype(bool)
    out = {"n_seeds": len(seeds), "cash_day_share": round(float(in_cash.mean()), 3),
           "cash_flag_identical_across_seeds": all((s["in_cash"].astype(bool) == in_cash).all() for s in seeds.values())}
    sleeves = {}
    idx = any_day.index
    mmf = pd.Series((1.015) ** (1 / 244) - 1, index=idx)
    for code in etf_codes:
        df = load_etf(code)
        p = params_for(code)
        g_out, _ = run_grid(df, p, capital=1.0, active=in_cash.reindex(df.index).fillna(False),
                            start=str(idx[0].date()), end=str(idx[-1].date()))
        always, _ = run_grid(df, p, capital=1.0, start=str(idx[0].date()), end=str(idx[-1].date()))
        sleeves[code] = {"overlay": g_out["equity"].pct_change().fillna(0.0).reindex(idx).fillna(0.0),
                         "always": always["equity"].pct_change().fillna(0.0).reindex(idx).fillna(0.0)}
    on_sleeve = in_cash | in_cash.shift(1, fill_value=False)       # 离场那天的持有收益也算
    variants = {}
    for sid, day in seeds.items():
        r_s = day["daily_ret"].astype(float)
        base = _metrics(r_s)
        rows = {"strategy": base, "strategy+mmf": _metrics(r_s + mmf.where(in_cash, 0.0))}
        for code, sl in sleeves.items():
            rows[f"overlay_{code}"] = _metrics(r_s + sl["overlay"].where(on_sleeve, 0.0))
            for w in (0.8, 0.7):
                rows[f"split{int(w * 100)}_{code}"] = _metrics(w * r_s + (1 - w) * sl["always"])
        variants[sid] = rows
    summary = {}
    keys = next(iter(variants.values())).keys()
    for k in keys:
        if k == "strategy":
            continue
        d_ret = [variants[s][k]["total_pct"] - variants[s]["strategy"]["total_pct"] for s in variants]
        d_dd = [variants[s][k]["dd_pct"] - variants[s]["strategy"]["dd_pct"] for s in variants]
        d_sh = [variants[s][k]["sharpe"] - variants[s]["strategy"]["sharpe"] for s in variants]
        d_rc = [variants[s][k]["recent126_pct"] - variants[s]["strategy"]["recent126_pct"] for s in variants]
        summary[k] = {"ret_delta_med": round(float(np.median(d_ret)), 2), "ret_delta_pos": int(sum(x > 0 for x in d_ret)),
                      "dd_delta_med": round(float(np.median(d_dd)), 2), "sharpe_delta_med": round(float(np.median(d_sh)), 3),
                      "recent126_delta_med": round(float(np.median(d_rc)), 2)}
    base_med = {m: round(float(np.median([variants[s]["strategy"][m] for s in variants])), 2)
                for m in ("total_pct", "dd_pct", "sharpe", "recent126_pct")}
    out.update(strategy_median=base_med, deltas=summary)
    return out


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    ROOT = Path(a.root)
    res = {"primary": PRIMARY.__dict__, "etfs": ETFS}
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(str(ROOT),)) as ex:
        res["g1"] = list(ex.map(_g1_job, g1_jobs(), chunksize=4))
        res["g2"] = list(ex.map(_g2_job, g2_jobs(), chunksize=2))
    res["g3"] = g3(list(ETFS))
    out = ROOT / "summary_grid.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"写出 {out}: G1 {len(res['g1'])} 次, G2 {len(res['g2'])} 次, G3 {res['g3'].get('n_seeds')} 种子")


def _init(root):
    global ROOT
    ROOT = Path(root)


if __name__ == "__main__":
    main()
