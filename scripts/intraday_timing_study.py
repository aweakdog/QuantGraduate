# -*- coding: utf-8 -*-
"""日内择时·阶段 1: 余日漂移可预测性研究 (跑在 eez040, 读 data/processed/min1 分钟面板)

问题: 卖出日已定, 盘中哪个时刻卖。等价于在每个决策点 T 预测 "从 T 到收盘的漂移"
    y = close_day / p_T - 1
预测相关系数 ρ 与奖金的换算 (日线测算 2026-09-08): 期望捞回 ≈ ρ × |开盘-收盘| 均值(172bp),
所以 ρ=0.08 ≈ +14bp/边, 这是阶段 1 的门。

设计
  * 决策点 T: 15 个 (09:45 起每 15 分钟, 上午末 11:29, 下午 13:15 起每 15 分钟到 14:45)
  * 信息集: 严格 T 之前的 bar (bar 用起始分钟标记, minute < T)
  * 特征分三组, 分开记账因为实盘可得性不同:
      L1 (QMT 都有):  开盘以来/近 15 bar 收益, 跳空, 距日内高低点, 相对量(同时刻 20 日均),
                      笔数, 价差, 挂单不平衡, 距涨跌停, 昨日/5 日收益, 全池同时刻均值(市场日内)
      L2 (逐笔才有): 主动买占比(累计 / 近 15 bar)
      T:             决策点序号
  * 模型: LightGBM 回归, 一个模型吃全部决策点 (T 作为特征); expanding 训练, 每季度重训,
          1 日禁区; 首个测试季 2023-10 (与 PL 面板窗口对齐)
  * 两臂: full (L1+L2) 与 l1only, 差值 = 逐笔信息值多少 ρ
  * 评估: 样本外 Pearson/Spearman 逐年 × 逐决策点; 单特征基线 ρ (纯均值回归规则能拿多少);
          PL 四线 20 种子的真实卖出/买入事件子集上的 ρ 与"首次触发即成交"策略的 bp
          (卖: 扣该分钟半价差; 不触发 = 收盘成交 = 0bp), 与决策点集合内的 oracle 上限

产出 data/processed/intraday_timing/: rows_<arm>.parquet (样本外行含 pred), summary.json, 打印报告

用法
────
    python scripts/intraday_timing_study.py                  # 全面板
    python scripts/intraday_timing_study.py --end 20240630   # 试跑
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
MIN1 = ROOT / Path(os.environ.get("MIN1_OUT", "data/processed/min1"))
OUT = ROOT / "data/processed/intraday_timing"
PL_GLOB = "data/processed/wf_daily_PL_{line}_s*_te2026-09-04_*.json"
PL_LINES = ("aggr5w", "aggr10w", "steady5w", "fyf100w")

GRID = ([925] + [h * 100 + m for h in (9, 10, 11) for m in range(60) if (h, m) >= (9, 30) and (h, m) < (11, 30)]
        + [h * 100 + m for h in (13, 14) for m in range(60) if (h, m) < (14, 57)] + [1457])
POS = {m: i for i, m in enumerate(GRID)}
CKPTS = (945, 1000, 1015, 1030, 1045, 1100, 1115, 1129, 1315, 1330, 1345, 1400, 1415, 1430, 1445)
LAG = 15
HIST = 20
MIN_BARS = 100

L1_FEATS = ("ret_open", "ret_lag", "ret_pc", "gap", "dist_hi", "dist_lo", "rel_vol", "rel_vol_lag", "rel_ntrd",
            "spread_mean", "spread_lag", "qimb_lag", "up_room", "dn_room", "r1", "r5", "mkt_open", "mkt_lag")
L2_FEATS = ("buy_share", "buy_share_lag")
T_FEATS = ("ckpt",)
ARMS = {"full": L1_FEATS + L2_FEATS + T_FEATS, "l1only": L1_FEATS + T_FEATS}

LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=int(os.environ.get("LGB_THREADS", "16")))
LGB_ROUNDS = 400


def pivot(df, col):
    m = df.pivot(index="code", columns="minute", values=col).reindex(columns=GRID)
    return m.to_numpy(dtype="float64")


def ffill2d(a):
    """沿分钟轴前向填充 (停牌/无成交分钟沿用上一价)"""
    out = a.copy()
    for j in range(1, out.shape[1]):
        m = np.isnan(out[:, j])
        out[m, j] = out[m, j - 1]
    return out


def nancum(a):
    return np.nancumsum(np.nan_to_num(a, nan=0.0), axis=1)


def day_rows(df, state):
    """一天的面板 -> 每码每决策点一行 (特征 + 标签); state 维护 20 日同时刻量的滚动均值等"""
    codes = np.array(sorted(df["code"].unique()))
    close = ffill2d(pivot(df, "close"))
    high, low = pivot(df, "high"), pivot(df, "low")
    vol, amt, buy, ntrd = pivot(df, "vol"), pivot(df, "amt"), pivot(df, "buy_amt"), pivot(df, "n_trd")
    spread, qimb = pivot(df, "spread_bp"), pivot(df, "qimb")
    nbars = (~np.isnan(pivot(df, "close"))).sum(axis=1)
    pc = df.groupby("code")["prev_close"].first().reindex(codes).to_numpy(dtype="float64")
    open_ = pd.DataFrame(pivot(df, "open"), index=codes).bfill(axis=1).iloc[:, 0].to_numpy()
    day_close = close[:, -1]
    cvol, camt, cbuy, cntrd = nancum(vol), nancum(amt), nancum(buy), nancum(ntrd)
    run_hi = np.fmax.accumulate(np.nan_to_num(high, nan=-np.inf), axis=1)
    run_lo = np.fmin.accumulate(np.nan_to_num(low, nan=np.inf), axis=1)
    lim = np.where(np.char.startswith(codes.astype(str), "688") | np.char.startswith(codes.astype(str), "300"), 0.20, 0.10)

    # 日度上下文: 昨日/5 日收益来自前收盘序列, 20 日日均量
    pcs = pd.Series(pc, index=codes)
    hist_pc = state.setdefault("pc", deque(maxlen=6))
    r1 = (pcs / hist_pc[-1].reindex(codes) - 1).to_numpy() if len(hist_pc) >= 1 else np.full(len(codes), np.nan)
    r5 = (pcs / hist_pc[0].reindex(codes) - 1).to_numpy() if len(hist_pc) >= 6 else np.full(len(codes), np.nan)

    cs = df["date"].iloc[0]
    out = []
    for T in CKPTS:
        k = POS[T]
        p = close[:, k - 1]
        p_lag = close[:, k - 1 - LAG] if k - 1 - LAG >= 0 else open_
        cv, ca, cb, cn = cvol[:, k - 1], camt[:, k - 1], cbuy[:, k - 1], cntrd[:, k - 1]
        cv_l, ca_l, cb_l = (cvol[:, k - 1 - LAG], camt[:, k - 1 - LAG], cbuy[:, k - 1 - LAG]) if k - 1 - LAG >= 0 else (0, 0, 0)
        hist_v = state.setdefault(("v", T), deque(maxlen=HIST))
        hist_vl = state.setdefault(("vl", T), deque(maxlen=HIST))
        hist_n = state.setdefault(("n", T), deque(maxlen=HIST))

        def hist_mean(h):
            return pd.concat(h, axis=1).mean(axis=1).reindex(codes).to_numpy() if h else np.full(len(codes), np.nan)
        avg_v, avg_vl, avg_n = hist_mean(hist_v), hist_mean(hist_vl), hist_mean(hist_n)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret_open = p / open_ - 1
            ret_lag = p / p_lag - 1
            rest_amt, rest_vol = camt[:, -1] - ca, cvol[:, -1] - cv
            feats = dict(
                ret_open=ret_open, ret_lag=ret_lag, ret_pc=p / pc - 1, gap=open_ / pc - 1,
                dist_hi=p / run_hi[:, k - 1] - 1, dist_lo=p / run_lo[:, k - 1] - 1,
                rel_vol=cv / avg_v, rel_vol_lag=(cv - cv_l) / avg_vl, rel_ntrd=cn / avg_n,
                spread_mean=np.nanmean(spread[:, :k], axis=1), spread_lag=np.nanmean(spread[:, max(k - LAG, 0):k], axis=1),
                qimb_lag=np.nanmean(qimb[:, max(k - LAG, 0):k], axis=1),
                up_room=pc * (1 + lim) / p - 1, dn_room=pc * (1 - lim) / p - 1, r1=r1, r5=r5,
                mkt_open=np.full(len(codes), np.nanmean(ret_open)), mkt_lag=np.full(len(codes), np.nanmean(ret_lag)),
                buy_share=cb / ca, buy_share_lag=(cb - cb_l) / (ca - ca_l),
                ckpt=np.full(len(codes), float(k)),
                y_close=(day_close / p - 1) * 1e4, y_vwap=(rest_amt / rest_vol / p - 1) * 1e4,
                half_spread=np.nan_to_num(spread[:, k - 1], nan=np.nanmedian(spread[:, k - 1])) / 2,
                p=p, day_close=day_close, ok=(nbars >= MIN_BARS) & np.isfinite(p) & np.isfinite(day_close) & (p > 0),
            )
        o = pd.DataFrame(feats)
        o.insert(0, "T", T)
        o.insert(0, "code", codes)
        o.insert(0, "date", cs)
        out.append(o[o["ok"]].drop(columns="ok"))
        hist_v.append(pd.Series(cv, index=codes))
        hist_vl.append(pd.Series(cv - cv_l, index=codes))
        hist_n.append(pd.Series(cn, index=codes))
    hist_pc.append(pcs)
    return pd.concat(out, ignore_index=True)


def build_rows(days):
    state, parts, t0 = {}, [], time.time()
    for i, f in enumerate(days, 1):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        parts.append(day_rows(df, state))
        if i % 100 == 0:
            print(f"  特征 [{i}/{len(days)}] {time.time() - t0:.0f}s", flush=True)
    rows = pd.concat(parts, ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    for c in rows.columns:
        if rows[c].dtype == "float64":
            rows[c] = rows[c].astype("float32")
    return rows


def walk_forward(rows, feats, label="y_close", refit_months=3, first_test="2023-10-01", embargo_days=1):
    import lightgbm as lgb
    y = rows[label].clip(-1000, 1000)
    d = rows["date"]
    starts = pd.date_range(first_test, d.max(), freq=f"{refit_months}MS")
    pred = pd.Series(np.nan, index=rows.index, dtype="float32")
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else d.max() + pd.Timedelta(days=1)
        tr = d < s - pd.Timedelta(days=embargo_days)
        te = (d >= s) & (d < e)
        if te.sum() == 0:
            continue
        ds = lgb.Dataset(rows.loc[tr, list(feats)], y[tr], free_raw_data=True)
        m = lgb.train(LGB_PARAMS, ds, num_boost_round=LGB_ROUNDS)
        pred[te] = m.predict(rows.loc[te, list(feats)]).astype("float32")
        print(f"    fold {s.date()}~{e.date()}: train {tr.sum():,} test {te.sum():,}", flush=True)
    return pred


def corr_table(rows, pred, label="y_close"):
    ok = pred.notna() & rows[label].notna()
    r = rows.loc[ok, ["date", "T", label]].assign(pred=pred[ok], y=rows.loc[ok, label])
    r["year"] = r["date"].dt.year

    def both(g):
        return pd.Series({"n": len(g), "pearson": g["pred"].corr(g["y"]), "spearman": g["pred"].corr(g["y"], method="spearman")})
    return {"all": both(r).to_dict(), "by_year": r.groupby("year")[["pred", "y"]].apply(both).round(4).to_dict("index"),
            "by_T": r.groupby("T")[["pred", "y"]].apply(both).round(4).to_dict("index")}


def baseline_corrs(rows, pred_mask, label="y_close"):
    r = rows[pred_mask]
    return {f: round(float(r[f].corr(r[label])), 4) for f in ("ret_open", "ret_lag", "gap", "mkt_open", "mkt_lag", "buy_share", "qimb_lag", "rel_vol")}


def load_events():
    ev = []
    for line in PL_LINES:
        for f in glob.glob(str(ROOT / PL_GLOB.format(line=line))):
            for t in json.load(open(f))["trades"]:
                ev.append((t["date"], t["code"][:6], t["action"]))
    ev = pd.DataFrame(ev, columns=["date", "code", "action"])
    ev["date"] = pd.to_datetime(ev["date"])
    return ev


def policy_eval(sub, thr_list=(0, 10, 20, 30), side="sell"):
    """sub: 某事件集合 (date, code) 的全部决策点行 (含 pred, y_close, half_spread)。
    卖: 首个 pred < -thr 的 T 成交, 收益 = p_T/close - 1 - 半价差 (bp), 不触发 = 0 (收盘成交)。买: 镜像。"""
    sub = sub.sort_values(["date", "code", "T"])
    g = sub.groupby(["date", "code"], sort=False)
    res = {}
    for thr in thr_list:
        if side == "sell":
            hit = sub["pred"] < -thr
            gain = -sub["y_close"] / (1 + sub["y_close"] / 1e4) - sub["half_spread"]   # p/close-1 = -(y)/(1+y)
        else:
            hit = sub["pred"] > thr
            gain = sub["y_close"] - sub["half_spread"]                                   # close/p-1 = y, 买得比收盘便宜
        first = sub[hit].groupby(["date", "code"], sort=False).head(1)
        per_event = pd.Series(0.0, index=g.size().index)
        per_event.loc[list(zip(first["date"], first["code"]))] = gain[first.index].to_numpy()
        res[f"thr{thr}"] = {"mean_bp": round(float(per_event.mean()), 2), "median_bp": round(float(per_event.median()), 2),
                            "trigger_rate": round(float(len(first) / len(per_event)), 3), "n_events": int(len(per_event))}
    if side == "sell":
        oracle = (-sub["y_close"] / (1 + sub["y_close"] / 1e4) - sub["half_spread"]).groupby([sub["date"], sub["code"]]).max().clip(lower=0)
    else:
        oracle = (sub["y_close"] - sub["half_spread"]).groupby([sub["date"], sub["code"]]).max().clip(lower=0)
    res["oracle_bp"] = round(float(oracle.mean()), 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20220901")
    ap.add_argument("--end", default="20991231")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--first-test", default="2023-10-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = sorted(p for p in MIN1.glob("*.parquet") if a.start <= p.stem <= a.end)
    if not days:
        sys.exit(f"{MIN1} 下没有 {a.start}~{a.end}")
    print(f"分钟面板 {len(days)} 天 {days[0].stem}~{days[-1].stem}; 决策点 {len(CKPTS)} 个", flush=True)
    rows = build_rows(days)
    print(f"样本行 {len(rows):,}  码日 {rows.groupby(['date', 'code']).ngroups:,}  "
          f"y_close 均值 {rows.y_close.mean():+.1f}bp 标准差 {rows.y_close.std():.0f}bp  |y| 均值 {rows.y_close.abs().mean():.0f}bp", flush=True)

    ev = load_events()
    key = pd.MultiIndex.from_frame(rows[["date", "code"]])
    sell_key = pd.MultiIndex.from_frame(ev.loc[ev.action == "sell", ["date", "code"]].drop_duplicates())
    buy_key = pd.MultiIndex.from_frame(ev.loc[ev.action == "buy", ["date", "code"]].drop_duplicates())
    is_sell, is_buy = key.isin(sell_key), key.isin(buy_key)
    print(f"PL 事件: 卖 {len(sell_key):,} 码日 (面板覆盖 {rows[is_sell].groupby(['date','code']).ngroups:,}), "
          f"买 {len(buy_key):,} 码日 (覆盖 {rows[is_buy].groupby(['date','code']).ngroups:,})", flush=True)

    summary = {"n_rows": int(len(rows)), "ckpts": CKPTS, "arms": {}}
    for arm, feats in ARMS.items():
        t0 = time.time()
        print(f"\n== 臂 {arm}: {len(feats)} 特征, 每 {a.refit_months} 月重训 ==", flush=True)
        pred = walk_forward(rows, feats, refit_months=a.refit_months, first_test=a.first_test)
        ok = pred.notna()
        ct = corr_table(rows, pred)
        res = {"corr": ct, "baseline_single_feature_corr": baseline_corrs(rows, ok),
               "events": {}}
        for name, mask, side in (("sell", is_sell, "sell"), ("buy", is_buy, "buy")):
            sub = rows[ok & mask].assign(pred=pred[ok & mask])
            if len(sub):
                res["events"][name] = {"corr": {"pearson": round(float(sub["pred"].corr(sub["y_close"])), 4),
                                                "spearman": round(float(sub["pred"].corr(sub["y_close"], method="spearman")), 4),
                                                "n_rows": int(len(sub))},
                                       "policy": policy_eval(sub, side=side)}
        # 全池上的策略 (不限 PL 事件), 看结论是否只在我们选的票上成立
        allsub = rows[ok].assign(pred=pred[ok])
        res["events"]["all_universe_sell"] = policy_eval(allsub, side="sell")
        summary["arms"][arm] = res
        rows.loc[ok].assign(pred=pred[ok], is_sell=is_sell[ok.to_numpy()], is_buy=is_buy[ok.to_numpy()]).to_parquet(
            OUT / f"rows_{arm}.parquet", index=False)
        c = ct["all"]
        print(f"  样本外 n={c['n']:,} Pearson {c['pearson']:.4f} Spearman {c['spearman']:.4f}  ({time.time() - t0:.0f}s)")
        print("  逐年:", {y: (v["pearson"], v["spearman"]) for y, v in ct["by_year"].items()})
        print("  逐决策点 Pearson:", {t: v["pearson"] for t, v in ct["by_T"].items()})
        print("  单特征基线 ρ:", res["baseline_single_feature_corr"])
        for name in ("sell", "buy"):
            if name in res["events"]:
                e = res["events"][name]
                print(f"  PL {name} 事件: ρ={e['corr']['pearson']:.4f} (n={e['corr']['n_rows']:,})  策略: "
                      + "  ".join(f"{k} {v['mean_bp']:+.1f}bp(触发{v['trigger_rate']:.0%})" for k, v in e["policy"].items() if k != "oracle_bp")
                      + f"  oracle {e['policy']['oracle_bp']:+.1f}bp", flush=True)
        print(f"  全池 sell 策略: " + "  ".join(f"{k} {v['mean_bp']:+.1f}bp" for k, v in res["events"]["all_universe_sell"].items() if k != "oracle_bp"))
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"\n完成 -> {OUT}", flush=True)


if __name__ == "__main__":
    try:
        from proctitle import lowkey
        lowkey("mltask/study")
    except Exception:
        pass
    main()
