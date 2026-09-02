"""因子武器库第二批(B2)研究增广器 —— "数据全在库"的便宜族一次建列

产出一份研究矩阵 (源矩阵 + 各族列), 每族配一张特征表(生产 80 列 + 本族列),
探索臂按 --features-from 各取自己那几列 => 训出来的仍是单族专才模型
(与 build_t1_augmented 同构; 合体模型 T1X 已判死, 不做)。

判决 (2026-09-03, 详见 docs/factor_family_ledger.md 与 experiment_board B2/T3_20 条)
────────────────────────────────────────────────────────────────────
  T2D ❌ 5 种子不过门 (中位 -13.7pp, 2/5; 生产列已含 mtss_1d_ma20/gross_margin_ma20)
  T3  ❌ 5 种子过门 (+6.0, 4/5) 但 20 面板不过 (同夜配对 +2.8pp, 13/20, p=0.13)
  T1C — 未跑: 同信号的 2 列版 08-30 已判死, 本变体不在默认族列表里
三族都不进生产。本文件保留的价值: (1) 判决可复现; (2) 载入器可复用 —— load_kline
(全市场 qfq 长表) / t2d_frame 里的 margin_detail 读法 / sw_l1_daily (申万一级 PIT 区间贴
标签), 下一批族(龙虎榜/解禁/股东户数)加一个 frame 函数 + FAMS 一行即可挂进来。

✅ 协议教训(本批首次暴露): 配对基线必须同夜同矩阵重训。同种子的同夜基线 vs 档案
XEBP 逐个差到 59pp(分布中位却一致) —— 种子路径对矩阵内容/列序极敏感。队列模板
/tmp/b2_041.sh + /tmp/t3_20_041.sh (041) 开头就是基线臂。

  T2D 两融衍生 4 列 (源 tushare margin_detail, **lag=1**)
    t2d_rz_chg5 / t2d_rz_chg20   融资余额 5/20 日变化率
    t2d_rz_net5                  5 日净融资买入(买入-偿还)累计 / 融资余额
    t2d_rz_buy_z                 融资买入额占成交额比的 20 日 z
    PIT: 交易所在 D+1 早晨才公布 D 的两融数据, 信号链 D 17:30 拿不到 => 滞后一天
    (与 T1A 逐笔同理, 见 build_t1_augmented.t1a_frame 长注)。
  T3 行业动量溢出 3 列 (源 qfq kline 全市场 + 申万一级 PIT 成分, lag=0)
    t3_ind_ret5 / t3_ind_ret20   同行业【剔除自身】等权 5/20 日收益 (lead-lag 溢出)
    t3_rel5                      自身 5 日收益 - 行业 5 日收益 (行业内相对强弱)
    行业归属按 in_date/out_date 区间逐日判定 (公告日口径, 与 --ind-cap 同表同构)。
  T1C 截面季节性 3 列 (源 qfq kline 月收益, Heston-Sadka, lag=0)
    t1c_seas       历史同月(仅往年, >=3 个观测)月收益均值
    t1c_seas_diff  同月均值 - 往年其他月份均值 (剥掉个股整体漂移)
    t1c_seas_hit   往年同月收益为正的比例
    只用 y'<y 的整月, 天然 PIT; 全为个股截面量, 不是日历哑变量(那种按日 demean
    后选股贡献为零, 见族账本"宏观时序族"条)。

用法
────
    python scripts/build_b2_augmented.py \
        --source training_data_pit_v24_tick1.parquet \
        --output training_data_pit_v24_tick1_b2.parquet --write-features
研究用, 不进夜链。哪族过门再按 build_t1_augmented 的路子做生产化。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
KLINE_DIR = ROOT / "data" / "raw" / "kline"
MARGIN_DIR = ROOT / "data" / "raw" / "tushare" / "margin_detail"
SW_MEMBER = ROOT / "data" / "raw" / "tushare" / "sw_member" / "sw_member.parquet"
SHARE_FLOAT_DIR = ROOT / "data" / "raw" / "tushare" / "share_float"
TOP_INST_DIR = ROOT / "data" / "raw" / "tushare" / "top_inst"
HOLDER_DIR = ROOT / "data" / "raw" / "tushare" / "stk_holdernumber"

T2D_COLS = ["t2d_rz_chg5", "t2d_rz_chg20", "t2d_rz_net5", "t2d_rz_buy_z"]
T3_COLS = ["t3_ind_ret5", "t3_ind_ret20", "t3_rel5"]
T1C_COLS = ["t1c_seas", "t1c_seas_diff", "t1c_seas_hit"]
T2C_COLS = ["t2c_days_next", "t2c_ratio_next", "t2c_ratio_fwd60", "t2c_ratio_past20"]
T2B_COLS = ["t2b_cnt20", "t2b_inst_net20", "t2b_days_since"]
T3H_COLS = ["t3h_chg", "t3h_days"]
# t1c 不在默认里: 同信号 08-30 已判死, 函数保留供 --only t1c 显式调用。
# 第二波 (t2c/t2b/t3h, 09-03 凌晨) 用 --source tick1_b2 --output tick1_b3 --only 三族 建列。
FAMS = {"t2d": T2D_COLS, "t3": T3_COLS}
ALL_FAMS = {**FAMS, "t1c": T1C_COLS, "t2c": T2C_COLS, "t2b": T2B_COLS, "t3h": T3H_COLS}

T2D_LAG = 1          # 两融 D+1 早公布
T3_MIN_PEERS = 5     # 行业内(剔自身)至少 5 只才算行业收益
T1C_MIN_YEARS = 3    # 同月至少 3 个往年观测
T2C_HORIZON = 60     # 解禁"还有几天"的上限(日历日); 没有已知事件 = 上限
T2C_PAST_DAYS = 28   # “近 20 交易日”的日历日近似
T2B_LAG = 1          # 龙虎榜盘后才公布, 与 17:30 信号链赛跑, 保守滞后一天
T2B_SINCE_CAP = 60   # 距上次上榜交易日数上限
T3H_DAYS_CAP = 250   # 距最近一次股东户数公告天数上限


def _c6_of(s):
    return s.astype(str).str.extract(r"(\d{6})")[0]


def load_kline(codes, start, cols=("open", "close", "amount")):
    """读一批 qfq kline -> 长表 (date, _c6, cols...)"""
    parts = []
    for c in codes:
        f = KLINE_DIR / f"{c}.parquet"
        if not f.exists():
            continue
        k = pd.read_parquet(f, columns=["date", *cols])
        k["date"] = pd.to_datetime(k["date"])
        k = k[k["date"] >= start]
        if k.empty:
            continue
        k["_c6"] = c
        parts.append(k)
    if not parts:
        raise RuntimeError(f"kline 目录 {KLINE_DIR} 读不到任何股票")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["_c6", "date"]).sort_values(["_c6", "date"])
    return out.reset_index(drop=True)


# ── T2D 两融衍生 ──────────────────────────────────────────────
def t2d_frame(codes, kline, lag=T2D_LAG):
    files = sorted(MARGIN_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"两融明细为空: {MARGIN_DIR}")
    want = set(codes)
    parts = []
    for f in files:
        m = pd.read_parquet(f, columns=["trade_date", "ts_code", "rzye", "rzmre", "rzche"])
        m["_c6"] = m["ts_code"].astype(str).str[:6]
        parts.append(m[m["_c6"].isin(want)])
    m = pd.concat(parts, ignore_index=True)
    m["date"] = pd.to_datetime(m["trade_date"].astype(str), format="%Y%m%d")
    m = (m.drop_duplicates(["_c6", "date"]).sort_values(["_c6", "date"])
          .reset_index(drop=True))
    g = m.groupby("_c6", sort=False)
    rz = m["rzye"].where(m["rzye"] > 0)
    m["t2d_rz_chg5"] = rz / g["rzye"].shift(5).where(lambda s: s > 0) - 1
    m["t2d_rz_chg20"] = rz / g["rzye"].shift(20).where(lambda s: s > 0) - 1
    net = m["rzmre"].fillna(0) - m["rzche"].fillna(0)
    m["t2d_rz_net5"] = net.groupby(m["_c6"]).transform(
        lambda s: s.rolling(5, min_periods=5).sum()) / rz
    # 融资买入占成交额: 成交额来自 kline (同日), 比值再做 20 日 z
    amt = kline[["_c6", "date", "amount"]]
    m = m.merge(amt, on=["_c6", "date"], how="left")
    ratio = m["rzmre"] / m["amount"].where(m["amount"] > 0)
    gr = ratio.groupby(m["_c6"])
    mu = gr.transform(lambda s: s.rolling(20, min_periods=15).mean())
    sd = gr.transform(lambda s: s.rolling(20, min_periods=15).std())
    m["t2d_rz_buy_z"] = (ratio - mu) / sd.where(sd > 0)
    if lag > 0:
        # 研究口径: 只在两融表自身的日期上滞后。若将来生产化, 矩阵末日(D)通常还没有
        # 两融行做载体, 需要像 build_t1_augmented.t1a_frame 那样活缘补行, 这里不做。
        m[T2D_COLS] = m.groupby("_c6", sort=False)[T2D_COLS].shift(lag)
    out = m[["date", "_c6"] + T2D_COLS].replace([np.inf, -np.inf], np.nan)
    print(f"  T2D 两融 {len(files)} 年文件, {out['_c6'].nunique()} 只, lag={lag}")
    return out.reset_index(drop=True)


# ── T3 行业动量溢出 ───────────────────────────────────────────
def sw_l1_daily(daily_keys):
    """给 (date, _c6) 长表贴申万一级行业 (PIT 区间判定)"""
    sw = pd.read_parquet(SW_MEMBER)
    sw["_c6"] = sw["ts_code"].astype(str).str[:6]
    sw["in_date"] = pd.to_datetime(sw["in_date"], errors="coerce")
    sw["out_date"] = pd.to_datetime(sw["out_date"], errors="coerce").fillna(
        pd.Timestamp("2099-12-31"))
    sw = sw.dropna(subset=["l1_name", "in_date"])
    sw = sw[["_c6", "in_date", "out_date", "l1_name"]].sort_values(["in_date"])
    keys = daily_keys.sort_values(["date"]).reset_index(drop=True)
    j = pd.merge_asof(keys, sw, left_on="date", right_on="in_date", by="_c6",
                      direction="backward")
    j.loc[j["date"] > j["out_date"], "l1_name"] = np.nan
    return j[["date", "_c6", "l1_name"]]


def t3_frame(all_kline):
    k = all_kline[["_c6", "date", "close"]].copy()
    g = k.groupby("_c6", sort=False)["close"]
    k["r5"] = np.log(k["close"] / g.shift(5))
    k["r20"] = np.log(k["close"] / g.shift(20))
    k = k.dropna(subset=["r5"])
    ind = sw_l1_daily(k[["date", "_c6"]])
    k = k.merge(ind, on=["date", "_c6"], how="left").dropna(subset=["l1_name"])
    grp = k.groupby(["date", "l1_name"], sort=False)
    s5, n5 = grp["r5"].transform("sum"), grp["r5"].transform("count")
    s20, n20 = grp["r20"].transform("sum"), grp["r20"].transform("count")
    k["t3_ind_ret5"] = ((s5 - k["r5"]) / (n5 - 1)).where(n5 > T3_MIN_PEERS)
    k["t3_ind_ret20"] = ((s20 - k["r20"].fillna(0)) / (n20 - 1)).where(
        (n20 > T3_MIN_PEERS) & k["r20"].notna())
    k["t3_rel5"] = k["r5"] - k["t3_ind_ret5"]
    out = k[["date", "_c6"] + T3_COLS].replace([np.inf, -np.inf], np.nan)
    print(f"  T3 行业溢出: 全市场 {k['_c6'].nunique()} 只参与行业均值, "
          f"{k['l1_name'].nunique()} 个一级行业")
    return out.reset_index(drop=True)


# ── T1C 截面季节性 ────────────────────────────────────────────
def t1c_frame(codes):
    """Heston-Sadka: 往年同月月收益均值 (只用 y'<y 的整月)"""
    parts = []
    for c in codes:
        f = KLINE_DIR / f"{c}.parquet"
        if not f.exists():
            continue
        k = pd.read_parquet(f, columns=["date", "close"])
        k["date"] = pd.to_datetime(k["date"])
        k = k.sort_values("date")
        mclose = k.set_index("date")["close"].resample("ME").last().dropna()
        mret = mclose.pct_change().dropna()
        if len(mret) < 24:
            continue
        tbl = pd.DataFrame({"y": mret.index.year, "m": mret.index.month, "r": mret.values})
        piv = tbl.pivot(index="y", columns="m", values="r").sort_index()
        # 同月: 往年 expanding 均值 (shift 掉当年)
        same_mean = piv.expanding().mean().shift(1)
        same_cnt = piv.notna().cumsum().shift(1)
        same_hit = (piv > 0).where(piv.notna()).expanding().mean().shift(1)
        # 其他月: 往年全部月份的和/数 减去同月的和/数
        tot_sum = piv.sum(axis=1).cumsum().shift(1)
        tot_cnt = piv.notna().sum(axis=1).cumsum().shift(1)
        same_sum = piv.fillna(0).cumsum().shift(1)
        ex_mean = (tot_sum.values[:, None] - same_sum) / (tot_cnt.values[:, None] - same_cnt)
        seas = same_mean.where(same_cnt >= T1C_MIN_YEARS)
        # pandas>=2.1 的 stack 不再丢 NaN, 显式丢: 往年观测不足的月份就该没有行
        rows = seas.stack().rename("t1c_seas").reset_index().dropna(subset=["t1c_seas"])
        rows["t1c_seas_diff"] = rows["t1c_seas"] - ex_mean.stack().reindex(
            pd.MultiIndex.from_frame(rows[["y", "m"]])).values
        rows["t1c_seas_hit"] = same_hit.where(same_cnt >= T1C_MIN_YEARS).stack().reindex(
            pd.MultiIndex.from_frame(rows[["y", "m"]])).values
        rows["_c6"] = c
        parts.append(rows)
    if not parts:
        raise RuntimeError("没有任何股票算出 T1C")
    out = pd.concat(parts, ignore_index=True)
    print(f"  T1C 季节性: {out['_c6'].nunique()} 只有 >= {T1C_MIN_YEARS} 年同月观测")
    return out  # 月粒度 (y, m, _c6) -> 后面按矩阵行的年月贴


# ── T2C 解禁日历 (第二波) ──────────────────────────────────────────
def _read_dir(d, cols=None):
    files = sorted(Path(d).glob("*.parquet"))
    if not files:
        raise RuntimeError(f"目录为空: {d}")
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def t2c_frame(keys):
    """share_float -> 每个 (date, _c6) 的解禁日历特征

    PIT: 事件在 ann_date **次日**才算已知(公告盘后出, 与 17:30 信号链赛跑, 保守处理);
    没有 ann_date 的事件在解禁日前一律视为未知, 只进"近期已解禁"一列。
    实测公告->解禁中位仅 6 天, 所以 days_next 大多数时候是"上限"(无已知事件)。
    """
    sf = _read_dir(SHARE_FLOAT_DIR, ["ts_code", "ann_date", "float_date", "float_ratio"])
    sf["_c6"] = sf["ts_code"].astype(str).str[:6]
    sf["fd"] = pd.to_datetime(sf["float_date"].astype(str), format="%Y%m%d", errors="coerce")
    sf["ann"] = pd.to_datetime(sf["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    sf = sf.dropna(subset=["fd", "float_ratio"])
    sf["known"] = (sf["ann"] + pd.Timedelta(days=1)).fillna(sf["fd"])
    sf = sf[sf["_c6"].isin(set(keys["_c6"]))]
    H = np.timedelta64(T2C_HORIZON, "D")
    P = np.timedelta64(T2C_PAST_DAYS, "D")
    parts = []
    for c, kk in keys.groupby("_c6", sort=False):
        d = kk["date"].values.astype("datetime64[D]")[:, None]
        ev = sf[sf["_c6"] == c]
        out = pd.DataFrame({"date": kk["date"].values, "_c6": c})
        if ev.empty:
            out["t2c_days_next"] = float(T2C_HORIZON)
            out["t2c_ratio_next"] = out["t2c_ratio_fwd60"] = out["t2c_ratio_past20"] = 0.0
            parts.append(out)
            continue
        fd = ev["fd"].values.astype("datetime64[D]")[None, :]
        kn = ev["known"].values.astype("datetime64[D]")[None, :]
        r = ev["float_ratio"].values.astype(float)
        up = (kn <= d) & (d < fd) & (fd - d <= H)          # 已知、未来 60 日内
        gap = (fd - d).astype("timedelta64[D]").astype(int)
        gap_m = np.where(up, gap, 10 ** 6)
        nxt = gap_m.min(axis=1)
        has = nxt <= T2C_HORIZON
        out["t2c_days_next"] = np.where(has, nxt, T2C_HORIZON).astype(float)
        out["t2c_ratio_next"] = np.where(has, r[gap_m.argmin(axis=1)], 0.0)
        out["t2c_ratio_fwd60"] = up.astype(float) @ r
        past = (fd <= d) & (d - fd < P)
        out["t2c_ratio_past20"] = past.astype(float) @ r
        parts.append(out)
    res = pd.concat(parts, ignore_index=True)
    print(f"  T2C 解禁: 事件 {len(sf)} 条 / {sf['_c6'].nunique()} 只; 有已知待解禁的行占 "
          f"{(res['t2c_days_next'] < T2C_HORIZON).mean():.1%}")
    return res


# ── T2B 龙虎榜席位 (第二波) ────────────────────────────────────────
def t2b_frame(kline, lag=T2B_LAG):
    """top_inst 席位明细 -> 日度 -> 交易日历上的 20 日滚动 (在 kline 日期上铺开, 没上榜=0)

    上榜是交易所机械规则(涨跌幅偏离/换手), 覆盖集不是事后挑的, 无 watchlist 式前视;
    但极稀疏(池内每日中位 3 只), 多数行为 0 / 上限。机构席位净买用 20 日成交额归一。
    """
    ti = _read_dir(TOP_INST_DIR, ["trade_date", "ts_code", "exalter", "net_buy"])
    ti["_c6"] = ti["ts_code"].astype(str).str[:6]
    ti["date"] = pd.to_datetime(ti["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    ti = ti.dropna(subset=["date"])
    ti["inst_net"] = np.where(ti["exalter"].astype(str).str.contains("机构"), ti["net_buy"].fillna(0), 0.0)
    daily = ti.groupby(["_c6", "date"], as_index=False).agg(inst_net=("inst_net", "sum"))
    daily["listed"] = 1.0
    k = kline[["_c6", "date", "amount"]].merge(daily, on=["_c6", "date"], how="left")
    k[["inst_net", "listed"]] = k[["inst_net", "listed"]].fillna(0.0)
    k = k.sort_values(["_c6", "date"]).reset_index(drop=True)
    g = k.groupby("_c6", sort=False)
    k["t2b_cnt20"] = g["listed"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    amt20 = g["amount"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    k["t2b_inst_net20"] = g["inst_net"].transform(lambda s: s.rolling(20, min_periods=20).sum()) / amt20.where(amt20 > 0)
    # 距上次上榜的交易日数: 上榜日记下行号, 前向填充后相减
    pos = pd.Series(np.arange(len(k)), index=k.index, dtype=float)
    last = pos.where(k["listed"] > 0).groupby(k["_c6"]).ffill()
    k["t2b_days_since"] = (pos - last).clip(upper=T2B_SINCE_CAP).fillna(T2B_SINCE_CAP)
    if lag > 0:
        k[T2B_COLS] = k.groupby("_c6", sort=False)[T2B_COLS].shift(lag)
    out = k[["date", "_c6"] + T2B_COLS].replace([np.inf, -np.inf], np.nan)
    print(f"  T2B 龙虎榜: 席位行 {len(ti):,}, 池内上榜股日 {int(k['listed'].sum())}, lag={lag}")
    return out.reset_index(drop=True)


# ── T3H 股东户数 (第二波) ───────────────────────────────────────────
def t3h_frame(keys):
    """stk_holdernumber -> 最近一次已知公告的户数环比变化 + 距公告天数 (ann_date 次日才已知)"""
    hn = _read_dir(HOLDER_DIR, ["ts_code", "ann_date", "end_date", "holder_num"])
    hn["_c6"] = hn["ts_code"].astype(str).str[:6]
    # ann_date 有 "20250512" 与 "2025-05-12 15:09:08" 两种写法混用
    hn["ann"] = pd.to_datetime(hn["ann_date"].astype(str).str.replace("-", "").str[:8],
                               format="%Y%m%d", errors="coerce").dt.normalize()
    hn["end"] = pd.to_datetime(hn["end_date"].astype(str).str.replace("-", "").str[:8],
                               format="%Y%m%d", errors="coerce")
    hn = hn.dropna(subset=["ann", "end", "holder_num"])
    hn = hn[hn["holder_num"] > 0]
    hn = hn.sort_values(["_c6", "end", "ann"]).drop_duplicates(["_c6", "end"], keep="last")
    hn["t3h_chg"] = np.log(hn["holder_num"] / hn.groupby("_c6")["holder_num"].shift(1))
    hn["known"] = hn["ann"] + pd.Timedelta(days=1)
    hn = hn.dropna(subset=["t3h_chg"]).sort_values("known")
    k = keys.sort_values("date").reset_index(drop=True)
    j = pd.merge_asof(k, hn[["_c6", "known", "t3h_chg"]], left_on="date", right_on="known",
                      by="_c6", direction="backward")
    j["t3h_days"] = (j["date"] - j["known"]).dt.days.clip(upper=T3H_DAYS_CAP)
    print(f"  T3H 股东户数: 公告 {len(hn):,} 条 / {hn['_c6'].nunique()} 只, "
          f"矩阵行有值 {j['t3h_chg'].notna().mean():.1%}")
    return j[["date", "_c6"] + T3H_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24_tick1.parquet")
    ap.add_argument("--output", default="training_data_pit_v24_tick1_b2.parquet")
    ap.add_argument("--only", nargs="*", choices=list(ALL_FAMS), default=None)
    ap.add_argument("--write-features", action="store_true")
    ap.add_argument("--features-base",
                    default="wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json")
    a = ap.parse_args()
    t0 = time.time()
    fams = a.only or list(FAMS)

    mat = pd.read_parquet(PROC / a.source)
    mat["date"] = pd.to_datetime(mat["date"])
    add = [c for f in fams for c in ALL_FAMS[f]]
    mat = mat.drop(columns=[c for c in add if c in mat.columns])
    mat["_c6"] = _c6_of(mat["code"])
    codes = sorted(mat["_c6"].dropna().unique())
    n0 = len(mat)
    print(f"矩阵 {n0:,} 行 {len(codes)} 只 {mat['date'].min():%F}->{mat['date'].max():%F}, 加 {fams}")

    start = mat["date"].min() - pd.Timedelta(days=120)
    out = mat
    if {"t2d", "t3", "t2b"} & set(fams):
        need_all = "t3" in fams
        all_codes = (sorted(p.stem for p in KLINE_DIR.glob("[036]*.parquet"))
                     if need_all else codes)
        kl = load_kline(all_codes, start)
        print(f"  kline {kl['_c6'].nunique()} 只 {len(kl):,} 行 (>= {start:%F})")
    for fam in fams:
        if fam == "t2d":
            f = t2d_frame(codes, kl[kl["_c6"].isin(set(codes))])
        elif fam == "t3":
            f = t3_frame(kl)
        elif fam == "t2c":
            f = t2c_frame(mat[["date", "_c6"]])
        elif fam == "t2b":
            f = t2b_frame(kl[kl["_c6"].isin(set(codes))])
        elif fam == "t3h":
            f = t3h_frame(mat[["date", "_c6"]])
        else:
            f = t1c_frame(codes)
            out["y"], out["m"] = out["date"].dt.year, out["date"].dt.month
            out = out.merge(f, on=["y", "m", "_c6"], how="left").drop(columns=["y", "m"])
            if len(out) != n0:
                print(f"ERROR: 合并 {fam} 后行数变了 {n0:,} -> {len(out):,}")
                return 2
            continue
        if f.duplicated(["date", "_c6"]).any():
            print(f"ERROR: {fam} 源表 (date, code) 有重复")
            return 2
        out = out.merge(f, on=["date", "_c6"], how="left")
        if len(out) != n0:
            print(f"ERROR: 合并 {fam} 后行数变了 {n0:,} -> {len(out):,}")
            return 2
    out = out.drop(columns=["_c6"])

    last = out[out["date"] == out["date"].max()]
    for fam in fams:
        cols = ALL_FAMS[fam]
        print(f"  {fam}: 全表非空 " + " ".join(f"{c}={out[c].notna().mean():.1%}" for c in cols)
              + f" | 末日全有值 {last[cols].notna().all(axis=1).mean():.1%}")
    tmp = (PROC / a.output).with_suffix(".parquet.b2tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(PROC / a.output)

    if a.write_features:
        base = json.loads((PROC / a.features_base).read_text(encoding="utf-8"))
        feats = list(base["selected_features"])
        for fam in fams:
            cols = ALL_FAMS[fam]
            assert not set(cols) & set(feats)
            p = PROC / f"features_V24PUT_{fam.upper()}.json"
            p.write_text(json.dumps({"selected_features": feats + cols}, ensure_ascii=False),
                         encoding="utf-8")
            print(f"  特征表 {p.name}: {len(feats)} + {len(cols)}")
    print(f"B2 增广完成 -> {a.output} ({out.shape[0]:,} x {out.shape[1]}) 耗时 {time.time()-t0:.0f}s")
    print("AUG_DONE_B2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
