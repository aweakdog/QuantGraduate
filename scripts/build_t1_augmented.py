"""T1A/T1B 增广列 —— 分点分配上线的数据层 (2026-08-31)

把 8 列加到线上训练矩阵上, 供各线按 --features-from 各取自己那 4 列:

  T1A 订单结构 4 列 (源: eez040 抽的 t1a_daily 面板, 见 t1a_order_features.py)
    t1a_big_buy_ma5    大单买入占比 5 日均
    t1a_long_buy_ma5   多笔买入占比 5 日均
    t1a_big_sell_ma5   大单卖出占比 5 日均
    t1a_long_sell_ma5  多笔卖出占比 5 日均
  T1B 隔夜-日内累计分解 4 列 (源: qfq kline, 无外部依赖)
    t1b_ovn20/60       20/60 日累计隔夜收益 sum(log(open/prev_close))
    t1b_intra20/60     20/60 日累计日内收益 sum(log(close/open))

为什么 8 列一起加, 而模型仍是专才
────────────────────────────────
矩阵带全部 8 列, 但每条线的特征表只选自己那 4 列 —— 训出来的仍是 T1A 专才或
T1B 专才模型。这**不是**上 T1X 合体模型(8 列同喂一个模型, 08-31 判死: bare 层
最强 +38.45pp 但全配置输专才 TX5 -11.05 / TX10 -2.6)。
这样做的收益是 train_file 不变 => **不动指纹, 不用 --init, 不碰真金账本**;
分线差异全落在 features_from(不进指纹), 以后给哪条线换族是改一行 + 重启。
与 GF1 门控当年的做法同构(CGO 6 列进同一矩阵, 门控线只换特征表与缓存文件)。

分配 (LSW 逐线扫描 20 种子配对中位, 见 experiment_board 2026-08-31)
    T1A  aggr10w +20.35 / steady2w +19.40 / aggr5w +8.50 / base5w_aggr(同 aggr5w)
    T1B  steady5w +16.15 / fyf100w +8.60 / base5w_steady(同 steady5w)
    基线 aggr2w (双族 wash, T1A 仅 +2.15 且 10/20)

口径锁定
────────
算法与研究构建 (/tmp/build_t1a_cols.py + /tmp/build_t1b_cols.py, T1A20/T1B20
20 面板证据就是它们跑出来的) 逐字一致:
  * T1A: 面板原始 4 列各自 rolling(5, min_periods=3).mean(), 按 code 分组
  * T1B: ovn=log(open/prev_close), intra=log(close/open), rolling(20/60,
    min_periods=**满窗**).sum() —— 不放宽 min_periods, 新股前 60 天就该是 NaN
  * 两族都按 (date, code6) 左连接进矩阵, 缺就 NaN (LightGBM 原生处理)

用法
────
    python scripts/build_t1_augmented.py                    # 原地增广线上矩阵
    python scripts/build_t1_augmented.py --only t1b         # 只加 T1B (无 040 依赖)
    python scripts/build_t1_augmented.py --write-features   # 另外生成两份特征表
daily_rebuild §4.3 每晚在 feature_engine 重建矩阵之后调用 (重建会丢增广列)。
幂等: 已有的这 8 列会先剔除再重算。
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
T1A_PANEL = PROC / "t1a_daily"

T1A_RAW = ["t1a_big_buy", "t1a_long_buy", "t1a_big_sell", "t1a_long_sell"]
T1A_COLS = [c + "_ma5" for c in T1A_RAW]
T1B_COLS = ["t1b_ovn20", "t1b_ovn60", "t1b_intra20", "t1b_intra60"]

T1A_MA = 5           # 面板 -> 特征的平滑窗
T1A_MA_MIN = 3       # 满窗不足时也出值 (逐笔断供一两天不该让整列变 NaN)
T1A_LAG = 1          # 生产正确对齐, 详见 t1a_frame 的长注
EDGE_MAX = 3         # 活缘补行上限, 与 build_tick_augmented 同值
MIN_COVER = {"t1a": 0.90, "t1b": 0.85}   # 与研究构建的 assert 同阈值

_COL_MAP = {"时间": "date", "开盘价": "open", "收盘价": "close"}


def t1a_frame(base_keys=None, lag=T1A_LAG):
    """t1a_daily 面板 -> ma5 平滑 + PIT 滞后后的 (date, _c6) 表

    为何必须 lag>=1 (2026-08-31 核实)
    ────────────────────────────────
    逐笔包是供应商 **D+1 早晨** 才发的: 20 天连续日志显示每天 16:40 那班拿到的
    最新永远是 D-1, 实测到达时刻稳定在 D+1 07:20~07:28。而信号链 D 17:30 就跑,
    所以"矩阵行 D 配面板 D"(lag=0, 研究构建的做法)在生产上拿不到 —— 同
    build_tick_augmented 头注的铁律与 MEMORY §8 PIT 对齐原则。
    lag=0 只保留给对比臂用(量化"当日信息值多少"), **不得进生产**。

    base_keys: 基矩阵的 (_c6, date) 表, 用于活缘补行 —— 面板末日总比矩阵末日
    早一天, 不补就会让信号日那行的 T1A 全 NaN(而且不报错, 静默失效)。
    """
    files = sorted(T1A_PANEL.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"T1A 面板为空: {T1A_PANEL} —— 需先在 eez040 跑 "
                           "scripts/t1a_order_features.py 并 rsync 过来")
    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    panel["_c6"] = panel["code"].astype(str).str.zfill(6)
    panel = panel.sort_values(["_c6", "date"])
    miss = [c for c in T1A_RAW if c not in panel.columns]
    if miss:
        raise RuntimeError(f"T1A 面板缺列 {miss} (抽取器版本不匹配?)")
    for c in T1A_RAW:
        panel[c + "_ma5"] = panel.groupby("_c6")[c].transform(
            lambda s: s.rolling(T1A_MA, min_periods=T1A_MA_MIN).mean())
    z = panel[["_c6", "date"] + T1A_COLS].copy()
    p_last = panel["date"].max()

    # 活缘补行 + 滞后 (逐字照 build_tick_augmented §2.5 的做法)
    n_edge = 0
    if lag > 0:
        if base_keys is not None:
            last = z.groupby("_c6", sort=False)["date"].max()
            tail = base_keys[["_c6", "date"]].merge(last.rename("_last"), on="_c6")
            tail = tail[tail["date"] > tail["_last"]]
            tail = tail.sort_values(["_c6", "date"]).groupby("_c6").head(EDGE_MAX)
            if len(tail):
                n_edge = len(tail)
                z = pd.concat([z, tail[["_c6", "date"]]], ignore_index=True)
        z = z.sort_values(["_c6", "date"])
        z[T1A_COLS] = z.groupby("_c6", sort=False)[T1A_COLS].shift(lag)
        if n_edge:
            # 只给补行 ffill 兜底(供应商断供时), 历史行逐字节不变
            emask = z["date"] > p_last
            filled = z.groupby("_c6", sort=False)[T1A_COLS].ffill(limit=EDGE_MAX)
            z.loc[emask, T1A_COLS] = filled.loc[emask, T1A_COLS]
    print(f"  T1A 面板 {len(files)} 天 {panel['_c6'].nunique()} 只 "
          f"(末日 {p_last:%Y-%m-%d}), lag={lag}"
          + (f", 活缘补 {n_edge} 行" if n_edge else ""))
    return z.reset_index(drop=True)


def t1b_frame(codes):
    """qfq kline -> 隔夜/日内累计分解表"""
    parts = []
    for c in codes:
        f = KLINE_DIR / f"{c}.parquet"
        if not f.exists():
            continue
        k = pd.read_parquet(f).rename(columns=_COL_MAP)
        if not {"date", "open", "close"}.issubset(k.columns):
            continue
        k["date"] = pd.to_datetime(k["date"])
        k = k.sort_values("date")
        ovn = np.log(k["open"] / k["close"].shift(1))
        intra = np.log(k["close"] / k["open"])
        parts.append(pd.DataFrame({
            "date": k["date"], "_c6": c,
            "t1b_ovn20": ovn.rolling(20, min_periods=20).sum(),
            "t1b_ovn60": ovn.rolling(60, min_periods=60).sum(),
            "t1b_intra20": intra.rolling(20, min_periods=20).sum(),
            "t1b_intra60": intra.rolling(60, min_periods=60).sum(),
        }))
    if not parts:
        raise RuntimeError(f"没有任何股票算出 T1B (kline 目录 {KLINE_DIR} 空?)")
    out = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    print(f"  T1B kline {len(parts)} 只")
    return out


def write_feature_tables(base_from):
    """按生产基础特征表 + 4 列, 生成两族的特征表

    ⚠ 基础部分刻意取**生产在用**的 FEATURES_FROM 而不是研究用的
    features_V24PUT_sel.json: 两者 80 列集合完全相同、只有顺序不同。取生产顺序
    可保证切线时唯一的变化就是 "+4 列", 基础模型不因列序变动而一起漂移。
    """
    base = json.loads((PROC / base_from).read_text(encoding="utf-8"))
    feats = list(base["selected_features"])
    for tag, cols in (("T1A", T1A_COLS), ("T1B", T1B_COLS)):
        dup = [c for c in cols if c in feats]
        if dup:
            raise RuntimeError(f"基础特征表里已有 {dup}, 不能重复追加")
        p = PROC / f"features_V24PUT_{tag}.json"
        p.write_text(json.dumps({"selected_features": feats + cols},
                                ensure_ascii=False), encoding="utf-8")
        print(f"  特征表 {p.name}: {len(feats)} + {len(cols)} = "
              f"{len(feats) + len(cols)} 列")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="training_data_pit_v24.parquet",
                    help="源矩阵 (data/processed/ 下)")
    ap.add_argument("--output", default=None, help="不给则原地覆盖 source")
    ap.add_argument("--only", choices=["t1a", "t1b"], default=None,
                    help="只加一族 (T1A 依赖 040 面板, 断供时可只加 T1B)")
    ap.add_argument("--t1a-lag", type=int, default=T1A_LAG,
                    help="T1A 的 PIT 滞后天数。生产必须 >=1(逐笔包 D+1 早晨才到); "
                         "0 仅用于对比臂量化当日信息的价值")
    ap.add_argument("--write-features", action="store_true",
                    help="另外生成 features_V24PUT_T1A/T1B.json")
    ap.add_argument("--features-base",
                    default="wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json",
                    help="生成特征表时的基础表 (须与 live_config.FEATURES_FROM 一致)")
    a = ap.parse_args()
    t0 = time.time()

    fams = [a.only] if a.only else ["t1a", "t1b"]
    src = PROC / a.source
    out_path = PROC / (a.output or a.source)
    if not src.exists():
        print(f"ERROR: 源矩阵不存在 {src}")
        return 2
    mat = pd.read_parquet(src)
    mat["date"] = pd.to_datetime(mat["date"])
    add = (T1A_COLS if "t1a" in fams else []) + (T1B_COLS if "t1b" in fams else [])
    # 幂等: 剔掉旧列再重算
    mat = mat.drop(columns=[c for c in add if c in mat.columns])
    c6 = mat["code"].astype(str).str.extract(r"(\d{6})")[0]
    codes = sorted(c6.dropna().unique())
    print(f"矩阵 {len(mat):,} 行 {len(codes)} 只 (源 {src.name}), 加 {fams}")

    mat["_c6"] = c6
    n0 = len(mat)
    out = mat
    if a.t1a_lag == 0 and "t1a" in fams:
        print("  ⚠ --t1a-lag 0: 仅对比臂口径, 生产拿不到当日逐笔, 别拿它上线")
    for fam in fams:
        f = (t1a_frame(base_keys=mat[["_c6", "date"]], lag=a.t1a_lag)
             if fam == "t1a" else t1b_frame(codes))
        if f.duplicated(["date", "_c6"]).any():
            print(f"ERROR: {fam} 源表 (date, code) 有重复, 合并会放大行数")
            return 2
        out = out.merge(f, on=["date", "_c6"], how="left")
        if len(out) != n0:
            print(f"ERROR: 合并 {fam} 后行数变了 {n0:,} -> {len(out):,}")
            return 2
    out = out.drop(columns=["_c6"])

    # 末日体检: 当晚出信号要用最新日的值, 末日覆盖塌了宁可整步失败
    last = out[out["date"] == out["date"].max()]
    covs = {}
    for fam in fams:
        cols = T1A_COLS if fam == "t1a" else T1B_COLS
        cov = last[cols].notna().all(axis=1).mean() if len(last) else 0.0
        covs[fam] = cov
        if cov < MIN_COVER[fam]:
            print(f"ERROR: 最新日 {fam} 覆盖率仅 {cov:.1%} "
                  f"(<{MIN_COVER[fam]:.0%}), 拒绝落盘")
            return 2

    tmp = out_path.with_suffix(".parquet.t1tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    if a.write_features:
        write_feature_tables(a.features_base)
    nn = {c: f"{out[c].notna().mean():.1%}" for c in add}
    print(f"T1 增广完成 -> {out_path.name} ({out.shape[0]:,} x {out.shape[1]}) "
          f"耗时 {time.time() - t0:.0f}s")
    print(f"全表非空率: {nn}")
    print("最新日覆盖: " + " ".join(f"{k}={v:.1%}" for k, v in covs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
