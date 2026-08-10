"""把融券余额特征挂到现有训练矩阵上, 产出新文件 (不碰原文件)

为什么不走 feature_engine 全量重建: 那要 6 小时, 而我们只想加 2 列。直接在
现有矩阵上 merge, 除了多出这 2 列以外与原矩阵【逐格相同】—— 这才是干净的 A/B,
全量重建反而会引入无关的重算漂移(见 diag_rebuild_drift.py 的教训)。

为什么是 rqye (融券余额): 16 张 tushare 表逐字段过完 IC + top5 分层双关卡后,
它是唯一存活的:
    top5 超额 两窗口都为正   窗口A +0.316% / 窗口B +0.103% (按流通市值归一后)
    扛过市值归一             未归一是 +0.422%/+0.165%, 归一后仍保住大部分,
                            说明不是市值代理(rzmre/rzche/rzye 归一后全转负)
    无零值并列              池内 rqye==0 每天只有 3 只(0.5%), 分布连续
    且它是全场唯一在【窗口A】top5 为正的因子 —— 窗口A 是我们唯一的堵点,
    模型在那里的 top5 超额是 -0.164%
诚实标注: 独立采样(每5日)的 t 值只有 A 1.3 / B 0.6, 低于事前定的 t>2。
所以这是"值得花一次机时证死或证实的线索", 不是已确立的因子。先验很弱。

前视检查: 融资融券明细由交易所在 T 日收盘后(约18-20点)发布。我们的执行是
t1close —— T 日收盘出信号, T+1 日尾盘成交, 信号计算发生在 T 日收盘之后,
那时 T 日的两融数据已发布。所以用 T 日 rqye 预测 T+1 建仓不构成前视。

    python scripts/add_margin_features.py
    python scripts/add_margin_features.py --out training_data_pit_2019_rq.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"
TS = ROOT / "data" / "raw" / "tushare"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="training_data_pit_2019.parquet")
ap.add_argument("--out", default="training_data_pit_2019_rq.parquet")
args = ap.parse_args()


def read_ts(name, cols):
    files = sorted(p for p in (TS / name).glob("*.parquet") if p.stem != "None")
    if not files:
        raise SystemExit(f"没有 {name} 数据")
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in files],
                  ignore_index=True)
    d["date"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d",
                               errors="coerce")
    return d.dropna(subset=["date"]).drop(columns=["trade_date"])


def main():
    src = PROC / args.src
    print(f"读训练矩阵 {src.name} ...", flush=True)
    t = pd.read_parquet(src)
    print(f"  {len(t):,} 行 x {t.shape[1]} 列")

    md = read_ts("margin_detail", ["trade_date", "ts_code", "rqye"])
    db = read_ts("daily_basic", ["trade_date", "ts_code", "circ_mv"])
    # circ_mv 单位是万元, rqye 是元 —— 统一到元, 否则比值量纲没有直观含义
    db["circ_mv"] = db["circ_mv"] * 1e4
    m = md.merge(db, on=["date", "ts_code"], how="left")
    m = m.rename(columns={"ts_code": "code"})

    # 特征1: 融券余额 / 流通市值 —— 这是过了关卡的那个版本(剥离规模成分)
    m["rq_bal_mv"] = m["rqye"] / m["circ_mv"].replace(0, np.nan)
    # 特征2: 上面这个的【当日截面分位】。为什么两个都要: 树模型对单个特征的
    # 单调变换不敏感, 但它训练在【跨日期池化】的数据上 —— 原始比值的分布随
    # 两融规模的时代变化而漂移, 截面分位是平稳的。两者对树模型不等价。
    m["rq_bal_pct"] = m.groupby("date")["rq_bal_mv"].rank(pct=True)

    keep = ["date", "code", "rq_bal_mv", "rq_bal_pct"]
    m = m[keep].dropna(subset=["rq_bal_mv"])
    print(f"  融券特征 {len(m):,} 行, {m['code'].nunique()} 只股, "
          f"{m['date'].min():%Y-%m-%d} ~ {m['date'].max():%Y-%m-%d}")

    before = t.shape[1]
    out = t.merge(m, on=["date", "code"], how="left")
    cov = out["rq_bal_mv"].notna().mean()
    print(f"  合并后 {out.shape[1]} 列 (新增 {out.shape[1]-before} 个)")
    print(f"  覆盖率 {cov*100:.1f}% —— 两融标的才有数据, 其余为 NaN "
          f"(LightGBM 原生处理缺失, 不填充)")
    if cov < 0.4:
        raise SystemExit(f"覆盖率过低 ({cov*100:.1f}%), 大概率是键没对上, 中止")
    if len(out) != len(t):
        raise SystemExit(f"行数变了 {len(t)} -> {len(out)}, merge 有重复键, 中止")

    dst = PROC / args.out
    out.to_parquet(dst, index=False)
    print(f"\n已写出 {dst}")
    print("下一步: eval_grid.py 用 mb_dmw_rq 变体跑 features -> caches -> eval,"
          "\n        与 mb_dmw 严格对照(只差这 2 列)")


if __name__ == "__main__":
    main()
