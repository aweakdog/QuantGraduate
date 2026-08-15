# -*- coding: utf-8 -*-
"""探查逐笔委托/成交/行情三个文件的字段语义与抽取成本

在设计微观结构特征之前必须先搞清楚:
  委托类型 / 委托代码 的取值 -> 能不能区分新增委托与撤单
  BS标志 -> 主动买卖方向可不可用
  叫卖序号 / 叫买序号 -> 成交能不能回连到委托 (决定能否做委托生命周期特征)
  单只股票的抽取耗时 -> 决定全量 2331 天 x 600 只可行不可行

用法
────
    python scripts/tick_probe.py 20260807 600519.SH
"""
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
import py7zr

warnings.filterwarnings("ignore")

TICK = Path("/home/yliog/tickdata/----逐笔委托成交行情-明细---")
FILES = ["逐笔委托.csv", "逐笔成交.csv", "行情.csv"]


def resolve(day: str):
    y, m = day[:4], day[4:6]
    for sub in (m, y + m):
        p = TICK / y / sub / f"{day}.7z"
        if p.exists():
            return p
    return None


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20260807"
    code = sys.argv[2] if len(sys.argv) > 2 else "600519.SH"
    pack = resolve(day)
    if pack is None:
        sys.exit(f"没找到 {day} 的包")

    tmp = tempfile.mkdtemp(dir="/tmp")
    try:
        t0 = time.time()
        with py7zr.SevenZipFile(pack, mode="r") as z:
            z.extract(path=tmp, targets=[f"{day}/{code}/{f}" for f in FILES])
        el = time.time() - t0
        print(f"包 {pack.name} {pack.stat().st_size / 1e9:.2f}GB")
        print(f"抽 1 只 x 3 文件耗时 {el:.1f}s")

        base = Path(tmp) / day / code
        o = pd.read_csv(base / "逐笔委托.csv", encoding="gbk")
        t = pd.read_csv(base / "逐笔成交.csv", encoding="gbk")
        q = pd.read_csv(base / "行情.csv", encoding="gbk")
        print(f"行数: 委托 {len(o)}  成交 {len(t)}  行情 {len(q)}")

        print("\n===== 逐笔委托 =====")
        for c in ["委托类型", "委托代码"]:
            print(f"  {c}: {o[c].value_counts(dropna=False).to_dict()}")
        print(f"  委托价格>0 占比 {(o['委托价格'] > 0).mean():.3f}"
              f"  委托数量>0 占比 {(o['委托数量'] > 0).mean():.3f}")
        print(f"  委托编号>0 {(o['委托编号'] > 0).mean():.3f}"
              f"  交易所委托号>0 {(o['交易所委托号'] > 0).mean():.3f}")
        print("  按 委托类型 x 委托代码 的量价:")
        g = o.groupby(["委托类型", "委托代码"]).agg(
            n=("委托数量", "size"), qty=("委托数量", "sum"),
            px_pos=("委托价格", lambda s: (s > 0).mean()))
        print(g.to_string())

        print("\n===== 逐笔成交 =====")
        print(f"  BS标志: {t['BS标志'].value_counts(dropna=False).to_dict()}")
        print(f"  成交代码: {t['成交代码'].value_counts(dropna=False).to_dict()}")
        print(f"  叫卖序号>0 {(t['叫卖序号'] > 0).mean():.3f}"
              f"  叫买序号>0 {(t['叫买序号'] > 0).mean():.3f}")
        print(f"  成交额合计 {(t['成交价格'] / 1e4 * t['成交数量']).sum() / 1e8:.3f} 亿")

        # 成交能否回连委托: 用交易所委托号做键
        add = o[o["委托类型"] == "A"] if "委托类型" in o else o
        ids = set(add["交易所委托号"].astype("int64"))
        for col in ["叫买序号", "叫卖序号"]:
            hit = t[col].astype("int64").isin(ids).mean()
            print(f"  {col} 命中 交易所委托号 比例 {hit:.3f}")

        print("\n===== 行情 =====")
        print(f"  快照数 {len(q)}  叫买总量/叫卖总量 非零占比 "
              f"{(q['叫买总量'] > 0).mean():.3f}/{(q['叫卖总量'] > 0).mean():.3f}")
        print(f"  10 档买量非零占比 {(q['申买量10'] > 0).mean():.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
