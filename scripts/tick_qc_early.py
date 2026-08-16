# -*- coding: utf-8 -*-
"""逐笔包早盘委托量质检: 抽取前置闸, 拦"供应商静默缺早盘"

事故原型 (2026-08-16 实测, docs/findings_2026-08-16 §4)
────────────────────────────────────────────────────────
123 渠道 7/9~7/21 的包体积比百度渠道小 6~18%: 96% 股票缺集合竞价委托,
9:30-10:00 缺 76%, 11:00 后完好。逐笔成交完好, 只有委托被截断。
这种缺失不报错、不缺文件、算出来的 ord_*/cxl_* 特征全是错的 —— 唯一的
拦法是分布检验: 坏日是全市场集体异常, 好日的"早盘委托占比"截面中位数
在窄带里波动 (成交时段 U 型, 委托更集中在开盘)。

法子
────
每包抽固定间隔采样的 ~120 只票, 只读 逐笔委托.csv 的 时间 列:
    share_pre10 = 行数(t < 10:00) / 行数(全天)
记 {日期, 截面中位, p25, 中位行数} 进 data/processed/tick_qc_early.parquet,
新日对历史(未标坏的记录)做稳健 z:
    z = (x - median_hist) / max(1.4826*MAD_hist, 0.01)
z < -5 或 x < 0.6*median_hist 判坏 (0713 实测坏日占比从 ~0.4 掉到 ~0.15,
z 深负; 0.01 的 MAD 地板防"历史太稳导致鸡毛蒜皮也报警")。
历史不足 8 天只记录不判定 (先跑 --baseline 建基线)。

坏日的处置由调用方定: tick_daily_extract 对坏日跳过抽取(不产毒特征),
矩阵端 --require-fresh 3 兜 staleness, 断供超限宁可停更。

用法
────
    python scripts/tick_qc_early.py --day 20260814                 # 查一天并记录
    python scripts/tick_qc_early.py --day 20260713 \
        --pack ~/tickdata123/202607/20260713.7z --no-record        # 验证坏包(不入库)
    python scripts/tick_qc_early.py --baseline 20260105 20260807   # 建基线(隔日采样)
退出码: 0=通过/仅记录  3=判坏  4=包缺失/读不出
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_micro_features import F_ORD, _rd, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/processed/tick_qc_early.parquet"
N_SAMPLE = 120
MIN_HIST = 8
Z_BAD, FLOOR_RATIO = -5.0, 0.6
T10 = 100000000          # HHMMSSmmm: 10:00:00.000


def sample_codes():
    """全 universe 排序后等距采样 —— 跨日稳定, 分布检验才有可比性"""
    c6 = set()
    for p in (ROOT / "data/universe/universe_pit.parquet",
              ROOT / "data/universe/universe_pit_2019.parquet"):
        if p.exists():
            c6 |= set(pd.read_parquet(p)["code"].astype(str)
                      .str.extract(r"(\d{6})")[0].dropna())
    c6 = sorted(c6)
    if not c6:
        sys.exit("找不到 universe 文件")
    step = max(1, len(c6) // N_SAMPLE)
    return [f"{c}.{'SH' if c[0] == '6' else 'SZ'}" for c in c6[::step]]


def measure_pack(pack: Path, codes):
    """返回逐股 (share_pre10, n_rows) 列表。只解压采样股的委托文件。"""
    import py7zr
    with py7zr.SevenZipFile(pack, mode="r") as z:
        names = set(z.getnames())
    dirs = {}
    tail = "/" + F_ORD
    for n in names:
        if n.endswith(tail):
            dirs[n.split("/")[-2]] = n[:-len(tail)]
    have = [c for c in codes if c in dirs]
    if not have:
        return []
    out = []
    tmp = tempfile.mkdtemp(prefix="qc_", dir="/tmp")
    try:
        tgt = [f"{dirs[c]}/{F_ORD}" for c in have]
        with py7zr.SevenZipFile(pack, mode="r") as z:
            z.extract(path=tmp, targets=tgt)
        for c in have:
            df = _rd(Path(tmp) / dirs[c] / F_ORD, {"时间"})
            if df is None or "时间" not in df.columns or not len(df):
                continue
            t = pd.to_numeric(df["时间"], errors="coerce").dropna()
            if not len(t):
                continue
            out.append((float((t < T10).mean()), int(len(t))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def qc_one(day: str, pack: Path):
    per = measure_pack(pack, sample_codes())
    if len(per) < N_SAMPLE // 3:
        print(f"{day}: 采样股票只解出 {len(per)} 只, 包不可用")
        return None
    sh = np.array([p[0] for p in per])
    rows = np.array([p[1] for p in per])
    return {"date": day, "n_stk": len(per),
            "med_pre10": float(np.median(sh)),
            "p25_pre10": float(np.percentile(sh, 25)),
            "med_rows": float(np.median(rows))}


def judge(rec, hist: pd.DataFrame):
    """返回 (bad, why)。hist = 未标坏的既有记录 (不含 rec 当日)。"""
    if len(hist) < MIN_HIST:
        return False, f"历史仅 {len(hist)} 天 (<{MIN_HIST}), 只记录不判定"
    x = rec["med_pre10"]
    h = hist["med_pre10"].to_numpy()
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    z = (x - med) / max(1.4826 * mad, 0.01)
    why = f"med_pre10={x:.3f} 基线={med:.3f} z={z:+.1f}"
    if z < Z_BAD or x < FLOOR_RATIO * med:
        return True, why + f" (z<{Z_BAD} 或 <{FLOOR_RATIO}x基线)"
    return False, why


def load_qc():
    if QC.exists():
        return pd.read_parquet(QC)
    return pd.DataFrame(columns=["date", "n_stk", "med_pre10", "p25_pre10",
                                 "med_rows", "bad", "why"])


def hist_good(df, day):
    """历史基线 = 未标坏且不含当日的既有记录"""
    if not len(df):
        return df
    return df[(~df["bad"].astype(bool)) & (df["date"] != day)]


def save_rec(rec, bad, why):
    df = load_qc()
    rec = dict(rec, bad=bool(bad), why=why)
    df = pd.concat([df[df["date"] != rec["date"]], pd.DataFrame([rec])],
                   ignore_index=True).sort_values("date")
    QC.parent.mkdir(parents=True, exist_ok=True)
    tmp = QC.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(QC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYYMMDD")
    ap.add_argument("--pack", help="显式包路径 (验证异渠道副本用)")
    ap.add_argument("--no-record", action="store_true", help="不写入 QC 库")
    ap.add_argument("--baseline", nargs=2, metavar=("START", "END"),
                    help="建基线: 区间内隔一天采一天, 全部记录")
    a = ap.parse_args()

    if a.baseline:
        s, e = a.baseline
        days = sorted(p.stem for p in
                      Path(os.environ.get(
                          "TICK_DIR",
                          "/home/yliog/tickdata/----逐笔委托成交行情-明细---"))
                      .glob("*/*/*.7z") if s <= p.stem <= e)[::2]
        print(f"基线: {len(days)} 天 ({s}~{e} 隔日采样)")
        for d in days:
            pk = resolve(d)
            if pk is None:
                continue
            rec = qc_one(d, pk)
            if rec is None:
                continue
            bad, why = judge(rec, hist_good(load_qc(), d))
            save_rec(rec, bad, why)
            print(f"  {d}: {'坏' if bad else '好'}  {why}")
        return 0

    if not a.day:
        ap.error("--day 或 --baseline 必选其一")
    pack = Path(a.pack).expanduser() if a.pack else resolve(a.day)
    if pack is None or not pack.exists():
        print(f"{a.day}: 找不到包")
        return 4
    rec = qc_one(a.day, pack)
    if rec is None:
        return 4
    bad, why = judge(rec, hist_good(load_qc(), a.day))
    print(f"{a.day}: {'✗ 坏包' if bad else '✓ 通过'}  {why}  "
          f"(采样 {rec['n_stk']} 只, 中位行数 {rec['med_rows']:.0f})")
    if not a.no_record:
        save_rec(rec, bad, why)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
