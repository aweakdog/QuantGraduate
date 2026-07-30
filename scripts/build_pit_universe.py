"""构建 point-in-time 无偏股票池

规则 (每个生效日 T 只使用 T 之前可得的信息):
  1. T 时点已上市满 --min-listed-years 年   (交易所权威 list_date)
  2. T 时点尚未退市                          (delist_date 为空 或 > T)
  3. T 前 --lookback 个交易日中实际有数据 >= 75%   (排除长期停牌)
  4. 按 --rank-by 排序取前 --top-n 名

排序口径 (均 PIT 安全):
  mcap : 流通市值 = vwap x outstanding_share, 其中 vwap = amount/volume 为真实未复权均价
         (本地 close 是前复权, 早期最多偏离真实价 50%, 不能直接乘股本)
  adv  : 日均成交额, 原始值, 但排名波动大 -> 成分股换手高

产出 data/universe/universe_pit.parquet:
  effective_date | code | mcap | adv | rank | has_kline

用法:
  python scripts/build_pit_universe.py --top-n 300 --freq semiannual --rank-by mcap
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "data/raw/kline"
META = ROOT / "data/universe/pit_metadata.parquet"
OUT = ROOT / "data/universe/universe_pit.parquet"


def _load_one(p: Path):
    """读单只K线, 返回 (code, 日期int64数组, amount数组, mcap数组)

    只回传 numpy 数组: 主循环里用 searchsorted + numpy 聚合, 比 pandas 快两个量级
    """
    try:
        k = pd.read_parquet(p, columns=["date", "amount", "volume",
                                        "outstanding_share"])
    except Exception:
        return None
    if not len(k):
        return None
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date")
    vol = k["volume"].to_numpy(dtype="float64", na_value=np.nan)
    amt = k["amount"].to_numpy(dtype="float64", na_value=np.nan)
    osh = k["outstanding_share"].to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcap = np.where(vol > 0, amt / vol, np.nan) * osh
    return p.stem, k["date"].to_numpy(dtype="datetime64[ns]").astype("int64"), amt, mcap


def trading_calendar() -> pd.DatetimeIndex:
    """用多只高流动性股票的日期并集近似交易日历"""
    ds = set()
    for c in ("000001", "600000", "000063", "600519"):
        p = KL / f"{c}.parquet"
        if p.exists():
            ds |= set(pd.to_datetime(pd.read_parquet(p, columns=["date"])["date"]))
    return pd.DatetimeIndex(sorted(ds))


def rebalance_dates(cal, start, end, freq):
    """生效日 = 每年1月/7月(半年) 或 每年1月(年度) 的首个交易日"""
    months = {"semiannual": (1, 7), "annual": (1,), "quarterly": (1, 4, 7, 10)}[freq]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for y in range(start.year, end.year + 1):
        for m in months:
            anchor = pd.Timestamp(year=y, month=m, day=1)
            nxt = cal[cal >= anchor]
            if len(nxt) and start <= nxt[0] <= end:
                out.append(nxt[0])
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--min-listed-years", type=float, default=2.0)
    ap.add_argument("--freq", default="semiannual",
                    choices=["annual", "semiannual", "quarterly"])
    ap.add_argument("--rank-by", default="mcap", choices=["mcap", "adv"])
    ap.add_argument("--min-adv", type=float, default=5e7,
                    help="流动性下限(元), 低于此日均成交额的剔除")
    ap.add_argument("--start", default="2022-09-01")
    ap.add_argument("--end", default="2026-07-27")
    ap.add_argument("--out", default=None,
                   help="输出文件名 (data/universe/ 下), 默认 universe_pit.parquet。"
                        "做实验时务必指定其它名字 —— 默认文件被 live_signal/daily_rebuild 依赖")
    ap.add_argument("--jobs", type=int, default=16, help="加载K线的并行进程数")
    a = ap.parse_args()

    out_path = OUT if not a.out else OUT.parent / a.out
    cfg_path = out_path.with_name(out_path.stem + "_config.json")

    meta = pd.read_parquet(META)
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    print(f"元数据 {len(meta)} 只 (在市 {meta['delist_date'].isna().sum()}, "
          f"已退市 {meta['delist_date'].notna().sum()})")

    cal = trading_calendar()
    rebs = rebalance_dates(cal, a.start, a.end, a.freq)
    print(f"交易日历 {len(cal)} 天 | 生效日 {len(rebs)} 个 ({a.freq}): "
          f"{', '.join(str(d.date()) for d in rebs)}")

    # 预载 kline: amount 用于流动性, vwap x outstanding_share 用于真实市值
    files = sorted(KL.glob("*.parquet"))
    print(f"\n加载 kline (amount, volume, outstanding_share) | {len(files)} 个文件, {a.jobs} 进程 ...")
    store = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(_load_one, files, chunksize=32):
            if res is not None:
                store[res[0]] = res[1:]
    print(f"  载入 {len(store)} 只")

    # meta 转 records: 避免 iterrows 每行构造 Series
    mrecs = [(r["code"], r["list_date"], r["delist_date"])
             for r in meta.to_dict("records")]

    recs = []
    for T in rebs:
        win = cal[cal < T][-a.lookback:]
        if len(win) < a.lookback * 0.5:
            continue
        w0 = win[0]
        min_cover = len(win) * 0.75
        t_i8, w0_i8 = T.value, w0.value
        cand = []
        for code, ld, dd in mrecs:
            # 条件1: 上市满 N 年
            if pd.isna(ld) or (T - ld).days < a.min_listed_years * 365.25:
                continue
            # 条件2: T 时点未退市
            if pd.notna(dd) and dd <= T:
                continue
            s = store.get(code)
            if s is None:
                continue
            # 条件3: 窗口内数据覆盖率 (日期已排序, 用二分定位取代全量布尔扫描)
            dts, amt, mc = s
            i0 = np.searchsorted(dts, w0_i8, side="left")
            i1 = np.searchsorted(dts, t_i8, side="left")
            if i1 - i0 < min_cover:
                continue
            adv = float(np.nanmean(amt[i0:i1]))
            mcap = float(np.nanmedian(mc[i0:i1]))
            if not (adv >= a.min_adv) or not (mcap > 0):
                continue
            cand.append((code, mcap, adv))
        if not cand:
            continue
        d = pd.DataFrame(cand, columns=["code", "mcap", "adv"])
        d = d.sort_values(a.rank_by, ascending=False).head(a.top_n).reset_index(drop=True)
        d["rank"] = d.index + 1
        d["effective_date"] = T
        recs.append(d)
        print(f"  {T.date()}  候选合格 {len(cand):>4} -> 选中 {len(d)} | "
              f"市值门槛 {d['mcap'].min()/1e8:>7.1f}亿 | ADV门槛 {d['adv'].min()/1e8:.2f}亿")

    u = pd.concat(recs, ignore_index=True)
    u["has_kline"] = True
    u = u[["effective_date", "code", "mcap", "adv", "rank", "has_kline"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(out_path, index=False)
    print(f"\n已写入 {out_path}")
    print(f"  {len(u):,} 行 | {u['code'].nunique()} 只不重复股票 | "
          f"{u['effective_date'].nunique()} 个生效期")

    # 换手率: 相邻期成分变化
    print(f"\n=== 成分股换手 ===")
    prev = None
    for T, g in u.groupby("effective_date"):
        cur = set(g["code"])
        if prev is not None:
            add, drop = cur - prev, prev - cur
            print(f"  {pd.Timestamp(T).date()}  新进 {len(add):>3} | 剔除 {len(drop):>3} | "
                  f"保留 {len(cur & prev):>3} ({100*len(cur & prev)/len(cur):.0f}%)")
        prev = cur

    # 幸存者偏差残留
    print(f"\n=== 幸存者偏差残留检查 ===")
    win_del = meta[(meta["delist_date"] >= a.start) & (meta["delist_date"] <= a.end)]
    have = [c for c in win_del["code"] if c in store]
    print(f"  窗口内退市 {len(win_del)} 只, 其中本地有K线的仅 {len(have)} 只")
    print(f"  -> 这 {len(win_del)-len(have)} 只无法进入回测, 构成残留幸存者偏差")
    print(f"  -> 影响上界: 退市股占同期全市场约 {100*len(win_del)/len(meta):.1f}%")

    cfg_path.write_text(json.dumps({
        "top_n": a.top_n, "lookback": a.lookback,
        "min_listed_years": a.min_listed_years, "freq": a.freq,
        "rank_by": a.rank_by, "min_adv": a.min_adv,
        "start": a.start, "end": a.end,
        "rule": f"{a.rank_by}排序取前N; 已上市满N年; 未退市; 窗口数据覆盖>=75%; ADV>={a.min_adv:.0e}",
        "mcap_definition": "median(vwap x outstanding_share), vwap=amount/volume 为真实未复权均价",
        "known_limitations": [
            f"退市股缺K线 {len(win_del)-len(have)}/{len(win_del)} 只, 残留幸存者偏差",
            "历史ST状态不可得, 未做ST剔除(偏保守: ST股通常跑输)",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
