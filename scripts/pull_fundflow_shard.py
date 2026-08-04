"""资金流分片拉取 —— 把一次长时间的回灌切成 N 份, 在多台机器上并行跑

为什么要分片
────────────
新浪资金流接口按【出口 IP】限流, 单机把全池回灌到 2015 要 6~10 小时。三台
服务器 (eez040/041/042) 有三个不同的公网 IP (143.89.46.40/.41/.42, 已实测),
所以三机并行是真的 3 倍, 不是假并行。

分片方式用 code 的稳定哈希取模, 而不是按顺序切段:
  · 顺序切段会让"代码靠前的老股票"集中在一片, 各片历史长度差很多, 耗时不均
  · 哈希取模天然把长短历史打散, 三片耗时接近
  · 而且同一只股票永远落在同一片, 断点续跑不会换机器重复拉

输出
────
每片写自己的 parquet: data/raw/fund_flow_full/shard_{i}of{n}.parquet
最后用 --merge 在主机上合并进 fundflow_history.parquet (合并前会备份旧表)。
分片文件互不重叠, 所以任一片失败只需重跑那一片。

用法
────
    # 在 eez040 上 (第0片):
    python3 scripts/pull_fundflow_shard.py --shard 0 --of 3 --since 2015-01-01
    # eez041 第1片, eez042 第2片 ...
    # 全部跑完后在 eez041:
    python3 scripts/pull_fundflow_shard.py --merge --of 3
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import pull_fundflow_sina as P  # noqa: E402

FF_DIR = ROOT / "data" / "raw" / "fund_flow_full"
CONS = FF_DIR / "fundflow_history.parquet"


def shard_of(code: str, n: int) -> int:
    """稳定哈希分片。不用内置 hash() —— 它带随机种子, 换进程就变, 断点续跑会错片"""
    h = hashlib.md5(str(code).encode()).hexdigest()
    return int(h, 16) % n


def universe_codes(codes_file: str | None = None) -> list[str]:
    """要拉的股票池。

    默认复用 pull_fundflow_sina 的口径(读 data/universe/watchlist_pit.json),
    保证与主流程一致。但并行拉取时 eez040/042 上没有整个 data 目录, 而且
    2015 版股票池要等 K 线回灌完、重建 PIT 之后才知道 —— 所以支持用
    --codes-file 显式传一份代码清单(每行一个 6 位代码, # 开头为注释)。
    三台机器必须用【同一份清单】, 否则分片会不一致, 出现漏拉或重复。
    """
    if codes_file:
        p = Path(codes_file)
        if not p.exists():
            raise SystemExit(f"ERROR: 找不到代码清单 {p}")
        out = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                out.append(str(ln).zfill(6)[:6])
        if not out:
            raise SystemExit(f"ERROR: {p} 里没有任何代码")
        return sorted(set(out))
    return P.universe_codes()


def do_pull(args):
    codes = [c for c in universe_codes(args.codes_file)
             if shard_of(c, args.of) == args.shard]
    out = FF_DIR / f"shard_{args.shard}of{args.of}.parquet"
    FF_DIR.mkdir(parents=True, exist_ok=True)

    # 断点续跑: 已经拉过的代码直接跳过
    done = {}
    if out.exists() and not args.force:
        try:
            old = pd.read_parquet(out)
            for c, g in old.groupby("code"):
                done[str(c)] = g
            print(f"续跑: 本片已有 {len(done)} 只, 跳过", flush=True)
        except Exception as e:
            print(f"旧分片读失败({type(e).__name__}), 从头拉", flush=True)

    since = pd.Timestamp(args.since)
    todo = [c for c in codes if c not in done]
    print(f"分片 {args.shard}/{args.of} | 本片 {len(codes)} 只 | 待拉 {len(todo)} 只 | "
          f"起点 {since:%F}", flush=True)

    rows = list(done.values())
    t0 = time.time()
    n_ok = n_empty = n_err = 0
    for i, code in enumerate(todo, 1):
        try:
            df = P.pull_history(code, since, args.sleep)
            if df is None or not len(df):
                n_empty += 1
            else:
                rows.append(P.cons_schema_row(code, df))
                n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 5:
                print(f"  [!] {code}: {type(e).__name__} {str(e)[:60]}", flush=True)
        # 每 25 只落一次盘: 这活要跑几小时, 中途断了不能全丢
        if i % 25 == 0 or i == len(todo):
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
            el = time.time() - t0
            rate = i / el if el else 0
            print(f"  [{i}/{len(todo)}] ok={n_ok} empty={n_empty} err={n_err} | "
                  f"{el/60:.1f}min | {rate*60:.1f}只/min | "
                  f"剩余~{(len(todo)-i)/rate/60 if rate else 0:.0f}min", flush=True)
    if rows:
        final = pd.concat(rows, ignore_index=True)
        final.to_parquet(out, index=False)
        dt = pd.to_datetime(final["date"])
        print(f"\n分片完成 -> {out}", flush=True)
        print(f"  {len(final):,} 行 | {final['code'].nunique()} 只 | "
              f"{dt.min():%F} ~ {dt.max():%F}", flush=True)
    else:
        print("\n本片没有任何数据", flush=True)


def do_merge(args):
    """把各分片合并进 consolidated 表。缺片就报错退出 —— 少一片等于池子少三分之一,
    这种残缺表一旦被特征引擎读进去, 那批股票的资金流特征会静默变 NaN。"""
    parts, missing = [], []
    for i in range(args.of):
        p = FF_DIR / f"shard_{i}of{args.of}.parquet"
        if not p.exists():
            missing.append(p.name)
            continue
        parts.append(pd.read_parquet(p))
    if missing:
        raise SystemExit(f"ERROR: 缺分片 {missing}, 合并会得到残缺的池子, 已拒绝。"
                         "\n  请先把缺的片跑完(可在任意机器上跑, 分片是按代码哈希定的)")
    new = pd.concat(parts, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    new["code"] = new["code"].astype(str).str.zfill(6)
    new = (new.sort_values(["code", "date"])
              .drop_duplicates(["code", "date"], keep="last")
              .reset_index(drop=True))

    if CONS.exists():
        bak = CONS.with_name(f"fundflow_history_premerge_{time.strftime('%Y%m%d_%H%M%S')}.parquet")
        CONS.replace(bak)
        print(f"旧表已备份: {bak.name}")
    new.to_parquet(CONS, index=False)
    dt = new["date"]
    print(f"已写入 {CONS}")
    print(f"  {len(new):,} 行 | {new['code'].nunique()} 只 | {dt.min():%F} ~ {dt.max():%F}")
    by_year = new.groupby(dt.dt.year)["code"].nunique()
    print("  各年覆盖股票数:")
    for y, n in by_year.items():
        print(f"    {y}: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0, help="本机负责第几片 (0 起)")
    ap.add_argument("--of", type=int, default=3, help="总共分几片")
    ap.add_argument("--since", default="2015-01-01", help="回溯起点")
    ap.add_argument("--sleep", type=float, default=1.2, help="每次请求间隔秒")
    ap.add_argument("--force", action="store_true", help="忽略已有分片, 从头拉")
    ap.add_argument("--codes-file", default=None,
                    help="显式的股票代码清单(每行一个6位代码)。三台机器必须用同一份, "
                         "否则分片不一致会漏拉或重复。不传则读 watchlist_pit.json")
    ap.add_argument("--merge", action="store_true", help="合并各分片进 consolidated 表")
    args = ap.parse_args()
    if args.of < 1 or not (0 <= args.shard < args.of):
        raise SystemExit("ERROR: 需满足 0 <= shard < of")
    if args.merge:
        do_merge(args)
    else:
        do_pull(args)


if __name__ == "__main__":
    main()
