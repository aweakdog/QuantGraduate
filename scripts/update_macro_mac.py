"""Mac 可用的宏观数据增量更新

设计要点 — 口径校验后再写入:
  每个序列先在【重叠期】比对新源与本地历史, 偏差超阈值则拒绝写入,
  避免把不同口径的数据拼到一条序列上造成假的水平跳变。

已验证同源 (偏差 ~0):
  全球半导体SOX          <- index_us_stock_sina('.SOX')          0.0025%
  中国大宗商品价格指数    <- macro_china_commodity_price_index    0.0000%
  CN2Y/CN5Y/US2Y/US5Y   <- bond_zh_us_rate                      0.00bp
  us_*                  <- stock_us_daily

无可用同源 (跳过, 属实盘缺口):
  USDJPY / USDCNH / USDIND  最佳代理相关性仅 0.47~0.60
  A50期货                    上证50 代理相关性 0.887

用法:
  python scripts/update_macro_mac.py --dry-run    # 只校验不写入
  python scripts/update_macro_mac.py
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "data" / "raw" / "macro"

US_SYMBOLS = ["AAPL", "AMD", "AMZN", "GLD", "GOOGL", "IWM", "KWEB", "MSFT",
              "MU", "NVDA", "SLV", "SMH", "TLT", "TSLA", "TSM", "USO",
              "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLV", "XLY"]

TOL_PCT = 1.0      # level 模式: 重叠期平均水平偏差容忍上限 (%)
TOL_CORR = 0.99    # scale 模式: 重叠期日收益相关性下限
MIN_OVERLAP = 20   # 至少 N 天重叠才允许校验通过

# 校验模式:
#   level — 特征直接用原始水平值(汇率/国债收益率), 必须精确同源
#   scale — 特征只用 pct_change/z_21d, 对常数倍缩放免疫;
#           故校验日收益相关性, 再按接缝水平校准后拼接
#           (本地 iFinD qfq 与新浪 qfq 的分红处理不同, 存在常数量级差)


def bar(i, n, t0, tag=""):
    el = time.time() - t0
    rate = i / el if el else 0
    k = int(28 * i / n)
    print(f"\r  [{'#'*k}{'-'*(28-k)}] {i}/{n} {100*i/n:5.1f}% | {el:4.0f}s | "
          f"ETA {(n-i)/rate/60 if rate else 0:4.1f}m  {tag:22s}", end="", flush=True)


def load_local(name):
    p = MACRO / f"{name}.parquet"
    if not p.exists():
        return None, p
    d = pd.read_parquet(p)
    if "日期" not in d.columns:
        return None, p
    d = d.copy()
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d = d.dropna(subset=["日期"]).sort_values("日期")
    return d, p


def merge_write(name, new, dry, mode="level"):
    """new: DataFrame[日期, 最新值]. 校验重叠期口径后合并写入"""
    loc, p = load_local(name)
    new = new.dropna(subset=["日期", "最新值"]).copy()
    new["日期"] = pd.to_datetime(new["日期"])
    new["最新值"] = pd.to_numeric(new["最新值"], errors="coerce")
    new = new.dropna(subset=["最新值"]).sort_values("日期")

    if loc is None:
        return "本地缺失", 0, None

    old_max = loc["日期"].max()
    v_old = pd.to_numeric(loc["最新值"], errors="coerce")
    ov = pd.DataFrame({"日期": loc["日期"], "o": v_old}).merge(
        new.rename(columns={"最新值": "n"}), on="日期").dropna().sort_values("日期")
    if len(ov) < MIN_OVERLAP:
        return f"重叠仅{len(ov)}天", 0, old_max

    denom = ov["o"].abs().replace(0, pd.NA)
    dev = ((ov["n"] - ov["o"]).abs() / denom).mean() * 100
    scale = 1.0

    if mode == "level":
        if pd.isna(dev) or dev > TOL_PCT:
            return f"口径不符 偏差{dev:.2f}%", 0, old_max
        tag = f"偏差{dev:.3f}%"
    else:  # scale
        corr = ov["o"].pct_change().corr(ov["n"].pct_change())
        if pd.isna(corr) or corr < TOL_CORR:
            return f"收益不符 corr={corr:.4f}", 0, old_max
        # 接缝处水平校准: 用最后 5 个重叠日的中位比值, 抗单日噪声
        tail = ov.tail(5)
        scale = float((tail["o"] / tail["n"]).median())
        if not (0.2 < scale < 5):
            return f"校准因子异常 {scale:.3f}", 0, old_max
        tag = f"corr={corr:.4f} 缩放x{scale:.4f}"

    add = new[new["日期"] > old_max].copy()
    if not len(add):
        return f"无新增({tag})", 0, old_max
    add["最新值"] = add["最新值"] * scale

    if dry:
        return f"可增{len(add)}行 {tag}", len(add), add["日期"].max()

    # 保留本地原有其余列结构
    extra = [c for c in loc.columns if c not in ("日期", "最新值")]
    for c in extra:
        add[c] = pd.NA
    out = pd.concat([loc, add[loc.columns]], ignore_index=True)
    out = out.drop_duplicates("日期", keep="first").sort_values("日期").reset_index(drop=True)
    tmp = p.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(p)
    return f"+{len(add)}行 {tag}", len(add), add["日期"].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import akshare as ak

    tasks = []   # (显示名, 取数函数, 校验模式)

    tasks.append(("全球半导体SOX", lambda: ak.index_us_stock_sina(symbol=".SOX")
                  .rename(columns={"date": "日期", "close": "最新值"})[["日期", "最新值"]],
                  "scale"))

    tasks.append(("中国大宗商品价格指数", lambda: ak.macro_china_commodity_price_index()
                  [["日期", "最新值"]], "level"))

    _bond = {}

    def bond(col):
        def f():
            if "df" not in _bond:
                _bond["df"] = ak.bond_zh_us_rate(start_date="20210101")
            d = _bond["df"]
            return d[["日期", col]].rename(columns={col: "最新值"})
        return f

    for nm, col in [("CN2Y", "中国国债收益率2年"), ("CN5Y", "中国国债收益率5年"),
                    ("US2Y", "美国国债收益率2年"), ("US5Y", "美国国债收益率5年")]:
        tasks.append((nm, bond(col), "level"))

    # 本地 us_* 为前复权(iFinD), stock_us_daily 默认不复权 -> 拆股票会口径不符
    # 两者分红处理仍有差异 -> 用 scale 模式(收益率校验+接缝校准)
    for s in US_SYMBOLS:
        tasks.append((f"us_{s}", (lambda ss: lambda: ak.stock_us_daily(symbol=ss, adjust="qfq")
                                  .rename(columns={"date": "日期", "close": "最新值"})
                                  [["日期", "最新值"]])(s), "scale"))

    n = len(tasks)
    print(f"=== 宏观数据更新 (Mac) | {n} 个序列 | "
          f"{'DRY-RUN' if a.dry_run else '写入'} ===\n")
    t0 = time.time()
    results = []
    for i, (name, fn, mode) in enumerate(tasks, 1):
        try:
            new = fn()
            st, cnt, mx = merge_write(name, new, a.dry_run, mode)
        except Exception as e:
            st, cnt, mx = f"FAIL {type(e).__name__}", 0, None
        results.append((name, st, cnt, mx))
        bar(i, n, t0, name)
        time.sleep(0.3)
    print("\n")

    print("=" * 68)
    okc = [r for r in results if r[2] > 0]
    print(f"{'序列':24s} {'结果':32s} 最新")
    print("-" * 68)
    for name, st, cnt, mx in results:
        flag = "*" if cnt > 0 else " "
        print(f"{flag}{name:23s} {st:32s} {mx.date() if mx is not None else '-'}")
    print("-" * 68)
    print(f"更新 {len(okc)} 个序列, 共 {sum(r[2] for r in results)} 行")

    print("\n【已知缺口 — 无同源可用, 属实盘上线问题】")
    for nm, why in [("USDJPY", "最佳代理 FXY corr=0.60"),
                    ("USDCNH", "最佳代理 CYB corr=0.47 且已停更"),
                    ("USDIND", "最佳代理 UUP corr=0.87, 水平不可比"),
                    ("A50期货", "上证50 代理 corr=0.887")]:
        loc, _ = load_local(nm)
        lm = loc["日期"].max().date() if loc is not None else "-"
        print(f"  {nm:10s} 停留 {lm}   ({why})")


if __name__ == "__main__":
    main()
