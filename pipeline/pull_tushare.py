"""Tushare 历史回填 + 每日增量

只拉当前 token 已确认可用的接口(见 pipeline/tushare_probe.py 的探测结果):
  daily_basic  市值/估值/换手  <- 训练集完全空缺的横截面维度
  adj_factor   复权因子        <- 修 akshare 前复权价随分红重算的漂移
  index_daily  真实指数基准    <- 替掉自建等权池
  stock_basic  行业(粗分类)    <- 440 个特征里零个行业信息

存储: data/raw/tushare/{接口}/{年}.parquet, 按年分片。
  按年分片而不是按股票: 这些接口都是"一次调用拿全市场一天", 按日期循环最省
  配额; 落盘也按日期维度切, 增量时只需重写当年那一片。

跑法:
  python pipeline/pull_tushare.py --probe                    # 只看缺哪些日期
  python pipeline/pull_tushare.py daily_basic                # 回填(自动续)
  python pipeline/pull_tushare.py daily_basic --start 20240101
  python pipeline/pull_tushare.py all                        # 四个接口都拉
  python pipeline/pull_tushare.py index_daily                # 指数不按日循环

中断可直接重跑 —— 已落盘的日期会跳过。
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import settings  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

log = get_logger("pull_tushare")

# 训练集从 2019-01-02 起, 回填对齐它
DEFAULT_START = "20190101"

# 基准指数: 沪深300(大盘) / 中证500(中盘) / 中证1000(小盘) / 上证综指
INDEXES = ["000300.SH", "000905.SH", "000852.SH", "000001.SH", "399006.SZ"]

# 2000 积分档是 200 次/分钟。留出余量按 0.35s 一次, 约 170 次/分钟。
# 免费档是 50 次/分钟, 用 --sleep 2 覆盖。
SLEEP = 0.35


def out_dir(name):
    d = settings.TUSHARE_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_existing(name):
    """已落盘的数据 —— 用来算"还缺哪些日期", 断点续传靠它"""
    files = sorted(out_dir(name).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def save_by_year(name, df, date_col="trade_date"):
    """按年合并写回。同一年重复跑不会累积重复行 —— 按主键去重后覆盖。"""
    if df.empty:
        return 0
    df[date_col] = df[date_col].astype(str)
    df["_y"] = df[date_col].str[:4]
    n = 0
    for y, part in df.groupby("_y"):
        p = out_dir(name) / f"{y}.parquet"
        part = part.drop(columns=["_y"])
        if p.exists():
            old = pd.read_parquet(p)
            part = pd.concat([old, part], ignore_index=True)
        keys = [c for c in ("ts_code", date_col) if c in part.columns]
        part = part.drop_duplicates(subset=keys, keep="last")
        part = part.sort_values(keys).reset_index(drop=True)
        part.to_parquet(p, index=False)
        n += len(part)
    return n


# 日历缓存路径。trade_cal 在免费档是 **1 次/小时**, 每次跑都去拉会直接
# 把整个回填卡死。日历是几乎不变的数据, 拉一次存下来就行。
CAL_PATH = lambda: settings.TUSHARE_DIR / "trade_cal.parquet"  # noqa: E731

# 兑底用的基准股: 几乎不停牌的大盘股, 多取几只取并集避免个别停牌日漏掉
CAL_FALLBACK_CODES = ["600519", "000001", "601398", "600036"]


def _cal_from_local_kline(start, end):
    """从本地 K 线推交易日 —— trade_cal 频率用完时的兑底

    没有日历就什么都干不了, 而本地本来就存了 7 年全市场日线,
    取几只不停牌的大盘股日期并集, 与交易日历实际上一致。
    """
    days = set()
    for c in CAL_FALLBACK_CODES:
        p = settings.KLINE_DIR / f"{c}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["date"])
        days |= set(pd.to_datetime(d["date"]).dt.strftime("%Y%m%d"))
    out = sorted(d for d in days if start <= d <= end)
    if out:
        log.warning("trade_cal 不可用, 已用本地 K 线推出 %d 个交易日", len(out))
    return out


def trade_days(pro, start, end):
    """交易日列表: 本地缓存 -> 接口 -> 本地 K 线兑底"""
    p = CAL_PATH()
    if p.exists():
        cal = pd.read_parquet(p)
        cal["cal_date"] = cal["cal_date"].astype(str)
        got = sorted(cal.loc[cal["is_open"] == 1, "cal_date"])
        # 缓存覆盖不到请求区间尾部时(比如过了几个月), 才重新拉
        if got and got[-1] >= end[:8]:
            return [d for d in got if start <= d <= end]
        log.info("日历缓存只到 %s, 尝试刷新", got[-1] if got else "空")
    try:
        cal = pro.trade_cal(exchange="SSE", start_date="20150101",
                            end_date="20301231")
        cal.to_parquet(p, index=False)
        log.info("交易日历已缓存 -> %s (%d 行)", p, len(cal))
        cal["cal_date"] = cal["cal_date"].astype(str)
        got = sorted(cal.loc[cal["is_open"] == 1, "cal_date"])
        return [d for d in got if start <= d <= end]
    except Exception as e:
        log.warning("trade_cal 不可用(%s)", str(e)[:80])
        return _cal_from_local_kline(start, end)


def pull_by_date(pro, name, start, end, sleep, fields=None, date_kw="trade_date",
                 refresh_days=0):
    """按日期循环的接口: 一次调用拿全市场一天

    date_kw: 大多数是 trade_date, 业绩预告/快报是 ann_date(公告日)。

    refresh_days: 强制重拉最近 N 个交易日, 即使本地已有。
        必要性: 有些接口是延迟发布的 —— 实测 margin_detail 在当日傍晚只有 1994 只,
        次日才补齐到 4422 只。而"一个日期拉过就永不再拉"会把这份残缺永久冻结进去,
        且不报错。重拉是安全的: save_by_year 按 (ts_code, date) 去重且 keep="last",
        新数据覆盖旧数据、只多不少。
        默认 0 是给全量回填用的(几千天, 不该多拉); 日更应当传 5 左右。
    """
    days = trade_days(pro, start, end)
    have = load_existing(name)
    done = set(have[date_kw].astype(str)) if date_kw in have.columns else set()
    if refresh_days > 0:
        done -= set(days[-refresh_days:])
    todo = [d for d in days if d not in done]
    log.info("%s: 区间内 %d 个交易日, 已有 %d, 待拉 %d",
             name, len(days), len(days) - len(todo), len(todo))
    if not todo:
        return

    buf, pulled, failed = [], 0, []
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        kw = {date_kw: d}
        if fields:
            kw["fields"] = fields
        df = None
        for attempt in range(4):
            try:
                df = getattr(pro, name)(**kw)
                break
            except Exception as e:
                msg = str(e)
                if "权限" in msg or "积分" in msg:
                    log.error("%s 无权限, 中止: %s", name, msg[:120])
                    return
                # 触发限频时官方建议直接等 —— 指数退避, 不要硬重试
                wait = 15 * (attempt + 1) if "每分钟" in msg or "频率" in msg else 3
                if attempt < 3:
                    time.sleep(wait)
                else:
                    failed.append(d)
                    log.warning("%s %s 失败: %s", name, d, msg[:90])
        if df is not None and len(df):
            buf.append(df)
            pulled += len(df)
        # 分批落盘: 全拉完再写, 中途断了就全白跑
        if len(buf) >= 60:
            save_by_year(name, pd.concat(buf, ignore_index=True), date_col=date_kw)
            buf = []
        if i % 120 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i) / 60
            log.info("  %s %d/%d 累计 %d 行, 已用 %.0f 分, 预计还需 %.0f 分",
                     name, i, len(todo), pulled, el / 60, eta)
        time.sleep(sleep)

    if buf:
        save_by_year(name, pd.concat(buf, ignore_index=True), date_col=date_kw)
    log.info("%s 完成: 新增 %d 行%s", name, pulled,
             f", {len(failed)} 天失败(重跑本命令会补)" if failed else "")


def pull_index(pro, start, end, sleep):
    """指数行情按 ts_code 循环, 一次给一只的全区间 —— 比按日循环省几百倍配额"""
    rows = []
    for code in INDEXES:
        try:
            df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
            if df is not None and len(df):
                rows.append(df)
                log.info("  %s: %d 行 (%s ~ %s)", code, len(df),
                         df["trade_date"].min(), df["trade_date"].max())
        except Exception as e:
            log.warning("%s 失败: %s", code, str(e)[:100])
        time.sleep(sleep)
    if rows:
        n = save_by_year("index_daily", pd.concat(rows, ignore_index=True))
        log.info("index_daily 完成: 落盘 %d 行", n)


def pull_stock_basic(pro):
    """行业分类。注意: 这是【当前快照】, 不是 PIT ——

    行业归属偶尔会变(重组/主营变更), 用它做历史特征严格来说有轻微前视。
    申万历史成分(index_member_all)才是 PIT 的, 但当前 token 没权限。
    先用它把"有没有行业信息"这一步跨过去, 效果确认了再决定要不要买。
    """
    frames = []
    for st in ("L", "D", "P"):  # 上市/退市/暂停 —— 只取 L 会有生存者偏差
        try:
            df = pro.stock_basic(exchange="", list_status=st,
                                 fields="ts_code,symbol,name,area,industry,market,"
                                        "list_date,delist_date")
            if df is not None and len(df):
                df["list_status"] = st
                frames.append(df)
                log.info("  list_status=%s: %d 只", st, len(df))
        except Exception as e:
            log.warning("stock_basic %s 失败: %s", st, str(e)[:100])
        time.sleep(0.35)
    if frames:
        out = out_dir("stock_basic") / "stock_basic.parquet"
        allc = pd.concat(frames, ignore_index=True)
        allc.to_parquet(out, index=False)
        log.info("stock_basic 完成: %d 只, %d 个行业 -> %s",
                 len(allc), allc["industry"].nunique(), out)


def pull_sw_member(pro, sleep):
    """申万行业成分 —— 带 in_date/out_date, 这才是 PIT 行业

    stock_basic 的 industry 是当前快照, 用它做 2021 年的行业特征会引入前视
    (重组/主营变更过的股会被归到它今天的行业)。这个接口给出每段归属
    的生效/失效日期, 才能按交易日回溯"当时属于哪个行业"。

    单次最大 2000 行, 全市场 5000+ 股×多段历史 远超上限, 所以按一级行业循环。
    """
    try:
        cls = pro.index_classify(level="L1", src="SW2021")
    except Exception as e:
        log.error("index_classify 失败: %s", str(e)[:120])
        return
    cls.to_parquet(out_dir("sw_member") / "index_classify_L1.parquet", index=False)
    log.info("申万一级行业 %d 个", len(cls))

    frames = []
    for i, row in enumerate(cls.itertuples(), 1):
        l1 = getattr(row, "index_code", None) or getattr(row, "industry_code")
        # is_new="N" 才会把历史上进出过的成分都给出来
        for is_new in ("Y", "N"):
            try:
                df = pro.index_member_all(l1_code=l1, is_new=is_new)
                if df is not None and len(df):
                    df["is_new"] = is_new
                    frames.append(df)
            except Exception as e:
                log.warning("index_member_all %s is_new=%s: %s", l1, is_new,
                            str(e)[:90])
            time.sleep(sleep)
        if i % 10 == 0:
            log.info("  行业 %d/%d, 累计 %d 块", i, len(cls), len(frames))

    if frames:
        allm = pd.concat(frames, ignore_index=True)
        allm = allm.drop_duplicates(subset=[c for c in ("ts_code", "l3_code", "in_date")
                                            if c in allm.columns])
        p = out_dir("sw_member") / "sw_member.parquet"
        allm.to_parquet(p, index=False)
        log.info("sw_member 完成: %d 行, %d 只股 -> %s",
                 len(allm), allm["ts_code"].nunique(), p)


def universe_codes():
    """要拉财务的股票范围: PIT 股票池出现过的全部代码

    财务接口非 VIP 版只能按 ts_code 拉, 5537 只全拉要 5537 次调用。只拉股票池
    里出现过的(含已过期的历史成分, 不是只取最新 —— 否则引入生存者偏差)。
    """
    p = settings.UNIVERSE_DIR / "universe_pit_2019.parquet"
    u = pd.read_parquet(p)
    codes = sorted(u["code"].astype(str).str[:6].unique())
    out = []
    for c in codes:
        # tushare 要 ts_code 带交易所后缀
        sfx = "SH" if c.startswith(("6", "9")) else ("BJ" if c.startswith(("4", "8")) else "SZ")
        out.append(f"{c}.{sfx}")
    return out


def pull_by_code(pro, name, sleep, start, end):
    """按股票循环的接口(财务类): 一次拿一只的全区间

    这类接口带真实 ann_date(公告日), 比我们现在用"报告期+规则偏移"推出来的
    发布日准 —— 实际公告日在报告期后 1~4 个月不等, 规则偏移会系统性泄露或滞后。
    """
    codes = universe_codes()
    have = load_existing(name)
    done = set(have["ts_code"].astype(str)) if "ts_code" in have.columns else set()
    todo = [c for c in codes if c not in done]
    log.info("%s: 股票池 %d 只, 已有 %d, 待拉 %d", name, len(codes),
             len(codes) - len(todo), len(todo))
    if not todo:
        return
    buf, pulled = [], 0
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        for attempt in range(4):
            try:
                df = pro.query(name, ts_code=c, start_date=start, end_date=end)
                if df is not None and len(df):
                    buf.append(df)
                    pulled += len(df)
                break
            except Exception as e:
                msg = str(e)
                if "权限" in msg or "积分" in msg:
                    log.error("%s 无权限, 中止: %s", name, msg[:120])
                    return
                time.sleep(15 * (attempt + 1) if "频" in msg else 3)
        if len(buf) >= 200:
            save_by_year(name, pd.concat(buf, ignore_index=True), date_col="ann_date")
            buf = []
        if i % 200 == 0 or i == len(todo):
            el = time.time() - t0
            log.info("  %s %d/%d 累计 %d 行, 已用 %.0f 分, 预计还需 %.0f 分",
                     name, i, len(todo), pulled, el / 60,
                     el / i * (len(todo) - i) / 60)
        time.sleep(sleep)
    if buf:
        save_by_year(name, pd.concat(buf, ignore_index=True), date_col="ann_date")
    log.info("%s 完成: %d 行", name, pulled)


# 按日期循环的接口表: name -> (date_kw, fields, 说明)
BY_DATE = {
    "daily_basic": ("trade_date",
                    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
                    "volume_ratio,pe,pe_ttm,pb,ps_ttm,dv_ratio,dv_ttm,total_share,"
                    "float_share,free_share,total_mv,circ_mv",
                    "市值/估值/换手 —— 训练集完全空缺的横截面维度"),
    "adj_factor": ("trade_date", None, "复权因子"),
    "daily": ("trade_date", None, "未复权日线(配 adj_factor 做后复权)"),
    "stk_limit": ("trade_date", None, "涨跌停价 —— 现在是从 K 线猜的"),
    "suspend_d": ("trade_date", None, "停牌"),
    "moneyflow": ("trade_date", None, "资金流(分单等级)"),
    "margin_detail": ("trade_date", None, "融资融券明细"),
    "top_list": ("trade_date", None, "龙虎榜"),
    "top_inst": ("trade_date", None, "龙虎榜机构席位明细"),
    "hk_hold": ("trade_date", None, "北向(沪深股通)个股持股"),
    "block_trade": ("trade_date", None, "大宗交易"),
    "share_float": ("ann_date", None, "限售解禁(按公告日, 天然PIT)"),
    "index_dailybasic": ("trade_date", None, "指数估值"),
    "moneyflow_hsgt": ("trade_date", None, "北向资金"),
    "forecast": ("ann_date", None, "业绩预告(按公告日)"),
    "express": ("ann_date", None, "业绩快报(按公告日)"),
}

# 按股票循环的接口
BY_CODE = ["fina_indicator", "income", "balancesheet", "cashflow",
           "stk_holdernumber"]

# 全量回填顺序: 越靠前越重要 —— 万一半夜断了, 先保住最有价值的
ALL_ORDER = ["stock_basic", "sw_member", "index_daily", "daily_basic",
             "adj_factor", "daily", "stk_limit", "suspend_d",
             "fina_indicator", "forecast", "express", "moneyflow",
             "margin_detail", "moneyflow_hsgt", "index_dailybasic", "top_list",
             "top_inst", "hk_hold", "block_trade", "share_float",
             "stk_holdernumber"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="probe",
                    choices=(["probe", "all", "index_daily", "stock_basic",
                              "sw_member"] + list(BY_DATE) + BY_CODE))
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=pd.Timestamp.now().strftime("%Y%m%d"))
    ap.add_argument("--sleep", type=float, default=SLEEP,
                    help="每次调用间隔秒数; 免费档 50次/分 请用 2")
    ap.add_argument("--refresh-days", type=int, default=0,
                    help="强制重拉最近 N 个交易日(即使本地已有)。用于延迟发布的接口: "
                         "margin_detail 当日只有约 1994 只, 次日才补齐到 4422 只, "
                         "不重拉就会把残缺永久冻结。日更建议 5, 全量回填保持 0")
    args = ap.parse_args()

    if not settings.TUSHARE_TOKEN:
        raise SystemExit("ERROR: .env 里没有 TUSHARE_TOKEN")
    import tushare as ts
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    if args.what == "probe":
        days = trade_days(pro, args.start, args.end)
        print(f"区间 {args.start}~{args.end} 共 {len(days)} 个交易日\n")
        for name in ("daily_basic", "adj_factor", "index_daily", "daily"):
            have = load_existing(name)
            if have.empty:
                print(f"  {name:<12} 无数据, 待拉 {len(days)} 天")
                continue
            done = set(have["trade_date"].astype(str))
            miss = [d for d in days if d not in done]
            print(f"  {name:<12} 已有 {len(have):>9,} 行 / {len(done)} 天, "
                  f"待拉 {len(miss)} 天")
        sb = out_dir("stock_basic") / "stock_basic.parquet"
        print(f"  {'stock_basic':<12} {'已有' if sb.exists() else '无数据'}")
        return

    todo = ALL_ORDER if args.what == "all" else [args.what]
    for name in todo:
        log.info("─" * 60)
        try:
            if name == "stock_basic":
                pull_stock_basic(pro)
            elif name == "sw_member":
                pull_sw_member(pro, args.sleep)
            elif name == "index_daily":
                pull_index(pro, args.start, args.end, args.sleep)
            elif name in BY_CODE:
                pull_by_code(pro, name, args.sleep, args.start, args.end)
            else:
                date_kw, fields, _ = BY_DATE[name]
                pull_by_date(pro, name, args.start, args.end, args.sleep,
                             fields=fields, date_kw=date_kw,
                             refresh_days=args.refresh_days)
        except Exception as e:
            # 一个接口挂了不能拖死整夜的回填, 记下来继续下一个
            log.error("%s 异常中止: %s", name, str(e)[:150])


if __name__ == "__main__":
    main()
