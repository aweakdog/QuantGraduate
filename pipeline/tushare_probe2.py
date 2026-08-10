"""Tushare 第二轮探测: 迁移可行性

第一轮(tushare_probe.py)确认了 20 个核心接口可用。这一轮探的是
"如果以后数据源都收敛到 Tushare, 现有的 440 个特征还剩哪些拿不到"。

重点看三块现在【不是】来自 Tushare 的东西:
  宏观/海外  205 个特征(cn_ 121 + 美股指数 75 + 汇率 9), 占了将近一半
  概念板块   con_ 48 个特征, 现在来自同花顺
  事件公告   ev_/tev_/ann_ 44 个特征, 现在来自 ifind + 东财

能覆盖就能迁移, 不能覆盖的就得保留原采集链路。

    python pipeline/tushare_probe2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import settings  # noqa: E402

# (接口名, 参数, 说明, 对应我们现有的哪组特征)
TESTS = [
    # ── 概念/板块: 现在 con_ 48 个特征来自同花顺 ──
    ("ths_index", dict(exchange="A", type="N"), "同花顺概念板块列表", "con_ 48"),
    ("ths_member", dict(ts_code="885800.TI"), "概念板块成分股", "con_ 48"),
    ("ths_daily", dict(ts_code="885800.TI", start_date="20260701",
                       end_date="20260731"), "概念板块日行情", "con_ 48"),
    ("concept", dict(), "概念分类(旧版)", "con_"),
    ("concept_detail", dict(id="TS2"), "概念成分(旧版)", "con_"),

    # ── 宏观: 现在 cn_ 121 个特征来自自建 macro 目录 ──
    ("cn_gdp", dict(), "GDP", "cn_"),
    ("cn_cpi", dict(), "CPI", "cn_"),
    ("cn_ppi", dict(), "PPI", "cn_"),
    ("cn_pmi", dict(), "PMI", "cn_pmi"),
    ("cn_m", dict(), "货币供应 M0/M1/M2", "cn_"),
    ("shibor", dict(), "Shibor 利率", "cn_"),
    ("us_tycr", dict(), "美债收益率曲线", "cn_/海外"),

    # ── 海外/汇率: sp/dj/nq/sox 75 个 + usd* 9 个 ──
    ("index_global", dict(trade_date="20260731"), "国际指数(道指/纳指/标普)",
     "sp_/dj_/nq_/sox_ 75"),
    ("fx_daily", dict(trade_date="20260731"), "外汇日行情", "usdind/usdcnh/usdjpy 9"),

    # ── 我们完全没有的新维度 ──
    ("hk_hold", dict(trade_date="20260731"), "沪深港通持股明细", "无"),
    ("stk_holdernumber", dict(ts_code="600519.SH"), "股东户数", "无"),
    ("stk_holdertrade", dict(ts_code="600519.SH"), "股东增减持", "无"),
    ("block_trade", dict(trade_date="20260731"), "大宗交易", "无"),
    ("repurchase", dict(ann_date="20260731"), "股份回购", "无"),
    ("share_float", dict(ann_date="20260731"), "限售股解禁", "无"),
    ("cyq_perf", dict(ts_code="600519.SH"), "每日筹码分布", "无"),
    ("stk_factor", dict(ts_code="600519.SH", start_date="20260701",
                        end_date="20260731"), "技术因子(MACD/KDJ/RSI等现成)",
     "macd/rsi/atr 等 24"),
    ("stk_surv", dict(ts_code="600519.SH"), "机构调研", "无"),
    ("broker_recommend", dict(month="202607"), "券商月度金股", "无"),

    # ── 公告: 现在 announcements 1625 个文件来自东财 ──
    ("anns_d", dict(ann_date="20260731"), "公告全文(需单独权限)", "ann_ 3"),
]


def main():
    if not settings.TUSHARE_TOKEN:
        raise SystemExit("ERROR: .env 里没有 TUSHARE_TOKEN")
    import tushare as ts
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    ok, no = [], []
    print(f"{'接口':<20}{'行数':>7}  {'说明':<24}对应现有特征")
    print("-" * 84)
    for name, kw, desc, maps in TESTS:
        try:
            df = pro.query(name, **kw)
            n = len(df) if df is not None else 0
            cols = list(df.columns)[:6] if n else []
            ok.append((name, n, desc, maps, cols))
            print(f"[可用] {name:<14}{n:>7}  {desc:<24}{maps}")
            if cols:
                print(f"       列: {','.join(cols)}")
        except Exception as e:
            msg = str(e)
            no.append((name, desc, maps, msg))
            short = "无权限" if ("权限" in msg or "积分" in msg) else msg[:50]
            print(f"[不可] {name:<14}{'':>7}  {desc:<24}{maps}  | {short}")

    print("\n" + "=" * 84)
    print(f"可用 {len(ok)} / 不可用 {len(no)}")
    if no:
        print("\n拿不到的(这些维度得保留现有采集链路):")
        for name, desc, maps, _ in no:
            print(f"  {name:<18}{desc:<26}影响特征: {maps}")


if __name__ == "__main__":
    main()
