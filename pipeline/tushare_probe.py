"""探测 Tushare 账号实际权限 —— 拉数据之前先跑这个

为什么需要它: Tushare 的权限是按积分卡的, 而"某接口要多少积分"散落在几十个
文档页里, 且会变。与其照文档猜, 不如拿真 token 每个接口试一条数据 —— 报错
信息里会直接写"抱歉，您没有访问该接口的权限，需要XXXX积分"。

跑法:
    python pipeline/tushare_probe.py

输出: 每个接口 可用 / 缺积分(附所需分数) / 其他错误, 最后给出建议。
不写任何文件, 不消耗多少配额(每个接口只取 1~2 行)。
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import settings  # noqa: E402

# 按"对当前问题的价值"排序 —— 前 4 个是补 A 窗负 IC 缺口的关键:
#   行业分类 / 市值估值换手 / 真实指数基准 / 复权因子
# (interface, kwargs, 说明)
PROBES = [
    ("stock_basic", {"exchange": "", "list_status": "L",
                     "fields": "ts_code,name,industry,market,list_date"},
     "股票列表+行业(粗) — 免费档通常可用"),
    ("trade_cal", {"exchange": "SSE", "start_date": "20200101",
                   "end_date": "20200110"},
     "交易日历 — 回填循环要用它, 不能靠猜"),
    ("daily", {"trade_date": "20220815"},
     "未复权日线 — 120 积分免费档唯一能拿的行情"),
    ("adj_factor", {"trade_date": "20220815"},
     "复权因子 — 修我们前复权价会随分红漂移的问题"),
    ("daily_basic", {"trade_date": "20220815",
                     "fields": "ts_code,turnover_rate,volume_ratio,pe_ttm,pb,total_mv,circ_mv"},
     "★市值/估值/换手 — 我们目前完全空缺的横截面因子"),
    ("index_member_all", {"ts_code": "600519.SH"},
     "★申万行业成分 — 我们 440 个特征里零个行业信息"),
    ("index_classify", {"level": "L1", "src": "SW2021"},
     "申万行业分类表"),
    ("index_daily", {"ts_code": "000300.SH", "start_date": "20220801",
                     "end_date": "20220815"},
     "★真实指数基准(沪深300) — 替掉自建等权池"),
    ("index_dailybasic", {"trade_date": "20220815"},
     "指数每日指标(PE/PB等)"),
    ("suspend_d", {"trade_date": "20220815"}, "停牌 — 可交易性判定"),
    ("stk_limit", {"trade_date": "20220815"}, "每日涨跌停价 — 现在是从K线猜的"),
    ("namechange", {"ts_code": "000001.SZ"}, "改名记录 — 识别 ST"),
    ("fina_indicator", {"ts_code": "600519.SH", "start_date": "20200101",
                        "end_date": "20220831"},
     "财务指标(带真实公告日) — 比我们规则化推的发布日准"),
    ("forecast", {"ann_date": "20220815"}, "业绩预告 — A股很强的事件信号"),
    ("express", {"ann_date": "20220815"}, "业绩快报"),
    ("moneyflow", {"trade_date": "20220815"}, "个股资金流(已有东财版, 可交叉验证)"),
    ("moneyflow_hsgt", {"trade_date": "20220815"}, "北向资金"),
    ("margin_detail", {"trade_date": "20220815"}, "融资融券明细"),
    ("top_list", {"trade_date": "20220815"}, "龙虎榜"),
    ("dividend", {"ts_code": "600519.SH"}, "分红送转"),
]


def main():
    token = settings.TUSHARE_TOKEN
    if not token:
        print("ERROR: 没读到 TUSHARE_TOKEN\n")
        print("在仓库根目录的 .env 里加一行(注意不要有引号和空格):")
        print("  TUSHARE_TOKEN=你的token")
        print("\ntoken 在 https://tushare.pro/user/token 页面复制")
        raise SystemExit(1)

    try:
        import tushare as ts
    except ImportError:
        print("ERROR: 没装 tushare\n  .venv/bin/pip install tushare")
        raise SystemExit(1)

    print(f"token 已读到 (末4位 ...{token[-4:]}), tushare {ts.__version__}\n")
    pro = ts.pro_api(token)

    ok, need, other = [], [], []
    for name, kw, desc in PROBES:
        try:
            df = getattr(pro, name)(**kw)
            n = 0 if df is None else len(df)
            cols = "" if df is None or df.empty else " | " + ",".join(list(df.columns)[:8])
            print(f"  [可用]  {name:<18} {n:>5} 行  {desc}{cols}")
            ok.append(name)
        except Exception as e:
            msg = str(e).replace("\n", " ")
            m = re.search(r"(\d+)\s*积分", msg)
            if "积分" in msg or "权限" in msg:
                pts = m.group(1) if m else "?"
                print(f"  [缺分]  {name:<18} 需 {pts} 积分  {desc}")
                need.append((name, pts, desc))
            else:
                print(f"  [报错]  {name:<18} {msg[:70]}  {desc}")
                other.append((name, msg))

    print(f"\n{'='*72}")
    print(f"可用 {len(ok)} 个 / 缺积分 {len(need)} 个 / 其他错误 {len(other)} 个")
    print("\n注意: 免费档(120积分)每个接口限 **1 次/小时**, 所以上面的[可用]只代表")
    print("      '有权限', 不代表能回填。daily_basic 要按交易日循环 1841 次,")
    print("      1次/小时 = 77 天才拉完。2000 积分档是 200 次/分钟 -> 约 11 分钟。")
    print("      跑完这个探测就已经用掉了每个接口当前这一小时的额度。")
    if need:
        pts = [int(p) for _, p, _ in need if p.isdigit()]
        print(f"\n要解锁这些接口, 至少需要 {max(pts) if pts else '?'} 积分:")
        for n, p, d in need:
            print(f"  {n:<18} 需 {p:>5} 积分   {d}")
        print("\n官网口径: 2000 积分 = 200 元/年, 5000 积分 = 500 元/年")
    if other:
        print("\n非权限类错误(可能是参数或网络, 值得单独看):")
        for n, m in other:
            print(f"  {n}: {m[:110]}")


if __name__ == "__main__":
    main()
