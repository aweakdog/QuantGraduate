"""探测 资金流/两融/事件公告/宏观 在 Mac 上可用的 akshare 接口

对每个候选接口报告: 是否可用 / 数据源主机 / 返回列 / 最新日期
东财(eastmoney)当前可能被限流, 优先找 交易所官方/新浪/腾讯 的源

重要: 被限流的东财接口会【无超时挂起】(requests 默认不设 timeout),
      所以用 signal.alarm 给每个探测加 20s 硬超时。
"""
import signal
import time
import warnings

warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd

TODAY = "20260727"
TIMEOUT = 20


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def probe(domain, name, host, fn):
    t0 = time.time()
    signal.alarm(TIMEOUT)
    try:
        df = fn()
        signal.alarm(0)
    except Timeout:
        signal.alarm(0)
        print(f"  [{domain}] {name:38s} {host:10s} 超时>{TIMEOUT}s (挂死/被封)")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"  [{domain}] {name:38s} {host:10s} FAIL {type(e).__name__}: {str(e)[:55]}")
        time.sleep(1)
        return
    try:
        if df is None or (hasattr(df, "__len__") and not len(df)):
            print(f"  [{domain}] {name:38s} {host:10s} 空结果")
            return
        n = len(df)
        cols = list(df.columns)[:9] if hasattr(df, "columns") else []
        latest = ""
        if hasattr(df, "columns"):
            dc = next((c for c in df.columns
                       if str(c) in ("日期", "date", "trade_date", "时间", "月份")), None)
            if dc is not None:
                try:
                    latest = f"  最新={pd.to_datetime(df[dc], errors='coerce').max().date()}"
                except Exception:
                    latest = f"  最新={df[dc].max()}"
        print(f"  [{domain}] {name:38s} {host:10s} OK {n:>6}行 {time.time()-t0:4.1f}s{latest}")
        print(f"        列: {cols}")
    except Exception as e:
        print(f"  [{domain}] {name:38s} {host:10s} 解析异常 {type(e).__name__}: {str(e)[:50]}")
    time.sleep(1)


print("=" * 78)
print("一、资金流  (需要: main_force_net, main_force_pct, dde_net)")
print("=" * 78)
probe("资金流", "stock_individual_fund_flow", "东财",
      lambda: ak.stock_individual_fund_flow(stock="000063", market="sz"))
probe("资金流", "stock_individual_fund_flow_rank", "东财",
      lambda: ak.stock_individual_fund_flow_rank(indicator="今日"))
probe("资金流", "stock_fund_flow_individual", "同花顺",
      lambda: ak.stock_fund_flow_individual(symbol="即时"))

print("\n" + "=" * 78)
print("二、两融  (需要: rzye/mtss_balance)")
print("=" * 78)
probe("两融", "stock_margin_detail_szse", "深交所",
      lambda: ak.stock_margin_detail_szse(date=TODAY))
probe("两融", "stock_margin_detail_sse", "上交所",
      lambda: ak.stock_margin_detail_sse(date=TODAY))
probe("两融", "stock_margin_szse", "深交所",
      lambda: ak.stock_margin_szse(date=TODAY))

print("\n" + "=" * 78)
print("三、事件/公告")
print("=" * 78)
probe("公告", "stock_notice_report", "东财",
      lambda: ak.stock_notice_report(symbol="全部", date="2026-07-27"))
probe("事件", "stock_news_em", "东财",
      lambda: ak.stock_news_em(symbol="000063"))

print("\n" + "=" * 78)
print("四、宏观")
print("=" * 78)
probe("宏观", "bond_zh_us_rate (中美国债收益率)", "东财",
      lambda: ak.bond_zh_us_rate(start_date="20260601"))
probe("宏观", "macro_china_pmi", "东财",
      lambda: ak.macro_china_pmi())
probe("宏观", "macro_usa_ism_pmi", "东财",
      lambda: ak.macro_usa_ism_pmi())
probe("宏观", "currency_boc_sina (USDCNY)", "新浪",
      lambda: ak.currency_boc_sina(symbol="美元", start_date="20260601", end_date="20260727"))
probe("宏观", "stock_us_daily NVDA", "新浪",
      lambda: ak.stock_us_daily(symbol="NVDA"))
probe("宏观", "index_us_stock_sina .IXIC", "新浪",
      lambda: ak.index_us_stock_sina(symbol=".IXIC"))
