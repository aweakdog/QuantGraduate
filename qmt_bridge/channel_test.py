"""下单通道测试 (用户 10-09 允许): 盘中按跌停价挂 100 股, 确认已报后立即撤单, 不会成交。

  .venv\\Scripts\\python.exe -m qmt_bridge.channel_test --i-understand
只在连续竞价时段 (9:30-11:30, 13:00-14:56) 运行; 跌停价必须低于现价 5% 以上才下单。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from .readonly_probe import DEFAULT_USERDATA, connect_with_timeout, mask
from .trading import STATUS_NAME, STOCK_BUY, TERMINAL, TradingTrader, place_limit


def in_continuous_session(now):
    t = now.strftime("%H:%M:%S")
    return "09:30:05" <= t <= "11:29:00" or "13:00:05" <= t <= "14:56:00"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-understand", action="store_true", help="确认这会产生一笔委托与撤单")
    ap.add_argument("--code", default="600000.SH")
    args = ap.parse_args(argv)
    if not args.i_understand:
        raise SystemExit("需要 --i-understand")
    if not in_continuous_session(datetime.now()):
        raise SystemExit("不在连续竞价时段, 不测试")

    from xtquant import xtdata
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount

    log = {"at": datetime.now().isoformat(timespec="seconds"), "code": args.code, "steps": []}
    d = xtdata.get_instrument_detail(args.code)
    tick = xtdata.get_full_tick([args.code]).get(args.code) or {}
    last, down, up = float(tick.get("lastPrice") or 0), float(d["DownStopPrice"]), float(d["UpStopPrice"])
    log["steps"].append({"last": last, "down_stop": down, "up_stop": up})
    if not (last > 0 and down < last * 0.95):
        raise SystemExit(f"跌停价离现价不够远 (last={last}, down={down}), 不测试")

    t = TradingTrader(XtQuantTrader(DEFAULT_USERDATA, int(time.time()) % 100000 + 400000))
    t.start()
    if connect_with_timeout(t, 20) != 0:
        raise SystemExit("连不上 miniQMT")
    infos = [i for i in (t.query_account_infos() or []) if getattr(i, "account_type", None) == 2]
    if not infos:
        raise SystemExit("资金账号不在线")
    acc = StockAccount(str(infos[0].account_id), "STOCK")
    t.subscribe(acc)
    oid = place_limit(t, acc, args.code, STOCK_BUY, 100, down, up, down, 2000.0, "q10w-channel-test")
    log["steps"].append({"placed": oid, "price": down, "volume": 100, "account": mask(infos[0].account_id)})

    def status():
        o = t.query_stock_order(acc, oid)
        return (getattr(o, "order_status", None), getattr(o, "traded_volume", None), getattr(o, "status_msg", "")) if o else (None, None, "")

    t0 = time.time()
    while time.time() - t0 < 10:
        st = status()
        if st[0] in (50, 55, 56) or st[0] in TERMINAL:
            break
        time.sleep(0.5)
    log["steps"].append({"before_cancel": st, "status_name": STATUS_NAME.get(st[0])})
    rc = t.cancel_order_stock(acc, oid)
    log["steps"].append({"cancel_rc": rc})
    t0 = time.time()
    while time.time() - t0 < 10:
        st = status()
        if st[0] in TERMINAL:
            break
        time.sleep(0.5)
    log["steps"].append({"final": st, "status_name": STATUS_NAME.get(st[0])})
    asset = t.query_stock_asset(acc)
    log["steps"].append({"cash_after": getattr(asset, "cash", None), "frozen_after": getattr(asset, "frozen_cash", None)})
    print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
    sys.stdout.flush()
    os._exit(0 if st[0] == 54 else 1)


if __name__ == "__main__":
    main()
