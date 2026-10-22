"""现金管理实测 (用户 10-22 批准): 货币 ETF T+0 往返 + 1000 元 1 天国债逆回购。

  .venv\\Scripts\\python.exe -m qmt_bridge.cash_test --i-understand
  1) 买 100 股 511880 银华日利 (限价=卖一, 立即成交), 2) 立即按买一卖出同样 100 股 (验证 T+0 当天可卖),
  3) 卖出 10 张 204001 GC001 (逆回购 1000 元, 价格=年化利率, 取买一保证成交), 次日自动到账。
只在连续竞价时段跑; 账户必须没有策略持仓; 每步都查委托状态, 20 秒不成交就撤单并停止。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from .readonly_probe import ACCOUNT_STATUS_NAME, DEFAULT_USERDATA, account_status, connect_with_timeout, mask
from .trading import STATUS_NAME, STOCK_BUY, STOCK_SELL, TERMINAL, TradingTrader, is_cash_equiv, place_limit

ETF, REPO = "511880.SH", "204001.SH"


def in_session(now):
    t = now.strftime("%H:%M:%S")
    return "09:30:10" <= t <= "11:28:00" or "13:00:10" <= t <= "14:55:00"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-understand", action="store_true")
    a = ap.parse_args(argv)
    if not a.i_understand:
        raise SystemExit("需要 --i-understand")
    if not in_session(datetime.now()):
        raise SystemExit("不在连续竞价时段")
    from xtquant import xtdata
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount
    xtdata.enable_hello = False

    log = {"at": datetime.now().isoformat(timespec="seconds"), "steps": []}
    t = TradingTrader(XtQuantTrader(DEFAULT_USERDATA, int(time.time()) % 100000 + 600000))
    t.start()
    if connect_with_timeout(t, 20) != 0:
        raise SystemExit("连不上 miniQMT")
    infos = [i for i in (t.query_account_infos() or []) if getattr(i, "account_type", None) == 2]
    if not infos:
        raise SystemExit("资金账号不在线")
    acc = StockAccount(str(infos[0].account_id), "STOCK")
    t.subscribe(acc)
    state = account_status(t, infos[0].account_id)
    if state != 0:
        raise SystemExit(f"交易账户状态 {state}({ACCOUNT_STATUS_NAME.get(state, '未知')}), 下不了单; 先在 QMT 里重新登录交易账户")
    held = [p for p in (t.query_stock_positions(acc) or []) if int(getattr(p, "volume", 0) or 0) > 0
            and not is_cash_equiv(getattr(p, "stock_code", ""))]
    asset = t.query_stock_asset(acc)
    log["account"] = mask(infos[0].account_id)
    log["before"] = {"cash": asset.cash, "frozen": asset.frozen_cash, "strategy_positions": len(held)}
    if held or asset.cash < 12000:
        raise SystemExit(f"前置条件不满足: 策略持仓 {len(held)} 只, 现金 {asset.cash}")

    def quote(code):
        d = xtdata.get_instrument_detail(code)
        k = xtdata.get_full_tick([code]).get(code) or {}
        return (float((k.get("bidPrice") or [0])[0]), float((k.get("askPrice") or [0])[0]),
                float(d["UpStopPrice"]), float(d["DownStopPrice"]))

    def wait(oid, secs=20):
        t0, o = time.time(), None
        while time.time() - t0 < secs:
            o = t.query_stock_order(acc, oid)
            if o is not None and getattr(o, "order_status", None) in TERMINAL:
                break
            time.sleep(0.5)
        st = getattr(o, "order_status", None)
        if st not in TERMINAL:
            t.cancel_order_stock(acc, oid)
            time.sleep(1)
            o = t.query_stock_order(acc, oid)
            st = getattr(o, "order_status", None)
        return {"order_id": oid, "status": STATUS_NAME.get(st, st), "traded_volume": getattr(o, "traded_volume", None),
                "traded_price": getattr(o, "traded_price", None), "msg": getattr(o, "status_msg", "")}

    # 1) 买货币 ETF
    bid, ask, up, down = quote(ETF)
    oid = place_limit(t, acc, ETF, STOCK_BUY, 100, ask, up, down, 11000.0, "q10w-cash-test-B")
    r1 = wait(oid)
    log["steps"].append({"step": "buy 511880 x100", "limit": ask, **r1})
    if r1["status"] != "已成":
        print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
        os._exit(1)
    # 2) 立即卖出 (T+0)
    bid, ask, up, down = quote(ETF)
    oid = place_limit(t, acc, ETF, STOCK_SELL, 100, bid, up, down, 11000.0, "q10w-cash-test-S")
    r2 = wait(oid)
    log["steps"].append({"step": "sell 511880 x100 (same day, T+0)", "limit": bid, **r2})
    # 3) 逆回购 1000 元 (卖出 10 张, 价格=年化利率)
    rbid, rask, rup, rdown = quote(REPO)
    oid = place_limit(t, acc, REPO, STOCK_SELL, 10, rbid, rup, rdown, 1000.0, "q10w-cash-test-R")
    r3 = wait(oid)
    log["steps"].append({"step": "reverse repo GC001 x10 (1000 元, 1 天)", "rate_limit": rbid, "book": [rbid, rask], **r3})
    time.sleep(1)
    asset = t.query_stock_asset(acc)
    pos = [{"code": p.stock_code, "volume": p.volume, "market_value": getattr(p, "market_value", None)}
           for p in (t.query_stock_positions(acc) or []) if int(getattr(p, "volume", 0) or 0) > 0]
    trades = [{"code": x.stock_code, "type": x.order_type, "vol": x.traded_volume, "px": x.traded_price,
               "amt": getattr(x, "traded_amount", None)} for x in (t.query_stock_trades(acc) or [])]
    log["after"] = {"cash": asset.cash, "frozen": asset.frozen_cash, "market_value": asset.market_value,
                    "total_asset": asset.total_asset, "positions": pos, "trades_today": trades}
    print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    import faulthandler
    faulthandler.dump_traceback_later(150, exit=True)      # 卡住就打印调用栈并强退, 不再无声挂起
    try:
        main()
    except SystemExit as e:                                 # 正常 raise SystemExit 会等 xtquant 后台线程而挂住
        print(f"[cash_test] 停止: {e}", flush=True)
        os._exit(2)
