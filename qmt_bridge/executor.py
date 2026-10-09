"""阶段二执行器: 14:55 盘中卖出, 14:57 收盘集合竞价买入 (Windows 计划任务 14:53:30 启动)。

  .venv\\Scripts\\python.exe -m qmt_bridge.executor --auto          # 计划任务用
  .venv\\Scripts\\python.exe -m qmt_bridge.executor --force-shadow  # 只算不下单

实盘 = 网页总开关开 且 本地 D:\\qmtcode\\TRADING_ENABLED 文件存在 且 --auto; 缺一条都是影子模式
(用真实行情算「会下什么单」, 回报给 041, 不发任何委托)。

流程 (单据由 041 /api/qmt/ticket 给出, 问题非空就不下单):
  1. 下单前对账: QMT 持仓必须与线账本逐只一致; 今天已有本系统委托则不再下 (防重复)
  2. 14:55:00 卖出: 限价 = max(跌停价, 现价×0.99) 立即成交; 14:56:45 没成交完的撤掉,
     剩余在 14:57 收盘竞价按跌停价挂 (竞价统一价成交, 跌停价只是保证参与)
  3. 14:57:10 买入: 限价 = min(涨停价, 现价×1.02), 进收盘竞价按收盘价成交; 可用钱 =
     min(QMT 可用资金, 线现金 + 卖出回款) 再留 0.5%; 不够就减手, 不足一手跳过; 涨停跳过
  4. 15:00:40 收集最终状态, 回报 041。记账仍由 18:50 自动确认按真实成交完成。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

from .push_snapshot import DEFAULT_KEY, sign
from .readonly_probe import DEFAULT_USERDATA, ReadOnlyTrader, connect_with_timeout
from .trading import (
    STOCK_BUY,
    STOCK_SELL,
    TERMINAL,
    OrderRejectedError,
    TradingTrader,
    place_limit,
    round_tick,
    to_xt_code,
)

BASE_URL = "http://eez041.ece.ust.hk:8737"
LOCAL_SWITCH = Path(r"D:\qmtcode\TRADING_ENABLED")
T_SELL, T_SELL_DEADLINE, T_BUY, T_BUY_LATEST, T_DONE = "14:55:00", "14:56:45", "14:57:10", "14:59:30", "15:00:40"
FEE_RATE, MIN_FEE, CASH_RESERVE = 0.001, 5.0, 0.005
BUY_PREMIUM, SELL_DISCOUNT = 0.02, 0.01


# ── 纯计算 ────────────────────────────────────────────────────────
def plan_sells(ticket, positions, quotes):
    """positions: {code6: {"volume", "can_use"}}; 返回 (委托单, 跳过)。"""
    orders, skips = [], []
    for s in ticket.get("sell") or []:
        c = s["code"]
        q = quotes.get(c) or {}
        vol = min(int(s["shares"]), int((positions.get(c) or {}).get("can_use") or 0))
        if vol <= 0:
            skips.append({"code": c, "side": "sell", "reason": "没有可卖数量(T+1 或冻结)"})
            continue
        last, down, up = q.get("last") or 0, q.get("down"), q.get("up")
        if not (last > 0 and down and up):
            skips.append({"code": c, "side": "sell", "reason": "停牌或无行情"})
            continue
        if last <= down + 1e-9:
            skips.append({"code": c, "side": "sell", "reason": "跌停卖不出"})
            continue
        ref = min(last, q.get("bid1") or last)
        orders.append({"code": c, "side": "sell", "volume": vol, "phase": "continuous",
                       "price": max(down, round_tick(ref * (1 - SELL_DISCOUNT))), "up": up, "down": down})
    return orders, skips


def plan_buys(ticket, quotes, cash):
    """按计划顺序买, 用现价×1.02 作限价占用资金; 不够减手, 不足一手跳过。返回 (委托单, 跳过)。"""
    orders, skips = [], []
    avail = cash * (1 - CASH_RESERVE)
    cap = float(ticket.get("slot_cap") or 0)
    for b in ticket.get("buy") or []:
        c = b["code"]
        q = quotes.get(c) or {}
        last, down, up = q.get("last") or 0, q.get("down"), q.get("up")
        if not (last > 0 and down and up):
            skips.append({"code": c, "side": "buy", "reason": "停牌或无行情"})
            continue
        if last >= up - 1e-9:
            skips.append({"code": c, "side": "buy", "reason": "涨停买不进"})
            continue
        price = min(up, round_tick(last * (1 + BUY_PREMIUM), up=True))
        unit = price * (1 + FEE_RATE)
        vol = int(b["shares"]) // 100 * 100
        if cap and vol * price > cap:
            vol = int(cap // price // 100 * 100)
        if vol * unit + MIN_FEE > avail:
            vol = int(max(0.0, avail - MIN_FEE) // unit // 100 * 100)
        if vol < 100:
            skips.append({"code": c, "side": "buy", "reason": "现金不足一手" if avail < cap else "超过单笔上限"})
            continue
        if vol < int(b["shares"]):
            skips.append({"code": c, "side": "buy", "reason": f"减手 {b['shares']}→{vol}"})
        avail -= vol * unit + MIN_FEE
        orders.append({"code": c, "side": "buy", "volume": vol, "price": price, "phase": "auction",
                       "up": up, "down": down, "max_value": cap or vol * price})
    return orders, skips


def positions_of(raw_positions):
    out = {}
    for p in raw_positions or []:
        c = str(getattr(p, "stock_code", ""))[:6]
        v, cu = int(getattr(p, "volume", 0) or 0), int(getattr(p, "can_use_volume", 0) or 0)
        if v > 0:
            o = out.setdefault(c, {"volume": 0, "can_use": 0})
            o["volume"] += v
            o["can_use"] += cu
    return out


# ── 编排 ──────────────────────────────────────────────────────────
class Clock:
    def now(self):
        return datetime.now()

    def sleep(self, s):
        time.sleep(s)

    def wait_until(self, hms):
        while self.now().strftime("%H:%M:%S") < hms:
            self.sleep(0.5)


def quotes_for(xtdata, codes):
    xt = [to_xt_code(c) for c in codes]
    ticks = xtdata.get_full_tick(xt) if xt else {}
    out = {}
    for c, x in zip(codes, xt, strict=True):
        t, d = ticks.get(x) or {}, xtdata.get_instrument_detail(x) or {}
        bids = t.get("bidPrice") or [0]
        out[c] = {"last": float(t.get("lastPrice") or 0), "bid1": float(bids[0] or 0),
                  "up": float(d.get("UpStopPrice") or 0), "down": float(d.get("DownStopPrice") or 0)}
    return out


def run(ticket, live, trader, account, quote_fn, clock, remark_prefix):
    """返回回报 dict。live=False 时 trader 只用于查询, 绝不委托。"""
    rep = {"today": ticket["today"], "profile": ticket["profile"], "mode": "live" if live else "shadow",
           "signal_date": ticket["signal_date"], "started_at": clock.now().isoformat(timespec="seconds"),
           "orders": [], "skips": [], "errors": [], "status": "aborted"}
    if clock.now().strftime("%H:%M:%S") > T_BUY_LATEST:
        rep["errors"].append(f"启动太晚 ({clock.now():%H:%M:%S}), 今天不下单")
        return rep
    pos = positions_of(trader.query_stock_positions(account))
    want = {lot["code"]: int(lot["shares"]) for lot in ticket.get("lots") or []}
    if {c: p["volume"] for c, p in pos.items()} != want:
        qmt_txt = ", ".join(f"{c}:{p['volume']}" for c, p in sorted(pos.items()))   # Windows 是 3.11, 不能嵌套同种引号
        rep["errors"].append(f"下单前对账不一致: QMT {{{qmt_txt}}} vs 线账本 {want}")
        return rep
    mine = [o for o in trader.query_stock_orders(account, False) or []
            if str(getattr(o, "order_remark", "")).startswith(remark_prefix)]
    if mine:
        rep["errors"].append(f"今天已有 {len(mine)} 笔本系统委托, 不重复下单")
        return rep
    asset = trader.query_stock_asset(account)
    acct_cash = float(getattr(asset, "cash", 0) or 0)

    def submit(o):
        if not live:
            o["order_id"] = None
            rep["orders"].append(o)
            return
        try:
            o["order_id"] = place_limit(trader, account, to_xt_code(o["code"]),
                                        STOCK_SELL if o["side"] == "sell" else STOCK_BUY, o["volume"], o["price"],
                                        o["up"], o["down"], o.get("max_value", 1e12), f"{remark_prefix}-{o['phase'][0].upper()}")
        except (OrderRejectedError, PermissionError) as e:
            rep["errors"].append(f"{o['code']} 委托被拦: {e}")
            return
        rep["orders"].append(o)

    def refresh(o):
        if not live or not o.get("order_id"):
            return
        r = trader.query_stock_order(account, o["order_id"])
        if r is not None:
            o.update(status=getattr(r, "order_status", None), traded_volume=int(getattr(r, "traded_volume", 0) or 0),
                     traded_price=getattr(r, "traded_price", None))

    # 卖出 (盘中)
    proceeds, auction_sells = 0.0, []
    if ticket.get("sell"):
        clock.wait_until(T_SELL)
        sells, skips = plan_sells(ticket, pos, quote_fn([s["code"] for s in ticket["sell"]]))
        rep["skips"] += skips
        for o in sells:
            submit(o)
        if live:
            while clock.now().strftime("%H:%M:%S") < T_SELL_DEADLINE:
                for o in sells:
                    refresh(o)
                if all(o.get("status") in TERMINAL for o in sells if o.get("order_id")):
                    break
                clock.sleep(1)
            for o in sells:
                refresh(o)
                left = o["volume"] - int(o.get("traded_volume") or 0)
                if o.get("order_id") and left > 0 and o.get("status") not in TERMINAL:
                    trader.cancel_order_stock(account, o["order_id"])
                if o.get("order_id") and left > 0:
                    auction_sells.append({**o, "volume": left, "price": o["down"], "phase": "auction", "order_id": None})
                proceeds += int(o.get("traded_volume") or 0) * float(o.get("traded_price") or 0)
        else:
            proceeds = sum(o["volume"] * o["price"] for o in sells)
    # 买入 (收盘竞价) + 卖出剩余进竞价
    clock.wait_until(T_BUY)
    if live:
        acct_cash = float(getattr(trader.query_stock_asset(account), "cash", 0) or 0)
    cash = min(acct_cash if live else acct_cash + proceeds, float(ticket["line_cash"]) + proceeds * (1 - FEE_RATE))
    for o in auction_sells:
        submit(o)
    buys, skips = plan_buys(ticket, quote_fn([b["code"] for b in ticket.get("buy") or []]), cash)
    rep["skips"] += skips
    rep["buy_cash"] = round(cash, 2)
    for o in buys:
        submit(o)
    clock.wait_until(T_DONE)
    for o in rep["orders"]:
        refresh(o)
    rep["status"] = "done"
    rep["finished_at"] = clock.now().isoformat(timespec="seconds")
    return rep


# ── 网络 ──────────────────────────────────────────────────────────
def signed_post(url, payload, key, timeout=20.0):
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    ts = str(int(time.time()))
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "X-QMT-Ts": ts, "X-QMT-Sig": sign(body, ts, key)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, {"error": f"网络错误: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="qmt10w")
    ap.add_argument("--auto", action="store_true", help="双钥匙都开才实盘")
    ap.add_argument("--force-shadow", action="store_true")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--key-file", default=DEFAULT_KEY)
    args = ap.parse_args(argv)
    key = Path(args.key_file).read_text(encoding="utf-8").strip().encode()
    today = date.today().isoformat()
    out_dir = Path(__file__).resolve().parents[1] / "exec"
    out_dir.mkdir(exist_ok=True)

    def finish(rep):
        (out_dir / f"exec_{today.replace('-', '')}_{rep['mode']}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        st, resp = signed_post(f"{args.base_url}/api/qmt/exec_report", rep, key)
        print(f"[{datetime.now():%H:%M:%S}] mode={rep['mode']} status={rep.get('status')} orders={len(rep.get('orders') or [])} "
              f"errors={rep.get('errors') or rep.get('problems')} report={st}")
        sys.stdout.flush()
        os._exit(0)

    st, resp = signed_post(f"{args.base_url}/api/qmt/ticket", {"profile": args.profile, "today": today}, key)
    if st != 200:
        finish({"today": today, "mode": "skipped", "status": "no_ticket", "problems": [f"取单据失败 {st}: {resp.get('error')}"]})
    ticket, probs = resp["ticket"], resp["problems"]
    if probs:
        finish({"today": today, "mode": "skipped", "status": "problems", "problems": probs})
    if not ticket.get("sell") and not ticket.get("buy"):
        finish({"today": today, "mode": "skipped", "status": "nothing_to_do"})
    live = args.auto and not args.force_shadow and ticket.get("trading_enabled") and LOCAL_SWITCH.exists()

    from xtquant import xtdata
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount
    raw = XtQuantTrader(DEFAULT_USERDATA, int(time.time()) % 100000 + 500000)
    trader = TradingTrader(raw) if live else ReadOnlyTrader(raw)
    trader.start()
    if connect_with_timeout(trader, 20) != 0:
        finish({"today": today, "mode": "skipped", "status": "no_qmt", "problems": ["连不上 miniQMT"]})
    infos = [i for i in (trader.query_account_infos() or []) if getattr(i, "account_type", None) == 2]
    if not infos:
        finish({"today": today, "mode": "skipped", "status": "no_account", "problems": ["资金账号不在线"]})
    acc = StockAccount(str(infos[0].account_id), "STOCK")
    trader.subscribe(acc)
    rep = run(ticket, bool(live), trader, acc, lambda codes: quotes_for(xtdata, codes), Clock(),
              f"q10w-{today[5:7]}{today[8:10]}")
    rep["keys"] = {"web": bool(ticket.get("trading_enabled")), "local": LOCAL_SWITCH.exists(), "auto": args.auto}
    finish(rep)


if __name__ == "__main__":
    main()
