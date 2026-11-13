"""界面下单执行器 (2026-11-13 起, 替代依赖 miniQMT 的 executor): 用填单演练那套已验证的步骤真正提交股票委托。

  .venv\\Scripts\\python.exe -m qmt_bridge.ui_executor              # 计划任务 14:44 启动
  .venv\\Scripts\\python.exe -m qmt_bridge.ui_executor --shadow     # 只填单核对, 不提交 (= 带对账的演练)
  .venv\\Scripts\\python.exe -m qmt_bridge.ui_executor --channel-test 601818   # 实盘通道测试: 跌停价买 100 股 (不会成交)

实盘 = 单据 trading_enabled (网页总开关) 且 本地 D:\\qmtcode\\TRADING_ENABLED 存在; 缺一条就影子模式。
流程 (任何一步不对就停下后面所有委托、清空面板、报警; 宁可少下, 不可错下):
  1. 一天只跑一次: 第一次点提交前落 exec_done/<日>.json, 之后再启动直接退出
  2. 单据 problems 非空不下; 读屏: 持仓必须与线账本逐只一致, 今天委托表里已有这些代码的委托就不重复下
  3. 先卖后买。卖价 = max(跌停, 现价×0.99), 买价 = min(涨停, 现价×1.02) (连续竞价里立即成交);
     买入用钱 = min(QMT 可用, 线现金 + 预计卖出回款) 留 0.5%, 不够减手 (executor.plan_sells/plan_buys)
  4. 每笔: 填单 -> OCR 与模型读回一致 -> 提交按钮颜色 = 方向 -> 硬限制 (价在涨跌停内、偏离现价 ≤ 3%、整百股、
     单笔金额不超线现金) -> 点提交 -> 确认框内容逐项一致才点「确定」(否则按 Esc 取消)
  5. 下完重读委托表, 核对每笔都在; 回报 041。记账仍由 18:50 自动确认按券商真实成交完成。
与回测的差别: 回测按收盘价成交; 界面每笔约 40 秒, 只能在 14:44~14:56 连续竞价里按当时价格成交。
"""
import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

LOCAL_SWITCH = Path(r"D:\qmtcode\TRADING_ENABLED")
DONE_DIR = Path(r"D:\qmtcode\exec_done")
WINDOW = ("14:40:00", "14:56:00")           # 允许下单的时段 (14:57 起收盘集合竞价, 界面来不及)
MAX_DEV = 0.03
SUBMIT_BOX = (480, 790, 650, 880)            # 下单面板底部提交按钮所在区域
SIDE_CN = {"buy": "买入", "sell": "卖出"}


# ── 纯计算 (可测) ───────────────────────────────────────────────────
def quote_dict(raw):
    """rehearse.quote 的 (名称, 现价, 涨停, 跌停, 买一) -> executor 用的行情 dict"""
    name, last, up, down = raw[:4]
    return {"name": name, "last": last, "up": up, "down": down, "bid1": raw[4] if len(raw) > 4 else last}


def held(snap):
    """读屏快照 -> {代码: {volume, can_use}}, 不含逆回购/货币 ETF (与 041 对账口径一致)"""
    from .trading import is_cash_equiv
    out = {}
    for p in snap.get("positions") or []:
        if is_cash_equiv(p["stock_code"]) or int(p.get("volume") or 0) <= 0:
            continue
        o = out.setdefault(p["stock_code"][:6], {"volume": 0, "can_use": 0})
        o["volume"] += int(p["volume"])
        o["can_use"] += int(p.get("can_use_volume") or 0)
    return out


def precheck(ticket, snap):
    """返回问题列表 (空 = 可以下)。"""
    probs = list((ticket or {}).get("problems") or [])
    pos = held(snap)
    want = {lot["code"]: int(lot["shares"]) for lot in ticket.get("lots") or []}
    have = {c: p["volume"] for c, p in pos.items()}
    if have != want:
        probs.append(f"下单前对账不一致: QMT {have} vs 线账本 {want}")
    codes = {x["code"] for x in (ticket.get("buy") or []) + (ticket.get("sell") or [])}
    dup = sorted({o["stock_code"][:6] for o in snap.get("orders") or []} & codes)
    if dup:
        probs.append(f"今天委托表里已有 {dup} 的委托, 不重复下单")
    return probs


def guard(o, q, line_cash):
    """硬限制; 返回 (允许, 原因)。o: {side, code, price, volume}。"""
    if o["volume"] < 100 or o["volume"] % 100:
        return False, f"股数 {o['volume']} 不是整百"
    if not (q["down"] - 1e-6 <= o["price"] <= q["up"] + 1e-6):
        return False, f"价格 {o['price']} 不在跌停 {q['down']} ~ 涨停 {q['up']} 之间"
    if q["last"] <= 0 or abs(o["price"] / q["last"] - 1) > MAX_DEV + 1e-6:
        return False, f"价格 {o['price']} 偏离现价 {q['last']} 超过 {MAX_DEV:.0%}"
    if o["side"] == "buy" and o["price"] * o["volume"] > line_cash * 1.005:
        return False, f"金额 {o['price'] * o['volume']:,.0f} 超过线内现金 {line_cash:,.0f}"
    return True, "ok"


def plan(ticket, snap, quotes):
    """返回 (委托列表 先卖后买, 跳过)。"""
    from .executor import FEE_RATE, plan_buys, plan_sells
    sells, s_skip = plan_sells(ticket, held(snap), quotes)
    proceeds = sum(o["volume"] * o["price"] for o in sells)
    cash = min(float((snap.get("asset") or {}).get("cash") or 0) + proceeds,
               float(ticket.get("line_cash") or 0) + proceeds * (1 - FEE_RATE))
    buys, b_skip = plan_buys(ticket, quotes, cash)                # plan_buys 自己再留 0.5%
    return sells + buys, s_skip + b_skip


# ── 界面动作 ────────────────────────────────────────────────────────
def press_esc():
    from .ui_order import _key
    _key(0x1B)
    _key(0x1B, up=True)


def submit_one(o, live, log):
    """填单 -> 核对 -> (实盘) 提交 + 确认。返回记录 (problems 非空 = 失败, 调用方停止后续)。"""
    from .ocr_read import locate_text
    from .rehearse import clear_qty, fill_one
    from .ui_agent import screenshot_windows
    from .ui_channel import click_xy_verified, confirm, side_by_button
    cn = SIDE_CN[o["side"]]
    ro = {"side": cn, "code": o["code"], "name": o.get("name"), "price": o["price"], "shares": o["volume"]}
    rec = fill_one(ro, log, keep=True)
    lab_q = rec.get("qty_label")
    if not rec["problems"] and side_by_button()[0] != cn:
        rec["problems"].append("提交按钮颜色与方向不符")
    if rec["problems"] or not live:
        if lab_q:
            clear_qty(lab_q, o["code"])
        rec["submitted"] = False
        return rec
    xy = locate_text(screenshot_windows(), cn, SUBMIT_BOX, 2)
    if not xy:
        rec["problems"].append(f"OCR 没在面板底部找到「{cn}」提交按钮")
        clear_qty(lab_q, o["code"])
        return rec
    sub = click_xy_verified(*xy, f"下单面板底部的「{cn}」提交按钮", cn[0], f"ex_submit_{o['code']}")
    if not sub["ok"]:
        rec["problems"].append(f"提交按钮复核没过: {sub.get('verify')}")
        clear_qty(lab_q, o["code"])
        return rec
    exp = {"code": o["code"], "side": cn, "price": round(o["price"], 3), "volume": o["volume"]}
    conf = confirm(exp, "order", until=WINDOW[1])
    rec["confirm"] = {k: conf.get(k) for k in ("ok", "why", "bad", "dialog")}
    if not conf.get("ok"):
        press_esc()                                  # 确认框不对: Esc = 取消, 不下这笔
        time.sleep(0.8)
        rec["problems"].append(f"确认框没通过, 已取消: {conf.get('why')} {conf.get('bad') or ''}")
        clear_qty(lab_q, o["code"])
        return rec
    rec["submitted"] = True
    log(f"  已提交 {cn} {o['code']} {o['price']}×{o['volume']}")
    return rec


def run(live, log=print, ticket=None, fetch_quote=None):
    from .executor import BASE_URL, signed_post
    from .push_snapshot import DEFAULT_KEY
    from .rehearse import quote
    from .screen_snapshot import take
    fetch_quote = fetch_quote or quote
    today = date.today().isoformat()
    marker = DONE_DIR / f"{today}.json"
    rep = {"today": today, "profile": "qmt10w", "mode": "live" if live else "shadow", "orders": [], "skips": [],
           "problems": [], "started_at": datetime.now().isoformat(timespec="seconds")}
    if live and marker.exists():
        return {**rep, "status": "already_done"}
    now = datetime.now().strftime("%H:%M:%S")
    if live and not (WINDOW[0] <= now <= WINDOW[1]):
        return {**rep, "status": "outside_window", "problems": [f"现在 {now} 不在下单时段 {WINDOW}"]}
    key = Path(DEFAULT_KEY).read_text(encoding="utf-8").strip().encode()
    if ticket is None:
        for _ in range(3):
            st, resp = signed_post(f"{BASE_URL}/api/qmt/ticket", {"profile": "qmt10w", "today": today}, key)
            if st == 200:
                break
            time.sleep(10)
        if st != 200:
            return {**rep, "status": "no_ticket", "problems": [f"取单据失败 {st}: {resp.get('error')}"]}
        ticket = {**resp["ticket"], "problems": resp.get("problems") or []}
    if live and not ticket.get("trading_enabled"):
        live, rep["mode"] = False, "shadow"
        rep["problems"].append("网页自动下单开关是关的, 只演算")
    if not (ticket.get("buy") or ticket.get("sell")):
        return {**rep, "status": "nothing_to_do"}
    snap = take("manual", log=log)
    if not snap.get("account_online"):
        return {**rep, "status": "snapshot_offline", "problems": [snap.get("error")]}
    probs = precheck(ticket, snap)
    if probs:
        return {**rep, "status": "precheck_failed", "problems": rep["problems"] + probs}
    codes = [x["code"] for x in (ticket.get("sell") or []) + (ticket.get("buy") or [])]
    quotes = {}
    for c in codes:
        try:
            quotes[c] = quote_dict(fetch_quote(c))
        except Exception as e:  # noqa: BLE001
            rep["skips"].append({"code": c, "reason": f"取行情失败 {e}"})
    orders, skips = plan(ticket, snap, quotes)
    rep["skips"] += skips
    for o in orders:
        o["name"] = quotes[o["code"]]["name"]
        now = datetime.now().strftime("%H:%M:%S")
        if live and now > WINDOW[1]:
            rep["problems"].append(f"{now} 已过下单时段, 剩下的不下: {o['code']} 起")
            break
        ok, why = guard(o, quotes[o["code"]], float(ticket.get("line_cash") or 0))
        if not ok:
            rep["problems"].append(f"{o['code']} 硬限制没过, 停止: {why}")
            break
        if live and not marker.exists():
            DONE_DIR.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "orders": orders},
                                         ensure_ascii=False, default=str), encoding="utf-8")
        rec = submit_one(o, live, log)
        rep["orders"].append({**o, "submitted": rec.get("submitted"), "seconds": rec.get("seconds"),
                              "problems": rec["problems"], "confirm": rec.get("confirm")})
        if rec["problems"]:
            rep["problems"].append(f"{SIDE_CN[o['side']]} {o['code']}: {'; '.join(rec['problems'])} —— 停止后续委托")
            break
    if live and any(x.get("submitted") for x in rep["orders"]):
        time.sleep(3)
        snap1 = take("manual", log=log)
        seen = {x["stock_code"][:6] for x in snap1.get("orders") or []}
        miss = [x["code"] for x in rep["orders"] if x.get("submitted") and x["code"] not in seen]
        if not snap1.get("account_online"):
            rep["problems"].append(f"下完单读屏离线, 请人工核对委托: {snap1.get('error')}")
        elif miss:
            rep["problems"].append(f"提交后委托表里没看到 {miss}, 请人工核对")
    rep["status"] = "done" if not rep["problems"] else "stopped"
    rep["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return rep


def channel_test(code, log=print):
    """实盘通道测试: 按跌停价买 100 股 (挂在跌停上几乎不可能成交), 走与正式下单完全相同的填单/核对/提交/确认。
    之后用 ui_channel 的撤单步骤撤掉。只在连续竞价时段。"""
    from .rehearse import quote
    q = quote_dict(quote(code))
    o = {"side": "buy", "code": code, "name": q["name"], "price": q["down"], "volume": 100}
    log(f"通道测试: 买入 {code} {q['name']} 100 股 @ 跌停 {q['down']} (现价 {q['last']})")
    return submit_one(o, True, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shadow", action="store_true", help="只填单核对, 不提交")
    ap.add_argument("--channel-test", metavar="CODE", help="实盘通道测试: 跌停价买 100 股")
    a = ap.parse_args(argv)
    from .ui_lock import ui_lock

    def log(m):
        print(m, flush=True)
    try:
        with ui_lock("ui_executor", wait_s=300):
            from .ui_agent import ensure_qmt_foreground
            probs = ensure_qmt_foreground(log)
            if a.channel_test:
                out = {"problems": probs} if probs else channel_test(a.channel_test, log)
            else:
                live = not a.shadow and LOCAL_SWITCH.exists()
                out = run(live, log)
                if not a.shadow and not LOCAL_SWITCH.exists():
                    out["problems"] = [f"本地开关 {LOCAL_SWITCH} 不存在, 只演算"] + list(out.get("problems") or [])
                from .executor import BASE_URL, signed_post
                from .push_snapshot import DEFAULT_KEY
                key = Path(DEFAULT_KEY).read_text(encoding="utf-8").strip().encode()
                signed_post(f"{BASE_URL}/api/qmt/exec_report", out, key)
    except TimeoutError as e:
        out = {"status": "ui_busy", "problems": [str(e)]}
    print(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **out}, ensure_ascii=False, default=str)[:4000],
          flush=True)
    return 0 if not out.get("problems") else 1


if __name__ == "__main__":
    sys.exit(main())
