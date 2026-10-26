"""读屏账户快照 (2026-10-26): miniQMT 被停用后, 从 QMT 普通客户端界面读资金/持仓/成交/委托, 按原 API 快照格式推给 041。

  .venv\\Scripts\\python.exe -m qmt_bridge.screen_snapshot --kind close [--dry-run]
必须在登录用户的桌面会话里、提权运行 (QMT 是提权进程, 切换页签的点击要同等权限)。

宁可报离线, 不让读错的数进账本:
  * 每张表用两种放大倍数各读一次, 规范化后必须一致; 不一致读第三次取多数, 仍无多数 -> 失败。
  * 算术交叉核对: 可用+冻结+市值≈总资产; 持仓市值合计≈总市值; 每笔成交 价×量≈金额; 成交量≤委托量。
  * 任一不过 -> account_online=False + error, 041 当离线处理 (告警, 当晚自动确认不拿它记账)。
只点导航 (底部 持仓/成交/委托 页签、左侧「交易」), 不碰任何下单按钮。
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .push_snapshot import DEFAULT_KEY, DEFAULT_URL, push
from .ui_agent import ask, crop

STATUS_CODE = {"未报": 48, "待报": 49, "已报": 50, "已报待撤": 51, "部成待撤": 52, "部撤": 53, "已撤": 54, "部成": 55,
               "已成": 56, "废单": 57}
# 10-26 截图量出 (QMT 最大化 1920x1080, 交易页): 底部页签与表格区域。点击前仍做独立复核。
TABS = {"positions": (840, 474, "持仓"), "trades": (905, 474, "成交"), "orders": (1000, 474, "委托")}
TABLE_BOX = (780, 482, 1920, 738)
FUNDS_BOX = (780, 742, 1920, 830)

FUNDS_PROMPT = ("这是 QMT「账号资金」表格的放大截图。只返回 JSON: {account_last4 (资金账号后4位), total_asset (总资产), "
                "available (可用金额), frozen (冻结金额), market_value (总市值), withdrawable (可取金额), "
                "login_status (登录状态那一列的文字)}。数字转数值, 看不清填 null, 不要猜。")
TABLE_PROMPTS = {
    "positions": ("这是 QMT「持仓」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{code (6位证券代码), name, "
                  "volume (持仓数量/当前拥股), can_use (可用数量), cost (成本价), market_value (市值)}]}。"
                  "表格为空就 rows=[]。数字转数值, 看不清填 null, 不要猜, 不要把别的列错填进来。"),
    "trades": ("这是 QMT「成交」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{time (成交时间), code (6位证券代码), "
               "side (买入/卖出), price (成交价格/成交均价), volume (成交数量), amount (成交金额), order_id (合同编号), "
               "trade_id (成交编号)}]}。表格为空就 rows=[]。数字转数值, 看不清填 null, 不要把别的列错填进来。"),
    "orders": ("这是 QMT「委托」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{time (委托时间), code (6位证券代码), "
               "side (买入/卖出), price (委托价格), volume (委托量), traded_volume (成交数量, 注意不是委托剩余量), "
               "status (委托状态), order_id (合同编号, 注意不是资金账号)}]}。表格为空就 rows=[]。"),
}


def xt_code(code6):
    c = str(code6)[:6]
    return c + (".SH" if c[0] in "569" else ".BJ" if c[0] in "48" else ".SZ")


def side_type(side):
    s = str(side or "")
    return 23 if "买" in s else 24 if "卖" in s else None


def _num(v):
    try:
        return round(float(str(v).replace(",", "").replace("股", "").strip()), 2)
    except (TypeError, ValueError):
        return None


def canon_rows(kind, rows):
    """规范化后用于比对: 数字统一两位小数、代码 6 位、按 (代码, 时间, 价, 量) 排序。"""
    keys = {"positions": ("code", "volume", "can_use", "cost", "market_value"),
            "trades": ("time", "code", "side", "price", "volume", "amount", "order_id"),
            "orders": ("time", "code", "side", "price", "volume", "traded_volume", "status", "order_id")}[kind]
    out = []
    for r in rows or []:
        c = {}
        for k in keys:
            v = r.get(k)
            c[k] = str(v)[:6] if k == "code" else (_num(v) if k in ("volume", "can_use", "cost", "market_value", "price",
                                                                    "amount", "traded_volume") else
                                                   (str(v).strip() if v is not None else None))
        out.append(c)
    return sorted(out, key=lambda c: json.dumps(c, ensure_ascii=False, sort_keys=True))


def canon_funds(f):
    return {k: _num(f.get(k)) for k in ("total_asset", "available", "frozen", "market_value")}


def agree(reads, canon):
    """多次读的规范化结果: 两次一致即采用; 否则取多数; 无多数返回 None。"""
    cs = [json.dumps(canon(r), ensure_ascii=False, sort_keys=True) for r in reads]
    top, n = Counter(cs).most_common(1)[0]
    return (json.loads(top) if n >= 2 else None), cs


def read_box(png, box, prompt, scale):
    x0, y0, x1, y1 = box
    part, _ = crop(png, (x0 + x1) / 2, (y0 + y1) / 2, w=x1 - x0, h=y1 - y0, scale=scale, mark=False)
    res, _ = ask(part, prompt)
    return res


def read_agreed(png, box, prompt, canon, scales=(2, 3, 2.5)):
    reads = [read_box(png, box, prompt, s) for s in scales[:2]]
    val, cs = agree(reads, canon)
    if val is None:
        reads.append(read_box(png, box, prompt, scales[2]))
        val, cs = agree(reads, canon)
    return val, {"reads": len(reads), "agreed": val is not None}, reads


def checks(funds, positions, trades, orders):
    """算术交叉核对, 返回问题列表 (空 = 通过)。"""
    probs = []
    t, a, fr, mv = (funds.get(k) for k in ("total_asset", "available", "frozen", "market_value"))
    if None in (t, a, mv):
        probs.append(f"资金读数不全: {funds}")
    elif abs(t - (a + (fr or 0) + mv)) > max(1.0, 0.002 * t):
        probs.append(f"可用 {a} + 冻结 {fr} + 市值 {mv} ≠ 总资产 {t}")
    if positions and mv is not None:
        s = sum(p.get("market_value") or 0 for p in positions)
        if abs(s - mv) > max(1.0, 0.005 * mv):
            probs.append(f"持仓市值合计 {s:.2f} ≠ 总市值 {mv}")
    for p in positions:
        if not p.get("volume") or p.get("volume") % 1 or (p.get("can_use") or 0) > p["volume"]:
            probs.append(f"持仓行异常: {p}")
    for tr in trades:
        if None in (tr.get("price"), tr.get("volume"), tr.get("amount")) or side_type(tr.get("side")) is None:
            probs.append(f"成交行不全: {tr}")
        elif abs(tr["price"] * tr["volume"] - tr["amount"]) > max(0.05, 0.002 * tr["amount"]):
            probs.append(f"成交 价×量 {tr['price']}×{tr['volume']} ≠ 金额 {tr['amount']}")
    for o in orders:
        if o.get("traded_volume") is not None and o.get("volume") is not None and o["traded_volume"] > o["volume"]:
            probs.append(f"委托成交量大于委托量: {o}")
    return probs


def build_snapshot(kind, funds, positions, trades, orders, account_last4, problems, meta):
    masked = "********" + str(account_last4 or "????")[-4:]
    snap = {"kind": kind, "host": os.environ.get("COMPUTERNAME", ""), "probe_at": datetime.now().isoformat(timespec="seconds"),
            "source": "screen", "account": masked, "account_status": 0 if not problems else None,
            "account_online": not problems, "screen": meta,
            "asset": {"account_type": 2, "account_id": masked, "cash": funds.get("available"),
                      "frozen_cash": funds.get("frozen"), "market_value": funds.get("market_value"),
                      "total_asset": funds.get("total_asset")},
            "positions": [{"account_id": masked, "stock_code": xt_code(p["code"]), "volume": int(p["volume"]),
                           "can_use_volume": int(p.get("can_use") or 0), "frozen_volume": None, "on_road_volume": None,
                           "yesterday_volume": None, "open_price": p.get("cost"), "avg_price": p.get("cost"),
                           "market_value": p.get("market_value")} for p in positions],
            "orders": [{"stock_code": xt_code(o["code"]), "order_id": o.get("order_id"), "order_sysid": None,
                        "order_time": o.get("time"), "order_type": side_type(o.get("side")),
                        "order_volume": int(o.get("volume") or 0), "price_type": None, "price": o.get("price"),
                        "traded_volume": int(o.get("traded_volume") or 0), "traded_price": None,
                        "order_status": STATUS_CODE.get(str(o.get("status"))), "status_msg": o.get("status"),
                        "strategy_name": "", "order_remark": ""} for o in orders],
            "trades": [{"stock_code": xt_code(t["code"]), "order_type": side_type(t.get("side")), "traded_id": t.get("trade_id"),
                        "traded_time": t.get("time"), "traded_price": t.get("price"), "traded_volume": int(t["volume"]),
                        "traded_amount": t.get("amount"), "order_id": t.get("order_id"), "order_sysid": None,
                        "strategy_name": "", "order_remark": ""} for t in trades]}
    if problems:
        snap["error"] = "读屏核对不一致, 按离线处理: " + "; ".join(problems)[:600]
    return snap


def take(kind, log=print):
    from .ui_agent import login_normal_mode
    from .ui_channel import click_xy_checked, click_xy_verified, read, snap

    st, _ = read("state", "snap_state")
    if not (st.get("logged_in") and st.get("trade_online")):
        log(f"未登录或交易不在线 ({st}), 自动普通方式登录")
        if not login_normal_mode(log=log):
            return build_snapshot(kind, {}, [], [], [], None, ["QMT 普通方式登录失败"], {"state": st})
        time.sleep(5)
        st, _ = read("state", "snap_state2")
    if st.get("popups"):
        click_xy_checked(1310, 255, "snap_close_notice")       # 华泰信息通知 (只有它会常驻)
    if "交易" not in str(st.get("page")):
        click_xy_verified(22, 203, "最左侧竖排导航栏的「交易」", "交易", "snap_nav_trade")
        time.sleep(1.5)
    meta, problems, tables = {"tabs": {}}, [], {}
    for kind_t in ("positions", "trades", "orders"):
        x, y, label = TABS[kind_t]
        c = click_xy_verified(x, y, f"底部表格区的「{label}」页签", label, f"snap_tab_{kind_t}")
        if not c["ok"]:
            problems.append(f"切换「{label}」页签失败 (复核: {c.get('verify')})")
            tables[kind_t] = []
            continue
        time.sleep(1.2)
        png, path = snap(f"snap_{kind_t}")
        val, m, _ = read_agreed(png, TABLE_BOX, TABLE_PROMPTS[kind_t], lambda r, k=kind_t: canon_rows(k, r.get("rows")))
        meta["tabs"][kind_t] = {**m, "shot": str(path)}
        if val is None:
            problems.append(f"「{label}」表多次读数不一致")
        tables[kind_t] = val or []
    png, path = snap("snap_funds")
    funds, m, reads = read_agreed(png, FUNDS_BOX, FUNDS_PROMPT, canon_funds)
    meta["funds"] = {**m, "shot": str(path)}
    if funds is None:
        problems.append("资金表多次读数不一致")
        funds = {}
    last4 = next((r.get("account_last4") for r in reads if r.get("account_last4")), None)
    problems += checks(funds, tables["positions"], tables["trades"], tables["orders"])
    return build_snapshot(kind, funds, tables["positions"], tables["trades"], tables["orders"], last4, problems, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=("morning", "close", "manual"), required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--key-file", default=DEFAULT_KEY)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    snap_ = take(a.kind, log=lambda m: print(m, flush=True))
    out_dir = Path(__file__).resolve().parents[1] / "snapshots"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"snap_{snap_['probe_at'].replace(':', '').replace('-', '')}_{a.kind}_screen.json").write_text(
        json.dumps(snap_, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    line = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}] screen kind={a.kind} online={snap_['account_online']} "
            f"cash={snap_['asset'].get('cash')} pos={len(snap_['positions'])} trades={len(snap_['trades'])} "
            f"orders={len(snap_['orders'])} {time.time() - t0:.0f}s err={snap_.get('error', '')[:200]}")
    if a.dry_run:
        print(json.dumps(snap_, ensure_ascii=False, default=str), flush=True)
        print(line, flush=True)
        return 0
    key = Path(a.key_file).read_text(encoding="utf-8").strip().encode()
    status, text = push(snap_, a.url, key)
    print(line + f" push={status} {text[:300]}", flush=True)
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
