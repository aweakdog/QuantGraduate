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
TAB_STRIP = (780, 460, 1300, 488)     # 底部表格区的页签栏; 页签按 OCR 找到的文字位置点 (10-30: 选中的「持仓」偏了 29px)
HEADER_KEY = {"positions": "当前拥股", "trades": "成交时间", "orders": "委托剩余量"}   # 切页签后表头里必须有它
TABLE_BOX = (780, 482, 1920, 738)
TOP_TAB_STOCK = (110, 51, "股票交易")      # 10-27 量出: 顶部页签栏最左边; 那天早上停在「沪港通」, 三张表和资金表全空
EXPECTED_ACCOUNT_LAST4 = os.environ.get("QMT_ACCOUNT_LAST4", "6735")
FUNDS_BOX = (780, 742, 1920, 830)

SPACE_HINT = ("注意: 这个软件用空格分隔千位, 例如「50 3121.80」是 503121.80、「43 4000.00」是 434000.00, "
              "读数时把空格去掉再转数值, 不要丢位。")
FUNDS_PROMPT = ("这是 QMT「账号资金」表格的放大截图。只返回 JSON: {account_last4 (资金账号后4位), total_asset (总资产), "
                "available (可用金额), frozen (冻结金额), market_value (总市值), withdrawable (可取金额), "
                "login_status (登录状态那一列的文字)}。数字转数值, 看不清填 null, 不要猜。" + SPACE_HINT)
TABLE_PROMPTS = {
    "positions": ("这是 QMT「持仓」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{code (6位证券代码), name, "
                  "volume (持仓数量/当前拥股), can_use (可用数量), cost (成本价), last (最新价), market_value (市值)}]}。"
                  "表格为空就 rows=[]。数字转数值, 看不清填 null, 不要猜, 不要把别的列错填进来。" + SPACE_HINT),
    "trades": ("这是 QMT「成交」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{time (成交时间), code (6位证券代码), "
               "side (买入/卖出), price (成交价格/成交均价), volume (成交数量), amount (成交金额), order_id (合同编号), "
               "trade_id (成交编号)}]}。表格为空就 rows=[]。数字转数值, 看不清填 null, 不要把别的列错填进来。" + SPACE_HINT),
    "orders": ("这是 QMT「委托」表格的放大截图。逐行读出, 只返回 JSON: {rows: [{time (委托时间), code (6位证券代码), "
               "side (买入/卖出), price (委托价格), volume (委托量), traded_volume (成交数量, 注意不是委托剩余量), "
               "status (委托状态), order_id (合同编号, 注意不是资金账号)}]}。表格为空就 rows=[]。" + SPACE_HINT),
}


def xt_code(code6):
    """补交易所后缀。上交所: 5/6/9 开头、204 逆回购、11 可转债、01/02 国债; 北交所: 4/8 开头; 其余深交所。"""
    c = str(code6)[:6]
    sh = c[:1] in ("5", "6", "9") or c.startswith(("204", "11", "01", "02", "888"))     # 888880 = 沪市新标准券
    return c + (".SH" if sh else ".BJ" if c[:1] in ("4", "8") else ".SZ")


def side_type(side):
    s = str(side or "")
    if "逆回购" in s:
        return 24          # 逆回购 = 卖出 (借出资金); 成交表里写的是「通用回购逆回购」
    return 23 if "买" in s else 24 if "卖" in s else None


def is_repo_row(row):
    """逆回购相关的行: 204xxx/1318xx, 标准券 888880/131990, 或方向写着「回购」。价格是年化利率, 金额 = 张数 x 100。"""
    c = str(row.get("code") or "")[:6]
    return c.startswith(("204", "1318")) or c in ("888880", "131990") or "回购" in str(row.get("side") or "")


def _num(v):
    try:
        return round(float(str(v).replace(",", "").replace(" ", "").rstrip("股张元").strip()), 3)
    except (TypeError, ValueError):
        return None


def canon_rows(kind, rows):
    """规范化后用于比对: 数字统一两位小数、代码 6 位、按 (代码, 时间, 价, 量) 排序。"""
    # 持仓不比「市值」列: 10-29 QMT 重启后列变窄, 「50 2000.00」只显示「50 2」; 市值改由数量算 (position_value)
    keys = {"positions": ("code", "volume", "can_use", "cost", "last"),
            "trades": ("time", "code", "side", "price", "volume", "amount"),
            "orders": ("time", "code", "side", "price", "volume", "traded_volume", "status")}[kind]
    # 方向按买/卖代码比 (「卖出」与「通用回购逆回购」是同一方向, 两家模型写法不同); 合同号不比 (对账不用, 且常被读成资金账号)
    # 逆回购/标准券行只比 代码、张数、金额 (利率、时间、成本两家常读得不一样, 且这些行按现金处理不进对账; 10-28 因此报过离线)
    repo_keys = {"positions": ("code", "volume"), "trades": ("code", "volume", "amount"),
                 "orders": ("code", "volume", "traded_volume", "status")}[kind]
    out = []
    for r in rows or []:
        c = {}
        for k in (repo_keys if is_repo_row(r) else keys):
            v = r.get(k)
            if k == "code":
                c[k] = str(v)[:6]
            elif k == "side":
                c[k] = side_type(v)
            elif k in ("volume", "can_use", "cost", "market_value", "price", "amount", "traded_volume", "last"):
                c[k] = _num(v)
            elif k == "time":
                c[k] = str(v).strip()[:5] if v is not None else None     # 只比到分钟, 秒数常读错且对账不用
            else:
                c[k] = str(v).strip() if v is not None else None
        out.append(c)
    return sorted(out, key=lambda c: json.dumps(c, ensure_ascii=False, sort_keys=True))


def canon_funds(f):
    return {k: _num(f.get(k)) for k in ("total_asset", "available", "frozen", "market_value")}


def agree(reads, canon):
    """多次读的规范化结果: 两次一致即采用; 否则取多数; 无多数返回 None。"""
    cs = [json.dumps(canon(r), ensure_ascii=False, sort_keys=True) for r in reads]
    top, n = Counter(cs).most_common(1)[0]
    return (json.loads(top) if n >= 2 else None), cs


def read_box(png, box, prompt, scale, provider=None, fallbacks=None):
    x0, y0, x1, y1 = box
    part, _ = crop(png, (x0 + x1) / 2, (y0 + y1) / 2, w=x1 - x0, h=y1 - y0, scale=scale, mark=False)
    res, _ = ask(part, prompt, provider=provider, fallbacks=fallbacks)
    return res


# 有 OCR 时依次请这几家附和, 有一家与 OCR 一致就停 (10-28: 同一张图上 OCR/MiniMax/千问都读对「43 4000.00」,
# DeepSeek 每次都读成 43400, 所以它排最后)
BACKERS = (("minimax", 3), ("qwen", 3), ("deepseek", 2))


def read_agreed(png, box, prompt, canon, scales=(2, 3, 2.5), ocr_kind=None):
    """本地 OCR 读一遍 + 一家模型独立读一遍, 一致才采用; 不一致再让 DeepSeek 读一次取多数 (10-28 起)。
    OCR 不可用 (没装、认不出表头、出错) 时退回两家模型交叉核对: DeepSeek + MiniMax (10-27 MiniMax 把「50 3121.80」
    读成 50312.8, DeepSeek 把资金账号当合同号, 所以要两家)。"""
    from . import ocr_read
    from .ui_agent import _has_key
    ocr = None
    if ocr_kind and ocr_read.available():
        try:
            ocr = ocr_read.read_table(png, box, ocr_kind)
        except Exception:  # noqa: BLE001 —— OCR 出错就当没读到, 由模型兜底
            ocr = None
    if ocr is not None:
        reads, sources, val = [ocr], ["ocr"], None
        for prov, sc in BACKERS:
            if not _has_key(prov):
                continue
            try:
                reads.append(read_box(png, box, prompt, sc, prov, fallbacks=()))
            except Exception:  # noqa: BLE001 —— 这一家挂了就问下一家
                continue
            sources.append(prov)
            val, _ = ocr_backed(reads, canon)
            if val is not None:
                break
        return val, {"reads": len(reads), "agreed": val is not None, "sources": sources}, reads
    second = "minimax" if _has_key("minimax") else None
    reads = [read_box(png, box, prompt, scales[0], "deepseek"), read_box(png, box, prompt, scales[1], second)]
    sources = ["deepseek", second or "deepseek"]
    val, cs = agree(reads, canon)
    if val is None:
        reads.append(read_box(png, box, prompt, scales[2], "deepseek"))
        sources.append("deepseek")
        val, cs = agree(reads, canon)
    return val, {"reads": len(reads), "agreed": val is not None, "sources": sources}, reads


def ocr_backed(reads, canon):
    """有 OCR 读数时, 必须有一家模型与 OCR 一致才采用; 两家模型联手推翻 OCR 不算 (10-28: 「43 4000.00」两家模型
    都读成 43400, OCR 读对 434000, 按多数会采用错的)。reads[0] 是 OCR。"""
    cs = [json.dumps(canon(r), ensure_ascii=False, sort_keys=True) for r in reads]
    return (json.loads(cs[0]) if cs[0] in cs[1:] else None), cs


def position_value(p):
    """持仓市值按数量算, 不信「市值」列 (10-29: 列变窄后「50 2000.00」只显示「50 2」, OCR 和模型都读成 502):
    逆回购标准券 = 张数 x 100; 股票 = 股数 x 最新价; 两者都没有才用读到的市值。"""
    vol = p.get("volume") or 0
    if is_repo_row(p):
        return round(vol * 100.0, 2)
    if p.get("last") is not None:
        return round(vol * float(p["last"]), 2)
    return p.get("market_value") or 0


def checks(funds, positions, trades, orders):
    """算术交叉核对, 返回问题列表 (空 = 通过)。"""
    probs = []
    t, a, fr, mv = (funds.get(k) for k in ("total_asset", "available", "frozen", "market_value"))
    if None in (t, a, mv):
        probs.append(f"资金读数不全: {funds}")
    elif abs(t - (a + (fr or 0) + mv)) > max(1.0, 0.002 * t):
        probs.append(f"可用 {a} + 冻结 {fr} + 市值 {mv} ≠ 总资产 {t}")
    if positions and mv is not None:
        s = sum(position_value(p) for p in positions)
        if abs(s - mv) > max(1.0, 0.005 * mv):
            probs.append(f"持仓市值合计 {s:.2f} ≠ 总市值 {mv}")
    for p in positions:
        if closed_row(p):       # 当天清仓/逆回购到期后留下的 0 股 0 市值行 (10-28 开盘: 标准券 888880 到期留了一行)
            continue
        if not p.get("volume") or p.get("volume") % 1 or (p.get("can_use") or 0) > p["volume"]:
            probs.append(f"持仓行异常: {p}")
    for tr in trades:
        if is_repo_row(tr):                 # 逆回购: 金额 = 张数 x 100 (价格是利率, 不能乘)
            if tr.get("volume") is None or tr.get("amount") is None or abs(tr["volume"] * 100 - tr["amount"]) > 0.05:
                probs.append(f"逆回购成交 张数×100 ≠ 金额: {tr}")
        elif None in (tr.get("price"), tr.get("volume"), tr.get("amount")) or side_type(tr.get("side")) is None:
            probs.append(f"成交行不全: {tr}")
        elif abs(tr["price"] * tr["volume"] - tr["amount"]) > max(0.05, 0.002 * tr["amount"]):
            probs.append(f"成交 价×量 {tr['price']}×{tr['volume']} ≠ 金额 {tr['amount']}")
    for o in orders:
        if o.get("traded_volume") is not None and o.get("volume") is not None and o["traded_volume"] > o["volume"]:
            probs.append(f"委托成交量大于委托量: {o}")
    return probs


ERROR_KINDS = (  # (类别, 判断关键词) —— 按顺序匹配, 给 041 的邮件挑一句人话
    ("exception", ("读屏程序异常",)),
    ("login", ("登录失败", "未登录", "交易不在线")),
    ("window", ("分辨率", "主窗口")),
    ("layout", ("页签失败", "交易页失败", "不是预期的")),
    ("read", ("不一致", "≠", "读数不全", "异常:", "大于委托量")),
)


def classify(problems):
    text = " ".join(problems or [])
    return next((k for k, words in ERROR_KINDS if any(w in text for w in words)), "other") if problems else None


def qmt_version(root=None):
    """QMT 客户端版本 (resource/version 第一行); 读不到返回 None。"""
    try:
        root = root or next(p for p in Path("D:/").iterdir() if (p / "bin.x64" / "XtItClient.exe").exists())
        line = (Path(root) / "resource" / "version").read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        return line.split(":", 1)[1].strip()
    except (StopIteration, OSError, IndexError):
        return None


def version_note(current, seen_file=Path(r"D:\qmtcode\qmt_version_seen.txt")):
    """QMT 自动升级后提醒一次 (界面可能变)。返回提示文字或 None, 并记下新版本。"""
    if not current:
        return None
    old = seen_file.read_text(encoding="utf-8").strip() if seen_file.exists() else None
    if old == current:
        return None
    seen_file.write_text(current, encoding="utf-8")
    return None if old is None else f"QMT 客户端版本 {old} -> {current}"


def closed_row(p):
    """0 股且 0 市值的持仓行 = 已清仓/已到期, 不算持仓。股数读不出 (None) 或市值不为 0 都不算。"""
    return p.get("volume") == 0 and not p.get("market_value")


def build_snapshot(kind, funds, positions, trades, orders, account_last4, problems, meta):
    positions = [p for p in positions if not closed_row(p)]
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
                           "market_value": position_value(p)} for p in positions],
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
        snap["error_kind"] = classify(problems)
    return snap


def click_tab(label, fallback_xy, step, click_xy_verified):
    """先用 OCR 在页签栏里找这个字, 找到唯一一处就点它的中心 (确定、不联网); 找不到才退回固定坐标 + 复核。"""
    from . import ocr_read
    from .ui_agent import click_windows, screenshot_windows
    from .ui_channel import CLICK_SETTLE
    if ocr_read.available():
        try:
            xy = ocr_read.locate_text(screenshot_windows(), label, TAB_STRIP, 2)
        except Exception:  # noqa: BLE001
            xy = None
        if xy:
            click_windows(*xy)
            time.sleep(CLICK_SETTLE)
            return {"ok": True, "xy": xy, "verify": {"by": "ocr-locate"}}
    return click_xy_verified(*fallback_xy, f"底部表格区的「{label}」页签", label, step)


def header_ok(png, kind):
    """切页签后核对表头 (只在有 OCR 时做; 没 OCR 时由模型读数时自行判断)。"""
    from . import ocr_read
    if not ocr_read.available():
        return True
    try:
        boxes = ocr_read.ocr_boxes(png, (TABLE_BOX[0], TABLE_BOX[1], TABLE_BOX[2], TABLE_BOX[1] + 60), 2)
    except Exception:  # noqa: BLE001
        return True
    return any(HEADER_KEY[kind] == b[0].strip() for b in boxes)


def take(kind, log=print):
    from .ui_agent import ensure_qmt_foreground, login_normal_mode
    from .ui_channel import click_xy_checked, click_xy_verified, read, snap

    # 先恢复/最大化/切前台; 只有确实没有主窗口时才重启登录 (10-26: 最小化被误判为未登录, 白白重启了一次)
    win_probs = ensure_qmt_foreground(log)
    if any("找不到 QMT 主窗口" in x for x in win_probs):
        log("没有 QMT 主窗口, 自动普通方式登录")
        if not login_normal_mode(log=log):
            return build_snapshot(kind, {}, [], [], [], None, ["QMT 普通方式登录失败"], {})
        time.sleep(5)
        win_probs = ensure_qmt_foreground(log)
    if win_probs:
        return build_snapshot(kind, {}, [], [], [], None, win_probs, {})
    st, _ = read("state", "snap_state")
    if not (st.get("logged_in") and st.get("trade_online")):
        return build_snapshot(kind, {}, [], [], [], None, [f"QMT 未登录或交易不在线: {st}"], {"state": st})
    if st.get("popups"):
        click_xy_checked(1310, 255, "snap_close_notice")       # 华泰信息通知 (只有它会常驻)
    # 每次都点左侧「交易」(幂等), 不信模型报的页面名 (10-26: 「模型交易」含「交易」二字, 误以为已在交易页)
    nav = click_xy_verified(22, 203, "最左侧竖排导航栏的「交易」", "交易", "snap_nav_trade")
    if not nav["ok"]:
        return build_snapshot(kind, {}, [], [], [], None, [f"切到交易页失败 (复核: {nav.get('verify')})"], {"state": st})
    time.sleep(2.0)
    x, y, label = TOP_TAB_STOCK
    top = click_xy_verified(x, y, f"顶部页签栏最左边的「{label}」页签", label, "snap_tab_stock")
    if not top["ok"]:
        return build_snapshot(kind, {}, [], [], [], None, [f"切到「{label}」页签失败 (复核: {top.get('verify')})"], {})
    time.sleep(1.5)
    meta, problems, tables = {"tabs": {}}, [], {}
    from .ui_agent import wait_clear
    over = wait_clear([TABLE_BOX, FUNDS_BOX, TAB_STRIP], log=log)
    if over:
        return build_snapshot(kind, {}, [], [], [], None,
                              [f"QMT 窗口被别的程序挡住: {[o[0] or o[1] for o in over]}"], {"state": st})
    for kind_t in ("positions", "trades", "orders"):
        x, y, label = TABS[kind_t]
        c = click_tab(label, (x, y), f"snap_tab_{kind_t}", click_xy_verified)
        if not c["ok"]:
            problems.append(f"切换「{label}」页签失败 (复核: {c.get('verify')})")
            tables[kind_t] = []
            continue
        time.sleep(1.2)
        png, path = snap(f"snap_{kind_t}")
        if not header_ok(png, kind_t):
            problems.append(f"切到「{label}」后表头里没有「{HEADER_KEY[kind_t]}」 (页签没切过去或布局变了)")
            tables[kind_t] = []
            continue
        val, m, _ = read_agreed(png, TABLE_BOX, TABLE_PROMPTS[kind_t], lambda r, k=kind_t: canon_rows(k, r.get("rows")),
                                ocr_kind=kind_t)
        meta["tabs"][kind_t] = {**m, "shot": str(path)}
        if val is None:
            problems.append(f"「{label}」表多次读数不一致")
        tables[kind_t] = val or []
    png, path = snap("snap_funds")
    funds, m, reads = read_agreed(png, FUNDS_BOX, FUNDS_PROMPT, canon_funds, ocr_kind="funds")
    meta["funds"] = {**m, "shot": str(path)}
    if funds is None:
        problems.append("资金表多次读数不一致")
        funds = {}
    last4 = next((str(r.get("account_last4")) for r in reads if r.get("account_last4")), None)
    if last4 != EXPECTED_ACCOUNT_LAST4:     # 停在沪港通/融资融券等别的页签时资金表是空的或是别的账号
        problems.append(f"资金表账号尾号 {last4} 不是预期的 {EXPECTED_ACCOUNT_LAST4} (可能停在别的页签)")
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
    from .ui_lock import ui_lock
    try:
        with ui_lock(f"screen_snapshot:{a.kind}", wait_s=240):
            snap_ = take(a.kind, log=lambda m: print(m, flush=True))
    except TimeoutError as e:
        snap_ = build_snapshot(a.kind, {}, [], [], [], None, [f"读屏程序异常: 界面被别的程序占用超过 4 分钟 ({e})"], {})
    except Exception as e:  # noqa: BLE001 —— 程序崩了也要推一份离线快照, 让 041 带着原因报警
        snap_ = build_snapshot(a.kind, {}, [], [], [], None, [f"读屏程序异常: {type(e).__name__}: {str(e)[:200]}"], {})
    snap_["qmt_version"] = qmt_version()
    note = version_note(snap_["qmt_version"])
    if note:
        snap_["notes"] = [note]
    out_dir = Path(__file__).resolve().parents[1] / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
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
