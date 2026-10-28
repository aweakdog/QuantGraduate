"""QMT 界面通道测试 (2026-10-26, 用户确认: 实盘跌停价挂 100 股浦发银行买单 -> 确认已报 -> 撤单 -> 确认已撤)。

逐步执行: 每次只跑一个动作 (一次提权计划任务), 截图 + JSON 结果落盘, 操作者看过再跑下一步。
  read <state|dialog|orders|cancel_list>        只读
  nav <side_trade|menu_orders|menu_cancel|menu_buy>  只点导航 (左侧菜单), 没有交易后果
  submit-buy --expect 600000:BUY:8.58:100 --limit-down 8.58
  confirm --expect 600000:BUY:8.58:100           确认框内容逐项一致才点肯定按钮
  cancel-select --expect ... / cancel-submit --expect ... / cancel-confirm --expect ...
  dismiss --title 提示 --button 确定            只限白名单里的信息类弹窗
硬限制 (guard): 只在连续竞价时段; 只允许 买入、100 股、价格 == 跌停价 的通道测试单 (不会成交);
表单/确认框读回必须和 --expect 完全一致, 否则不点。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .ui_agent import ask, click_windows, crop, locate_verified, png_size, screenshot_windows, verify_point

LOG_DIR = Path(r"D:\qmtcode\logs\ui_steps")
SESSIONS = (("09:30:10", "11:28:00"), ("13:00:10", "14:50:00"))
SIDE_CN = {"BUY": "买入", "SELL": "卖出"}
DISMISS_OK = {("提示", "确定"), ("委托提示", "确定"), ("系统提示", "确定"), ("撤单提示", "确定"), ("系统预留信息", "×")}

PROMPTS = {
    "state": ("这是 {w}x{h} 的 QMT 交易终端截图。只返回 JSON: {{logged_in (主界面是否已登录), "
              "trade_online (右下角状态栏「交易」是否绿色在线), available (可用资金), total_asset (总资产), "
              "form: {{side_tab, code, name, price, volume}} (左下下单面板, 没有就 null), "
              "popups (所有弹窗的标题列表), page (当前左侧菜单选中的页面名称)}}。数字转数值, 看不清填 null。"),
    "dialog": ("这是 {w}x{h} 的 QMT 交易终端截图。若屏幕上有弹窗, 逐个读出, 只返回 JSON: "
               "{{dialogs: [{{title, text (弹窗全部文字, 逐字), code (证券代码, 没有填 null), name, "
               "side (买入/卖出/撤单, 没有填 null), price (数值或 null), volume (数值或 null), "
               "order_id (合同号/委托编号, 没有填 null), buttons (按钮文字列表)}}]}}。没有弹窗就 dialogs=[]。"),
    "orders": ("这是 {w}x{h} 的 QMT 交易终端截图。读出当日委托列表 (表格) 的每一行, 只返回 JSON: "
               "{{page (当前页面名称), rows: [{{time, code, name, side, price, volume, traded_volume, status, "
               "order_id}}]}}。没有行就 rows=[]。数字转数值, 看不清填 null。"),
    "cancel_list": ("这是 {w}x{h} 的 QMT 交易终端截图, 当前应在「撤单」页面。读出可撤委托列表每一行, 只返回 JSON: "
                    "{{page, rows: [{{code, name, side, price, volume, status, order_id, selected (该行是否被勾选/选中)}}], "
                    "buttons (页面上与撤单相关的按钮文字)}}。"),
}
NAV = {"side_trade": ("最左侧竖排导航栏里的「交易」图标或文字", "交易"),
       "menu_orders": ("左侧功能菜单「查询」下面的「委托」", "委托"),
       "menu_cancel": ("左侧功能菜单从上往下第 4 项「撤单」(弯曲的返回箭头图标; 在「转股回售」下面、「撤任务」上面, "
                       "千万不是「转股回售」)", "撤单"),
       "menu_buy": ("左侧功能菜单最上面的「买入」", "买入")}


def in_session(now=None):
    t = (now or datetime.now()).strftime("%H:%M:%S")
    return any(a <= t <= b for a, b in SESSIONS)


def parse_expect(s):
    code, side, price, volume = s.split(":")
    if side not in SIDE_CN:
        raise ValueError(f"方向只能是 BUY/SELL: {side}")
    return {"code": code, "side": SIDE_CN[side], "price": round(float(price), 3), "volume": int(volume)}


def guard_channel_order(exp, limit_down):
    """只允许通道测试单: 买入 100 股、价格等于跌停价 (挂在跌停上几乎不可能成交)。返回 (允许, 原因)。"""
    if exp["side"] != "买入":
        return False, "通道测试只允许买入"
    if exp["volume"] != 100:
        return False, "通道测试只允许 100 股"
    if limit_down is None or abs(exp["price"] - float(limit_down)) > 1e-6:
        return False, f"价格 {exp['price']} 不等于跌停价 {limit_down}"
    return True, "ok"


def guard_repo_order(exp, bid_rate):
    """逆回购测试单: 只允许卖出 (借出) 上交所 GC001 (204001) 10 张 = 1000 元, 利率不高于买一 (才能立即成交),
    也不低于买一 0.2 个百分点以上 (防止填错)。返回 (允许, 原因)。"""
    if exp["code"] != "204001":
        return False, "逆回购测试只允许 204001 (GC001)"
    if exp["side"] != "卖出":
        return False, "逆回购只能「卖出」(借出资金)"
    if exp["volume"] != 10:
        return False, "逆回购测试只允许 10 张 (1000 元)"
    if bid_rate is None or not (bid_rate - 0.2 <= exp["price"] <= bid_rate + 1e-9):
        return False, f"利率 {exp['price']} 不在 [买一-0.2, 买一] = [{None if bid_rate is None else bid_rate - 0.2}, {bid_rate}]"
    return True, "ok"


def matches(found, exp, fields=("code", "side", "price", "volume")):
    """读回内容是否与预期逐项一致。返回 (一致, 不一致项)。"""
    bad = []
    for f in fields:
        v = found.get(f)
        if f == "price":
            ok = v is not None and abs(float(v) - exp["price"]) < 1e-6
        elif f == "volume":
            ok = v is not None and int(float(v)) == exp["volume"]
        elif f == "side":
            ok = v is not None and exp["side"] in str(v)
        else:
            ok = str(v) == exp[f]
        if not ok:
            bad.append(f"{f}: 读到 {v!r}, 期望 {exp[f]!r}")
    return not bad, bad


def snap(step):
    png = screenshot_windows()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{step}.png"
    path.write_bytes(png)
    return png, path


def read(kind, step=None):
    png, path = snap(step or f"read_{kind}")
    w, h = png_size(png)
    res, meta = ask(png, PROMPTS[kind].format(w=w, h=h))
    return res, {"shot": str(path), "latency_s": meta["latency_s"]}


def click(target, expect, step):
    png, path = snap(step + "_before")
    loc, ok, trail = locate_verified(png, target, expect)
    out = {"target": target, "ok": ok, "xy": (loc.get("x"), loc.get("y")), "trail": trail[-1], "before": str(path)}
    if ok:
        click_windows(loc["x"], loc["y"])
        time.sleep(1.5)
        out["after"] = str(snap(step + "_after")[1])
    return out


def submit_buy(exp, limit_down):
    if not in_session():
        return {"ok": False, "why": "不在连续竞价时段"}
    allowed, why = guard_channel_order(exp, limit_down)
    if not allowed:
        return {"ok": False, "why": why}
    st, meta = read("state", "submit_check")
    form = {**(st.get("form") or {}), "side": (st.get("form") or {}).get("side_tab")}
    same, bad = matches(form, exp)
    if not same:
        return {"ok": False, "why": "表单读回不一致", "bad": bad, "state": st, **meta}
    blocking = [p for p in st.get("popups") or [] if "系统预留信息" not in str(p)]
    if blocking:
        return {"ok": False, "why": f"有弹窗挡着: {blocking}", **meta}
    res = click("左下角下单面板最底部橙色的「买入」提交按钮 (不是左侧竖排的「买入」页签, 也不是左侧菜单)", "买入", "submit_buy")
    if res["ok"]:
        res["dialogs"] = read("dialog", "after_submit")[0]
    return res


SELL_TAB = (262, 587)     # 10-27 量出: 下单面板左侧竖排「卖出」页签 (竖排字, 放大后常只读出「卖」)
CODE_BOX = (545, 505)     # 下单面板右上「证券名称/代码」区, 点它可输入代码


def tab_sell():
    return click_xy_verified(*SELL_TAB, "下单面板左侧竖排的「卖出」页签", "", "tab_sell")   # 竖排字只读得出一个字; 纯导航


def type_code(code):
    """点证券代码区 -> 清空 -> 只敲数字。不按回车; 之后截图看是否出现下拉候选。"""
    from .ui_order import backspace, select_all, type_digits
    out = click_xy_checked(*CODE_BOX, "type_code")
    time.sleep(0.5)
    select_all()
    backspace()
    type_digits(code)
    time.sleep(2.0)
    out["after_typing"] = str(snap("type_code_typed")[1])
    return out


CANDIDATE_ROW = (633, 554)   # 输入代码后下拉候选的第一行 (10-27 量出)


def pick_candidate(code):
    """双击下拉候选第一行 (10-27: 单击不生效; 不用回车选 —— 焦点万一不在代码框, 回车可能等于提交)。之后截图核对。"""
    png, before = snap("pick_candidate_before")
    ok, v = verify_point(png, *CANDIDATE_ROW, f"代码下拉候选列表里「{code}」那一行", "")   # 同一行有代码和名称
    out = {"ok": ok, "xy": CANDIDATE_ROW, "verify": v, "before": str(before)}
    if ok:
        click_windows(*CANDIDATE_ROW)
        time.sleep(0.08)
        click_windows(*CANDIDATE_ROW)
        time.sleep(1.5)
        out["after_pick"] = str(snap("pick_candidate_done")[1])
    return out


def fill_sell(code, price, volume):
    """切到「卖出」页签 -> 输入证券代码 -> 填价格/数量 -> 读回。只填不交, 键盘只敲数字和小数点。"""
    from .ui_order import backspace, read_form, select_all, set_field, type_digits
    log = []
    steps = [click("左下角下单面板左侧竖排的「卖出」页签 (不是顶部菜单, 也不是底部的卖出提交按钮)", "卖出", "fill_sell_tab")]
    time.sleep(1.0)
    c = click("左下角下单面板右侧显示证券名称/代码的输入框 (写着「请输入股票代码」或当前证券代码)", "", "fill_sell_code")
    steps.append(c)
    if not c["ok"]:
        return {"ok": False, "why": "找不到证券代码输入框", "steps": steps}
    time.sleep(0.3)
    select_all()
    backspace()
    type_digits(code)
    time.sleep(2.5)
    set_field("下单面板里「委托价」或「价格」右边的数字输入框 (不是 +/- 按钮)", "", price, log.append)
    set_field("下单面板里「总量」或「数量」右边的数字输入框 (不是 +/- 按钮)", "", str(volume), log.append)
    time.sleep(1.0)
    form = read_form()
    return {"ok": True, "form": form, "steps": steps, "log": log}


SUBMIT_BTN_BOX = (515, 805, 610, 865)   # 下单面板提交按钮所在区域 (买入橙/卖出绿; 逆回购时按钮略下移)


def side_by_button():
    """看提交按钮的颜色判断当前是买入还是卖出 (10-27: 模型把竖排页签读错过, 方向是逆回购最要命的字段)。"""
    from PIL import ImageGrab
    im = ImageGrab.grab(bbox=SUBMIT_BTN_BOX, all_screens=True).convert("RGB")
    px = [im.getpixel((i, j)) for i in range(0, im.width, 2) for j in range(0, im.height, 2)]
    green = sum(1 for r, g, b in px if g > 140 and r < 90 and b < 90)
    orange = sum(1 for r, g, b in px if r > 200 and 50 < g < 150 and b < 70)
    n = len(px)
    side = "卖出" if green > 0.2 * n and green > 3 * orange else "买入" if orange > 0.2 * n and orange > 3 * green else None
    return side, {"green": green, "orange": orange, "n": n}


PRICE_BOX_REPO = (367, 752)   # 10-27 量出: 选中 GC001 后「委托价」输入框 (逆回购面板比股票面板靠上)


def set_price(price):
    """点量好的委托价框 (复核十字在输入框上) -> 全选 -> 退格 -> 只敲数字和小数点。不按回车。"""
    from .ui_order import backspace, select_all, type_digits
    out = click_xy_verified(*PRICE_BOX_REPO, "下单面板里「委托价」右边的数字输入框", "", "set_price",
                            anchor=(298, 752, "委托价"))
    if not out["ok"]:
        return out
    time.sleep(0.3)
    select_all()
    backspace()
    type_digits(price)
    time.sleep(0.8)
    out["typed"] = str(snap("set_price_typed")[1])
    return out


CODE_AREA = (520, 495, 720, 535)      # 10-27 量出: 下单面板右上「名称/代码」两行 (小字, 整屏读会把 204001 读成 60001)
QTY_AREA = (270, 735, 470, 890)       # 委托价 / 最大可卖 / 总量 / 预估金额
SELL_BTN = (563, 843)                 # 选中 GC001 后的绿色「卖出」提交按钮
ZOOM_PROMPT = ("这是交易软件下单面板一小块的放大图。只返回 JSON: {{code (6位证券代码, 没有填 null), name, price (委托价), "
               "volume (总量), est_amount (预估金额)}}。图里没有的字段填 null, 数字转数值, 逐字读, 不要猜。")


def read_form_zoomed():
    """代码区、价量区各自裁剪放大读; 3/4 倍各读一次, (代码, 价, 量, 预估金额) 一致才返回, 否则 5 倍再读取多数。
    10-27: 同一张图 3 倍读出预估金额 1000、4 倍读出 501000, 所以预估金额也要参与比对。"""
    png = screenshot_windows()

    def one(scale):
        r = {}
        for box in (CODE_AREA, QTY_AREA):
            x0, y0, x1, y1 = box
            part, _ = crop(png, (x0 + x1) / 2, (y0 + y1) / 2, w=x1 - x0, h=y1 - y0, scale=scale, mark=False)
            got, _ = ask(part, ZOOM_PROMPT)
            r.update({k: v for k, v in got.items() if v is not None})
        code = str(r.get("code") or "")[:6]
        if not code and str(r.get("name") or "").strip().upper() == "GC001":
            code = "204001"     # 上交所 1 天国债回购: 名称与代码一一对应 (代码那行小字常读不出, 名称稳定)
        return {"code": code, "price": r.get("price"), "volume": r.get("volume"), "est_amount": r.get("est_amount"),
                "name": r.get("name")}

    def key(x):
        return json.dumps({k: x[k] for k in ("code", "price", "volume", "est_amount")}, sort_keys=True, default=str)
    reads = [one(3), one(4)]
    if key(reads[0]) != key(reads[1]):
        reads.append(one(5))
    from collections import Counter
    top, n = Counter(key(x) for x in reads).most_common(1)[0]
    return (next(x for x in reads if key(x) == top) if n >= 2 else None), reads


def submit_sell_repo(exp, bid_rate):
    if not in_session():
        return {"ok": False, "why": "不在连续竞价时段"}
    allowed, why = guard_repo_order(exp, bid_rate)
    if not allowed:
        return {"ok": False, "why": why}
    side, px = side_by_button()
    if side != "卖出":
        return {"ok": False, "why": f"提交按钮颜色判断方向为 {side} (不是卖出)", "pixels": px}
    zoomed, reads = read_form_zoomed()
    if zoomed is None:
        return {"ok": False, "why": "放大读回两次不一致", "reads": reads}
    form = {**zoomed, "side": side}                            # 方向用按钮颜色, 不用模型读的页签
    same, bad = matches(form, exp)
    if not same:
        return {"ok": False, "why": "表单读回不一致", "bad": bad, "reads": reads}
    if zoomed.get("est_amount") is not None and abs(float(zoomed["est_amount"]) - exp["volume"] * 100) > 0.01:
        return {"ok": False, "why": f"预估金额 {zoomed['est_amount']} ≠ {exp['volume']} 张 × 100", "reads": reads}
    res = click_xy_verified(*SELL_BTN, "下单面板的绿色「卖出」提交按钮", "卖", "submit_sell")   # 「卖 出」中间有空格
    res["reads"] = reads
    if res["ok"]:
        res["dialogs"] = read("dialog", "after_submit_sell")[0]
    return res


def confirm(exp, kind="order"):
    if not in_session():
        return {"ok": False, "why": "不在连续竞价时段"}
    d, meta = read("dialog", f"confirm_{kind}_check")
    # 10-26 实测: 同一弹窗的标题两次读出不同的字 (系统预留信息 -> 系统预警信息, 下单确认 -> 下达任务),
    # 标题不可靠; 只认带证券代码的弹窗 (委托/撤单确认), 信息类通知不算, 且必须正好一个。
    dialogs = [x for x in d.get("dialogs") or [] if x.get("code")]
    if len(dialogs) != 1:
        return {"ok": False, "why": f"期望正好 1 个带委托内容的确认框, 读到 {len(dialogs)} 个", "dialogs": d.get("dialogs"), **meta}
    dlg = dialogs[0]
    fields = ("code", "price", "volume") if kind == "cancel" else ("code", "side", "price", "volume")
    same, bad = matches(dlg, exp, fields)
    if not same:
        return {"ok": False, "why": "确认框内容不一致", "bad": bad, "dialog": dlg, **meta}
    yes = next((b for b in dlg.get("buttons") or [] if b in ("确定", "是", "确认", "确认下单", "确定撤单")), None)
    if not yes:
        return {"ok": False, "why": f"找不到肯定按钮: {dlg.get('buttons')}", "dialog": dlg, **meta}
    res = click(f"显示证券代码 {exp['code']}、价格 {exp['price']}、数量 {exp['volume']} 的确认弹窗底部的「{yes}」按钮", yes,
                f"confirm_{kind}")
    res["dialog"] = dlg
    if res["ok"]:
        res["after_dialogs"] = read("dialog", f"after_confirm_{kind}")[0]
    return res


def cancel_select(exp):
    lst, meta = read("cancel_list", "cancel_select_check")
    rows = [r for r in lst.get("rows") or [] if matches({**r}, exp)[0]]
    if len(rows) != 1:
        return {"ok": False, "why": f"可撤列表里匹配的委托有 {len(rows)} 条", "list": lst, **meta}
    r = rows[0]
    return click(f"撤单页面表格里 证券代码 {exp['code']}、价格 {exp['price']}、数量 {exp['volume']} 的那一行 (点该行的勾选框或代码单元格)",
                 exp["code"], "cancel_select") | {"row": r}


def click_xy_checked(x, y, step):
    """固定坐标点击 (只用于关闭已知的信息类通知), 前后截图供人工核对。"""
    before = snap(step + "_before")[1]
    click_windows(x, y)
    time.sleep(1.5)
    return {"ok": True, "xy": (x, y), "before": str(before), "after": str(snap(step + "_after")[1])}


def ocr_anchor(png, x, y, text, rx=80, ry=20, tol=25):
    """本地 OCR 核对固定位置的文字 (不联网、结果确定): (x,y) 附近要正好有一处包含 text 的文字, 且其中心离 (x,y)
    不超过 tol 像素。返回 (True/False/None, 说明); None = 这里用不了 OCR (没装, 或没有可核对的文字)。"""
    from . import ocr_read
    if not text or not ocr_read.available():
        return None, "不用 OCR"
    try:
        c = ocr_read.locate_text(png, text, (max(0, x - rx), max(0, y - ry), x + rx, y + ry), 3)
    except Exception as e:  # noqa: BLE001
        return None, f"OCR 出错 {type(e).__name__}"
    if c is None:
        return False, f"附近没有唯一的「{text}」"
    d = ((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5
    return d <= tol, f"OCR 看到「{text}」偏 {d:.0f}px"


def click_xy_verified(x, y, target, expect, step, anchor=None):
    """量好的坐标 + 核对后才点。先用本地 OCR 核对 (点击处的文字 expect, 或给定的旁边固定标签 anchor=(x, y, 文字),
    用于输入框这类本身没有固定文字的目标); OCR 确认了就直接点, 确认不了才退回模型放大复核 (10-28 起)。"""
    png, before = snap(step + "_before")
    ax, ay, atext = anchor or (x, y, expect)
    a_ok, a_why = ocr_anchor(png, ax, ay, atext)
    if a_ok:
        ok, v = True, {"by": "ocr", "why": a_why}
    else:
        ok, v = verify_point(png, x, y, target, expect)
        v = {**v, "by": "model", "ocr": a_why}
    out = {"ok": ok, "xy": (x, y), "verify": v, "before": str(before)}
    if ok:
        click_windows(x, y)
        time.sleep(1.5)
        out["after"] = str(snap(step + "_after")[1])
    return out


TICK_COLUMN = (860, 890, 530, 720)          # 「委托」表格「选择」列的屏幕范围 (10-26 截图量出), 只许点这里


def tick_row(x, y):
    x0, x1, y0, y1 = TICK_COLUMN
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return {"ok": False, "why": f"({x},{y}) 不在委托表「选择」列范围内"}
    out = click_xy_checked(x, y, "tick_row")
    png = screenshot_windows()
    w, h = png_size(png)
    res, _ = ask(png, (f"这是 {w}x{h} 的 QMT 截图。看下方「委托」表格, 只返回 JSON: {{rows: [{{code, price, volume, status, "
                       "order_id, ticked (该行最左侧「选择」列的勾选框是否已勾上)}]}"))
    out["rows"] = res.get("rows")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["read", "nav", "submit-buy", "confirm", "cancel-select", "cancel-submit",
                                     "cancel-confirm", "dismiss", "close-notice", "tick-row", "cancel-selected",
                                     "cancel-prompt-ok", "fill-sell", "submit-sell-repo", "tab-sell", "type-code",
                                     "pick-candidate", "set-price", "side"])
    ap.add_argument("--code")
    ap.add_argument("--price")
    ap.add_argument("--volume", type=int)
    ap.add_argument("--bid", type=float)
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--expect")
    ap.add_argument("--limit-down", type=float)
    ap.add_argument("--title")
    ap.add_argument("--button")
    a = ap.parse_args(argv)
    exp = parse_expect(a.expect) if a.expect else None
    if a.step == "read":
        res, meta = read(a.arg)
        out = {"result": res, "meta": meta}
    elif a.step == "nav":
        out = click(*NAV[a.arg], f"nav_{a.arg}")
    elif a.step == "submit-buy":
        out = submit_buy(exp, a.limit_down)
    elif a.step == "confirm":
        out = confirm(exp, "order")
    elif a.step == "cancel-select":
        out = cancel_select(exp)
    elif a.step == "cancel-submit":
        out = click("撤单页面上的「撤单」按钮 (不是「全撤」「撤买」「撤卖」)", "撤单", "cancel_submit")
        if out["ok"]:
            out["dialogs"] = read("dialog", "after_cancel_submit")[0]
    elif a.step == "cancel-confirm":
        out = confirm(exp, "cancel")
    elif a.step == "tick-row":
        x, y = (int(v) for v in a.arg.split(","))
        out = tick_row(x, y)
    elif a.step == "cancel-selected":
        x, y = (int(v) for v in (a.arg or "1096,499").split(","))       # 10-26 截图量出
        out = click_xy_verified(x, y, "委托表格上方工具栏的「选中撤单」按钮", "撤单", "cancel_selected")  # 放大后常只读出「撤单」
        if out["ok"]:
            out["dialogs"] = read("dialog", "after_cancel_selected")[0]
    elif a.step == "tab-sell":
        out = tab_sell()
    elif a.step == "type-code":
        out = type_code(a.code)
    elif a.step == "set-price":
        out = set_price(a.price)
    elif a.step == "side":
        sd, pxs = side_by_button()
        out = {"ok": sd is not None, "side": sd, "pixels": pxs}
    elif a.step == "pick-candidate":
        out = pick_candidate(a.code)
    elif a.step == "fill-sell":
        out = fill_sell(a.code, a.price, a.volume)
    elif a.step == "submit-sell-repo":
        out = submit_sell_repo(exp, a.bid)
    elif a.step == "cancel-prompt-ok":
        # 「选中撤单」后的通用确认框 (标题「提示」, 正文「是否撤单」, 不含委托内容): 只在正文含「撤单」时点「确定」。
        # 参数只用 ASCII (中文经批处理传参会被 GBK 读坏, 10-26 踩过)。
        d, meta = read("dialog", "cancel_prompt_check")
        hits = [x for x in d.get("dialogs") or [] if "撤单" in str(x.get("text")) and "确定" in (x.get("buttons") or [])]
        if len(hits) != 1:
            out = {"ok": False, "why": f"期望正好 1 个「是否撤单」确认框, 读到 {len(hits)} 个", "dialogs": d.get("dialogs"), **meta}
        else:
            out = click("正文为「是否撤单」的提示框里的「确定」按钮", "确定", "cancel_prompt_ok")
            if out["ok"]:
                out["after_dialogs"] = read("dialog", "after_cancel_prompt")[0]
    elif a.step == "close-notice":
        out = click_xy_checked(1310, 255, "close_notice")      # 10-26 截图量出: 「系统预留信息」弹窗 × 的位置
    else:
        if (a.title, a.button) not in DISMISS_OK:
            out = {"ok": False, "why": f"「{a.title}」/「{a.button}」不在白名单"}
        else:
            out = click(f"标题为「{a.title}」的弹窗里的「{a.button}」按钮", "" if a.button == "×" else a.button, "dismiss")
    print(json.dumps({"step": a.step, "at": datetime.now().isoformat(timespec="seconds"), **out}, ensure_ascii=False, default=str),
          flush=True)
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
