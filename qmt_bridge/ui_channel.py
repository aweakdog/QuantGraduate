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

from .ui_agent import ask, click_windows, locate_verified, png_size, screenshot_windows, verify_point

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


def click_xy_verified(x, y, target, expect, step):
    """量好的坐标 + 独立复核: 放大该点, 十字下必须正是预期按钮才点 (密集工具栏里模型定位会偏一格)。"""
    png, before = snap(step + "_before")
    ok, v = verify_point(png, x, y, target, expect)
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
                                     "cancel-prompt-ok"])
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
