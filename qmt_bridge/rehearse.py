"""每日填单演练 (2026-10-28 起): 在 QMT 下单面板里按当天计划逐笔「填好 -> 读回核对 -> 清空」, **永远不点提交**。

  .venv\\Scripts\\python.exe -m qmt_bridge.rehearse [--max 5]
计划: 041 单据里的买卖; 空仓日用排名前几名 (若回场会买的) 当作买单。价格 = min(涨停, 现价×1.02) (与执行器买入一致),
卖出价 = max(跌停, 现价×0.99); 股数 = 线内现金/笔数 ÷ 价格, 取整百。每笔记录用时、OCR 读回与模型读回是否都与预期一致、
提交按钮颜色 (方向) 是否对。结果写 logs/rehearse.log, 推 041 (exec_report mode=rehearsal, 只有出问题才提醒)。
委托价框/总量框不用写死坐标: 先 OCR 找到「委托价」「总量」标签, 输入框在标签右边固定距离处 (股票和逆回购面板行高不同)。
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from datetime import date, datetime

from .ocr_read import parse_panel  # noqa: F401 —— 读回解析在 ocr_read 里, 这里保留同名供测试和调用

PANEL_LABELS = (250, 690, 560, 900)  # 委托价/总量/预估金额 这几个标签所在区域
CODE_AREA = (520, 495, 720, 535)
FORM_AREA = (250, 700, 520, 900)
PRICE_DX, QTY_DX = 70, 62            # 输入框中心 = 标签中心 + dx (10-27/28 量出)


def quote(code):
    """腾讯行情: (名称, 现价, 涨停, 跌停)。"""
    mkt = "sh" if code[0] in "569" else "sz"
    with urllib.request.urlopen(f"https://qt.gtimg.cn/q={mkt}{code}", timeout=10) as r:
        f = r.read().decode("gbk", "replace").split("~")
    return f[1], float(f[3]), float(f[47]), float(f[48])


def plan_orders(ticket, quotes, n=5):
    """把单据变成演练用的委托列表。quotes: {code: (名称, 现价, 涨停, 跌停)}。"""
    out = []
    for s in ticket.get("sell") or []:
        name, last, up, down = quotes[s["code"]]
        out.append({"side": "卖出", "code": s["code"], "name": name, "price": round(max(down, last * 0.99), 2),
                    "shares": int(s["shares"])})
    buys = ticket.get("buy") or []
    if not buys and not out:                                       # 空仓日: 用排名前 n 当作买单
        budget = float(ticket.get("line_cash") or 0) / max(1, n)
        buys = [{"code": c, "budget": budget} for c in (ticket.get("ranked") or [])[:n]]
    for b in buys:
        if b["code"] not in quotes:
            continue
        name, last, up, down = quotes[b["code"]]
        price = round(min(up, last * 1.02), 2)
        shares = int(b.get("shares") or math.floor(float(b.get("budget") or 0) / price / 100) * 100)
        if shares >= 100:
            out.append({"side": "买入", "code": b["code"], "name": name, "price": price, "shares": shares})
    return out


def same(read, o):
    return (read.get("code") == o["code"] and read.get("price") is not None and abs(float(read["price"]) - o["price"]) < 1e-6
            and read.get("volume") is not None and int(float(read["volume"])) == o["shares"])


def panel_code():
    """OCR 读下单面板右上角的 6 位代码。"""
    from .ocr_read import ocr_boxes, text_of
    from .ui_agent import screenshot_windows
    return next(iter(re.findall(r"\d{6}", text_of(ocr_boxes(screenshot_windows(), CODE_AREA, 3)))), None)


def ensure_code(o, tries=2):
    """输入代码、选候选, 再用 OCR 确认面板上就是这只; 不是就重来一次。"""
    from .ui_channel import pick_candidate, type_code
    for _ in range(tries):
        if panel_code() == o["code"]:
            return True
        type_code(o["code"])
        if pick_candidate(o["code"], o.get("name"))["ok"]:
            time.sleep(0.6)
    return panel_code() == o["code"]


def fill_one(o, log=print):
    """填一笔并读回; 不提交。返回记录。"""
    from .ocr_read import locate_text
    from .ui_agent import screenshot_windows
    from .ui_channel import click_xy_verified, read_form_checked, side_by_button, tab_side
    from .ui_order import backspace, select_all, type_digits
    t0, rec = time.time(), {**o, "problems": [], "t": {}}
    last = [t0]

    def lap(name):                                   # 各环节耗时, 找慢在哪
        now = time.time()
        rec["t"][name] = round(now - last[0], 1)
        last[0] = now
    # 先切方向再输代码, 输完用 OCR 确认面板上确实是这只: 切页签时 QMT 会换回该页上次用的股票 (10-28 第一笔变成 600000);
    # 面板还是逆回购 GC001 时点「买入」不生效 (个人不能正回购), 那就先输代码再切一次
    first = tab_side(o["side"])
    lap("tab")
    if not ensure_code(o):
        rec["problems"].append("面板上的代码不是目标股票")
        return {**rec, "seconds": round(time.time() - t0, 1)}
    if not first["ok"]:
        c = tab_side(o["side"])
        if not c["ok"] or not ensure_code(o):
            rec["problems"].append(f"切到「{o['side']}」失败 (按钮方向 {c['verify'].get('side')})")
            return {**rec, "seconds": round(time.time() - t0, 1)}
    lap("code")
    time.sleep(0.5)
    png = screenshot_windows()
    lab_p = locate_text(png, "委托价", PANEL_LABELS, 2)
    lab_q = locate_text(png, "总量", PANEL_LABELS, 2) or locate_text(png, "总里", PANEL_LABELS, 2)
    if not lab_p or not lab_q:
        rec["problems"].append(f"OCR 没找到标签 委托价={lab_p} 总量={lab_q}")
        return {**rec, "seconds": round(time.time() - t0, 1)}
    for (lx, ly), dx, text, val in ((lab_p, PRICE_DX, "委托价", f"{o['price']:.2f}"),
                                    (lab_q, QTY_DX, "总量", str(o["shares"]))):
        r = click_xy_verified(lx + dx, ly, f"下单面板里「{text}」右边的数字输入框", "", f"rh_{text}_{o['code']}",
                              anchor=(lx, ly, "总" if text == "总量" else text))
        if not r["ok"]:
            rec["problems"].append(f"点不到{text}框")
            return {**rec, "seconds": round(time.time() - t0, 1)}
        time.sleep(0.3)
        select_all()
        backspace()
        type_digits(val)
        time.sleep(0.5)
    lap("fill")
    est = locate_text(screenshot_windows(), "预估金额", PANEL_LABELS, 2)
    if est:                                                   # 点一下标签让输入框失焦, 预估金额才刷新
        from .ui_agent import click_windows
        click_windows(*est)
        time.sleep(1.0)
    lap("blur")
    form, reads = read_form_checked(qty_area=FORM_AREA)
    lap("readback")
    rec["ocr"], rec["model"] = reads[0], (reads[1] if len(reads) > 1 else None)
    rec["side_ok"] = side_by_button()[0] == o["side"]
    rec["ocr_ok"], rec["model_ok"] = same(rec["ocr"], o), bool(rec["model"]) and same(rec["model"], o)
    rec["backers"] = len(reads) - 1
    if not (form and same(form, o)):
        rec["problems"].append(f"读回不一致 OCR={rec['ocr']} 模型={reads[1:]}")
    r = click_xy_verified(lab_q[0] + QTY_DX, lab_q[1], "下单面板里「总量」右边的数字输入框", "", f"rh_clear_{o['code']}",
                          anchor=(lab_q[0], lab_q[1], "总"))
    if r["ok"]:                                               # 清空数量, 面板上不留填好的单
        select_all()
        backspace()
        type_digits("0")
    lap("clear")
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"  {o['side']} {o['code']} {o['name']} {o['price']}×{o['shares']}: OCR {'✓' if rec['ocr_ok'] else '✗'} "
        f"模型 {'✓' if rec['model_ok'] else '✗'} 方向 {'✓' if rec['side_ok'] else '✗'} {rec['seconds']}s {rec['t']}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max", type=int, default=5)
    a = ap.parse_args(argv)
    from pathlib import Path

    from .executor import BASE_URL, signed_post
    from .push_snapshot import DEFAULT_KEY
    from .ui_agent import ensure_qmt_foreground
    key = Path(DEFAULT_KEY).read_text(encoding="utf-8").strip().encode()
    today = date.today().isoformat()
    st, resp = signed_post(f"{BASE_URL}/api/qmt/ticket", {"profile": "qmt10w", "today": today}, key)
    if st != 200:
        print(f"取单据失败 {st}: {resp}", flush=True)
        return 1
    ticket = resp["ticket"]
    codes = {s["code"] for s in ticket.get("sell") or []} | {b["code"] for b in ticket.get("buy") or []} | \
        set((ticket.get("ranked") or [])[:a.max])
    quotes = {}
    for c in sorted(codes):
        try:
            quotes[c] = quote(c)
        except Exception as e:  # noqa: BLE001
            print(f"取 {c} 行情失败: {e}", flush=True)
    orders = plan_orders(ticket, quotes, a.max)
    print(f"[{datetime.now():%H:%M:%S}] 演练 {len(orders)} 笔 (空仓日用排名前 {a.max}): "
          + ", ".join(f"{o['side']}{o['code']}" for o in orders), flush=True)
    from .ui_lock import ui_lock
    try:
        with ui_lock("rehearse", wait_s=60):
            probs = ensure_qmt_foreground(print)
            results = [] if probs else [fill_one(o) for o in orders]
    except TimeoutError as e:
        print(f"界面被占用, 今天的演练跳过: {e}", flush=True)
        return 0
    bad = probs + [f"{r['side']} {r['code']}: {'; '.join(r['problems'])}" for r in results if r["problems"]]
    rep = {"today": today, "mode": "rehearsal", "results": results, **({"problems": bad} if bad else {})}
    signed_post(f"{BASE_URL}/api/qmt/exec_report", rep, key)
    ok = sum(1 for r in results if not r["problems"])
    print(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "orders": len(orders), "all_ok": ok,
                      "ocr_ok": sum(1 for r in results if r.get("ocr_ok")), "model_ok": sum(1 for r in results if r.get("model_ok")),
                      "avg_s": round(sum(r["seconds"] for r in results) / max(1, len(results)), 1), "problems": bad},
                     ensure_ascii=False)[:2000], flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
