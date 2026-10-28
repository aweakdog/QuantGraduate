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

BUY_TAB = (262, 523)                 # 下单面板左侧竖排「买入」页签 (10-27 截图量出)
SELL_TAB = (262, 587)
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


def parse_panel(code_text, form_text):
    """OCR 读回的下单面板文字 -> {code, price, volume, est_amount}。"""
    code = next(iter(re.findall(r"\d{6}", code_text)), None)
    pm = re.search(r"委托价\s*(\d+\.\d+)", form_text)
    vm = re.search(r"总[量里]\s*(\d+)", form_text)
    em = re.search(r"预估金额\S*?\s*([\d ,]+\.\d+)\s*元", form_text)
    return {"code": code, "price": float(pm.group(1)) if pm else None, "volume": int(vm.group(1)) if vm else None,
            "est_amount": float(em.group(1).replace(" ", "").replace(",", "")) if em else None}


def same(read, o):
    return (read.get("code") == o["code"] and read.get("price") is not None and abs(float(read["price"]) - o["price"]) < 1e-6
            and read.get("volume") is not None and int(float(read["volume"])) == o["shares"])


def fill_one(o, log=print):
    """填一笔并读回; 不提交。返回记录。"""
    from .ocr_read import locate_text, ocr_boxes, text_of
    from .ui_agent import screenshot_windows
    from .ui_channel import click_xy_verified, pick_candidate, read_form_zoomed, side_by_button, type_code
    from .ui_order import backspace, select_all, type_digits
    t0, rec = time.time(), {**o, "problems": []}
    tab = BUY_TAB if o["side"] == "买入" else SELL_TAB
    c = click_xy_verified(*tab, f"下单面板左侧竖排的「{o['side']}」页签", "", f"rh_tab_{o['code']}")
    time.sleep(1.0)
    side, _ = side_by_button()
    if not c["ok"] or side != o["side"]:
        rec["problems"].append(f"切到「{o['side']}」失败 (按钮方向 {side})")
        return {**rec, "seconds": round(time.time() - t0, 1)}
    type_code(o["code"])
    p = pick_candidate(o["code"])
    if not p["ok"]:
        rec["problems"].append("没选中下拉候选")
        return {**rec, "seconds": round(time.time() - t0, 1)}
    time.sleep(1.5)
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
    est = locate_text(screenshot_windows(), "预估金额", PANEL_LABELS, 2)
    if est:                                                   # 点一下标签让输入框失焦, 预估金额才刷新
        from .ui_agent import click_windows
        click_windows(*est)
        time.sleep(1.0)
    png = screenshot_windows()
    rec["ocr"] = parse_panel(text_of(ocr_boxes(png, CODE_AREA, 3)), text_of(ocr_boxes(png, FORM_AREA, 3)))
    model, _ = read_form_zoomed(qty_area=FORM_AREA)
    rec["model"] = model
    rec["side_ok"] = side_by_button()[0] == o["side"]
    rec["ocr_ok"], rec["model_ok"] = same(rec["ocr"], o), bool(model) and same(model, o)
    if not rec["ocr_ok"]:
        rec["problems"].append(f"OCR 读回不一致 {rec['ocr']}")
    if not rec["model_ok"]:
        rec["problems"].append(f"模型读回不一致 {model}")
    r = click_xy_verified(lab_q[0] + QTY_DX, lab_q[1], "下单面板里「总量」右边的数字输入框", "", f"rh_clear_{o['code']}",
                          anchor=(lab_q[0], lab_q[1], "总"))
    if r["ok"]:                                               # 清空数量, 面板上不留填好的单
        select_all()
        backspace()
        type_digits("0")
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"  {o['side']} {o['code']} {o['name']} {o['price']}×{o['shares']}: OCR {'✓' if rec['ocr_ok'] else '✗'} "
        f"模型 {'✓' if rec['model_ok'] else '✗'} 方向 {'✓' if rec['side_ok'] else '✗'} {rec['seconds']}s")
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
    probs = ensure_qmt_foreground(print)
    results = [] if probs else [fill_one(o) for o in orders]
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
