"""每日国债逆回购 (2026-10-27 起, 用户要求): 交易日 14:40 把闲置资金按 GC001 (204001) 借出一天, 次日自动到账。

  .venv\\Scripts\\python.exe -m qmt_bridge.repo_daily [--dry-run]
金额 = 读屏可用资金 - 保留。保留 = 1000 元; 当天 qmt10w 单据有买入 (或取不到单据) 时再留出线内现金 ——
策略当天要用的钱不借 (借出的钱次日开盘才回来)。
硬限制: 只在 14:30–14:50; 只卖出 204001; 张数为 10 的倍数且 ≥ 10; 金额 ≤ 可用 - 保留 且 ≤ MAX_AMOUNT;
利率 = 买一 - 0.05 向下取到 0.005, 且在 [买一-0.2, 买一]; 每天最多一次 (标记文件); 开关文件
D:\\qmtcode\\REPO_ENABLED 不存在就只填单核对、不提交。每步读回核对 (放大双读、按钮颜色、预估金额 = 张数×100),
任何不符即停。提交后再读屏: 可用资金应减少 ≈ 借出金额; 结果写进快照 notes 推给 041 (企业微信 + 邮件)。
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

ENABLED = Path(r"D:\qmtcode\REPO_ENABLED")
DONE_DIR = Path(r"D:\qmtcode\repo_done")
WINDOWS = (("10:00:00", "11:20:00"), ("14:30:00", "14:50:00"))   # 上午只在确认空仓日借 (利率通常更高); 下午兜底
RESERVE = 1000.0
MAX_AMOUNT = 2_000_000.0
QTY_BOX_REPO = (367, 848)          # 10-27 量出: 选中 GC001 后「总量」输入框
EST_LABEL = (300, 878)             # 「预估金额」标签 (纯文字, 点它无副作用)


def plan_amount(cash, ticket):
    """返回 (借出金额, 保留, 说明)。金额取 1000 的整数倍; 不足 1000 返回 0。"""
    reserve, why = RESERVE, "保留 1000 元"
    if ticket is None:
        reserve += 100000.0
        why += " + 取不到单据, 保守留出线内 10 万"
    elif ticket.get("buy"):
        reserve += float(ticket.get("line_cash") or 0)
        why += f" + 今天策略有买入, 留出线内现金 {float(ticket.get('line_cash') or 0):,.0f}"
    amount = min(MAX_AMOUNT, math.floor(max(0.0, float(cash or 0) - reserve) / 1000) * 1000)
    return float(amount), reserve, why


def morning_ok(ticket):
    """上午只在确认空仓日借: 能取到当天计划, 且计划不买也不卖。"""
    return ticket is not None and not ticket.get("buy") and not ticket.get("sell")


def repo_rate(bid):
    """委托利率 = 买一 - 0.05, 向下取到 0.005 (卖出利率低于买一即按买一立即成交)。"""
    return round(math.floor((bid - 0.05) / 0.005 + 1e-9) * 0.005, 3)


def guard_repo_daily(exp, bid, max_amount):
    if exp["code"] != "204001" or exp["side"] != "卖出":
        return False, "只允许卖出 204001"
    if exp["volume"] < 10 or exp["volume"] % 10:
        return False, f"张数 {exp['volume']} 不是 10 的倍数"
    if exp["volume"] * 100 > max_amount + 1e-6:
        return False, f"金额 {exp['volume'] * 100:,} 超过可借 {max_amount:,.0f}"
    if bid is None or not (bid - 0.2 <= exp["price"] <= bid + 1e-9):
        return False, f"利率 {exp['price']} 不在 [买一-0.2, 买一]"
    return True, "ok"


def quote_bid():
    with urllib.request.urlopen("https://qt.gtimg.cn/q=sh204001", timeout=10) as r:
        f = r.read().decode("gbk", "replace").split("~")
    return float(f[9])


def set_volume(lots):
    from .ui_channel import click_xy_verified
    from .ui_order import backspace, select_all, type_digits
    out = click_xy_verified(*QTY_BOX_REPO, "下单面板里「总量」右边的数字输入框", "", "repo_qty")
    if out["ok"]:
        time.sleep(0.3)
        select_all()
        backspace()
        type_digits(str(int(lots)))
        time.sleep(0.8)
    return out


def run(dry_run, log=print):
    from .executor import BASE_URL, signed_post
    from .push_snapshot import DEFAULT_KEY, DEFAULT_URL, push
    from .screen_snapshot import take
    from .ui_channel import (
        SELL_BTN,
        click_xy_verified,
        confirm,
        pick_candidate,
        read_form_zoomed,
        set_price,
        side_by_button,
        tab_sell,
        type_code,
    )

    today = date.today().isoformat()
    marker = DONE_DIR / f"{today}.json"
    if marker.exists():
        return {"ok": True, "status": "already_done"}
    now = datetime.now().strftime("%H:%M:%S")
    if not dry_run and not any(a <= now <= b for a, b in WINDOWS):
        return {"ok": False, "status": "outside_window", "now": now}
    morning = now < "12:00:00"
    key = Path(DEFAULT_KEY).read_text(encoding="utf-8").strip().encode()
    snap0 = take("manual", log=log)
    if not snap0.get("account_online"):
        return {"ok": False, "status": "snapshot_offline", "error": snap0.get("error")}
    cash = snap0["asset"]["cash"]
    st, resp = signed_post(f"{BASE_URL}/api/qmt/ticket", {"profile": "qmt10w", "today": today}, key)
    ticket = resp.get("ticket") if st == 200 else None
    if morning and not morning_ok(ticket):
        return {"ok": True, "status": "defer_to_afternoon", "why": "今天不是确认空仓日 (或取不到计划), 留到 14:40 再判断"}
    amount, reserve, why = plan_amount(cash, ticket)
    log(f"可用 {cash:,.2f}; {why}; 借出 {amount:,.0f}")
    if amount < 1000:
        return {"ok": True, "status": "nothing_to_lend", "cash": cash, "why": why}
    bid = quote_bid()
    exp = {"code": "204001", "side": "卖出", "price": repo_rate(bid), "volume": int(amount // 100)}
    ok, g = guard_repo_daily(exp, bid, cash - reserve)
    if not ok:
        return {"ok": False, "status": "guard", "why": g, "exp": exp}
    steps = {"tab": tab_sell(), "code": type_code("204001")}
    steps["pick"] = pick_candidate("204001")
    steps["price"] = set_price(f"{exp['price']:.3f}")
    steps["qty"] = set_volume(exp["volume"])
    from .ui_channel import click_xy_checked
    click_xy_checked(*EST_LABEL, "repo_blur")      # 点一下「预估金额」标签让总量框失焦, 预估金额才刷新 (10-27 演练时读到旧值 1000)
    time.sleep(1.0)
    if not all(s.get("ok") for s in steps.values()):
        return {"ok": False, "status": "fill_failed", "steps": {k: v.get("ok") for k, v in steps.items()}}
    form, reads = read_form_zoomed()
    side, _ = side_by_button()
    bad = []
    if form is None:
        bad.append("放大读回两次不一致")
    else:
        if form.get("code") != "204001":
            bad.append(f"代码 {form.get('code')}")
        if form.get("price") is None or abs(float(form["price"]) - exp["price"]) > 1e-6:
            bad.append(f"利率 {form.get('price')}")
        if form.get("volume") is None or int(float(form["volume"])) != exp["volume"]:
            bad.append(f"张数 {form.get('volume')}")
        if form.get("est_amount") is not None and abs(float(form["est_amount"]) - amount) > 0.01:
            log(f"注意: 预估金额读到 {form.get('est_amount')} (界面刷新慢或读错), 以张数和确认框为准")
    if side != "卖出":
        bad.append(f"按钮方向 {side}")
    if bad:
        return {"ok": False, "status": "readback_mismatch", "bad": bad, "reads": reads}
    if dry_run or not ENABLED.exists():
        set_volume(0)
        return {"ok": True, "status": "dry_run", "exp": exp, "reads": reads,
                "why": "演练 (--dry-run)" if dry_run else f"开关文件 {ENABLED} 不存在, 只演练"}
    sub = click_xy_verified(*SELL_BTN, "下单面板的绿色「卖出」提交按钮", "卖", "repo_submit")
    if not sub["ok"]:
        return {"ok": False, "status": "submit_refused", "verify": sub.get("verify")}
    conf = confirm(exp, "order")
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "exp": exp,
                                  "confirmed": conf.get("ok")}, ensure_ascii=False), encoding="utf-8")
    if not conf.get("ok"):
        return {"ok": False, "status": "confirm_refused", "why": conf.get("why"), "bad": conf.get("bad")}
    time.sleep(3)
    snap1 = take("manual", log=log)
    got = None if not snap1.get("account_online") else cash - snap1["asset"]["cash"]
    lent_ok = got is not None and abs(got - amount) <= 1 + amount * 0.0001
    note = (f"逆回购: 借出 {amount:,.0f} 元 (GC001, 委托利率 {exp['price']:.3f}%, 明天到账)" if lent_ok else
            f"逆回购: 已提交 {amount:,.0f} 元但可用资金只减少了 {got}, 请人工核对")
    snap1["notes"] = (snap1.get("notes") or []) + [note]
    status, _ = push(snap1, DEFAULT_URL, key)
    return {"ok": lent_ok, "status": "lent" if lent_ok else "check_failed", "amount": amount, "rate": exp["price"],
            "note": note, "push": status}


# check_failed = 已提交但事后核对没通过: run() 已把具体情况写进快照 notes 推送, 不再另发「没做成」(10-28 那条说法不对)
QUIET = {"lent", "already_done", "nothing_to_lend", "defer_to_afternoon", "dry_run", "outside_window", "check_failed"}


def failure_text(out, now=None):
    """没做成时发给 041 的一句话 (10-28: 上午因模型接口 529 中断, 之前失败只记日志不提醒)。"""
    now = now or datetime.now().strftime("%H:%M")
    why = out.get("why") or out.get("error") or out.get("bad") or ""
    tail = "下午 14:40 会再试一次" if now < "12:00" else "今天不再重试, 钱留在账上"
    return f"逆回购没做成 ({out.get('status')}): {str(why)[:200]}。{tail}"


def report_failure(out):
    try:
        from .executor import BASE_URL, signed_post
        from .push_snapshot import DEFAULT_KEY
        key = Path(DEFAULT_KEY).read_text(encoding="utf-8").strip().encode()
        signed_post(f"{BASE_URL}/api/qmt/exec_report",
                    {"today": date.today().isoformat(), "mode": "repo", "problems": [failure_text(out)]}, key)
    except Exception as e:  # noqa: BLE001
        print(f"失败提醒也没发出去: {type(e).__name__}: {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = run(a.dry_run, log=lambda m: print(m, flush=True))
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "status": "exception", "error": f"{type(e).__name__}: {str(e)[:300]}"}
    if not a.dry_run and out.get("status") not in QUIET:
        report_failure(out)
    print(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **out}, ensure_ascii=False, default=str)[:3000],
          flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
