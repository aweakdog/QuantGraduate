# -*- coding: utf-8 -*-
"""跟单提醒: 判断每条实盘线现在有没有待办, 有才发消息。

【为什么不是"每天定点发一条"】
每天固定发, 人很快就会对它麻木, 真正要紧的那天也一样划过去。所以这里只在
【确实有事】时发, 没事就一个字不发 —— 收到即意味着必须动手。

【三个时点, 对应三件会被忘掉的事】
  preclose  执行日 14:35 —— 尾盘窗口是 14:50-15:00, 提前 15 分钟催下单。
                            这是最容易忘的一件, 而在此之前系统【没有任何
                            盘中提醒】, 只有盘后的 17:30/19:30/21:30。
  signal    盘后跑完流水线 —— 新计划出来了, 预告下一个交易日要做什么。
  confirm   执行日盘后 —— 催回填真实成交价。实盘线不确认就【不记账也不出
                          新信号】(live_signal 的 require_confirm 闸门),
                          忘了确认等于整条线停摆, 后果比忘记下单更严重。

【文案与网页同源】
判定与文案全部走 action_page.build_today() —— 也就是网页 /api/today 用的
同一个函数。绝不在这里另写一套"今天该做什么"的算法: 两套实现迟早会分叉,
到时候消息说买、网页说不用买, 人只会更懵。

【只管实盘线】
7 条线里有 3 条是纸面模式(系统按行情自动记账), 没有人需要为它们动手,
提醒它们纯属噪音。只挑 is_auto() 为假的线。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import trading_calendar  # noqa: E402
from action_page import PROFILES, build_today, display_name, is_auto  # noqa: E402
from notify_channels import get_channel, load_config  # noqa: E402

SENT_PATH = ROOT / "data" / "live" / "notify_sent.json"
SITE_URL = "http://eez041.ece.ust.hk:8080/"

SLOTS = ("preclose", "signal", "confirm")


def _load_sent():
    if SENT_PATH.exists():
        try:
            return json.loads(SENT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_sent(d):
    SENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SENT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SENT_PATH)


def pipeline_busy(root, now=None, max_run_hours=4):
    """流水线是否正在重建。正在跑的时候【一个字都不能发】。

    daily_rebuild 是逐条线依次生成信号的(live_signal_xxx 一条约 11-14 秒),
    整个流程 27-33 分钟, 而且**没有任何并发锁**。如果提醒恰好在中途跑起来,
    就会读到"前两条线已更新、后两条还是昨天"的状态, 发出一条残缺的计划;
    等流水线跑完, 下一次触发内容变了又发一条不一样的。

    对跟单提醒来说, "第一条是残的"比"晚十五分钟收到"糟得多 —— 人照着残缺
    清单下单会漏掉某几笔, 而且从此不再相信这个提醒。

    判据: started_at 有值而 finished_at 为空。正常完成、跳过(非交易日)、
    抛异常失败三条路径都会写 finished_at, 所以为空就是真的还在跑。

    例外: 进程被 SIGKILL(OOM 或 systemd 超时)时 except 块来不及执行,
    finished_at 会永远为空。所以加一个时限 —— 否则一次崩溃就让提醒永久沉默,
    而这恰恰是最该避免的失效方式。
    """
    now = now or datetime.now()
    p = Path(root) / "data" / "live" / "pipeline_status.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 读不到状态就不阻断发送。看不懂状态而选择永久沉默, 比偶尔发错更危险。
        return False
    started, finished = st.get("started_at"), st.get("finished_at")
    if not started or finished:
        return False
    try:
        t0 = datetime.fromisoformat(str(started))
    except ValueError:
        return False
    return (now - t0) < timedelta(hours=max_run_hours)


def _rows_text(rows, side):
    """把买卖清单压成几行。群消息里过长没人看, 所以只给代码/名称/股数。"""
    out = []
    for r in rows:
        nm = (r.get("name") or "")[:6]
        sh = r.get("shares") or 0
        out.append(f"　{side} {r.get('code','')} {nm} {sh}股")
    return out


def collect(root, now=None):
    """扫所有实盘线, 归纳出每条线【现在】的待办状态。

    只做归纳不做取舍 —— 哪些状态在哪个时点该发, 由 compose() 决定。
    这样同一份状态可以被三个时点复用, 也方便测试。
    """
    now = now or datetime.now()
    items = []
    for pid in PROFILES:
        if is_auto(pid):
            continue
        d = build_today(root, pid)
        win = d.get("exec_window") or {}
        fresh = d.get("freshness") or {}
        items.append({
            "pid": pid,
            "name": display_name(pid),
            "action": d.get("action"),
            "sell": d.get("sell") or [],
            "buy": d.get("buy") or [],
            "n_ops": len(d.get("sell") or []) + len(d.get("buy") or []),
            "rel": win.get("rel"),
            "phase": win.get("phase"),
            "when_text": win.get("when_text"),
            "slot_label": win.get("slot_label"),
            "awaiting": d.get("awaiting_confirm"),
            "can_confirm": bool(d.get("can_confirm")),
            "overdue": fresh.get("awaiting_overdue_days"),
            "stale": bool(fresh.get("stale")),
            "stale_reason": fresh.get("reason"),
            "hold_n": len(d.get("hold") or []),
        })
    return items


def compose(slot, items, now=None):
    """按时点挑出该提醒的线并生成文案。没有任何线需要提醒时返回 None。

    返回 None 是正常且常见的结果 —— 大多数日子什么都不该发。
    """
    now = now or datetime.now()

    if slot == "preclose":
        # 执行日就是今天, 且窗口还没过。phase=="after" 时不再催 ——
        # 窗口已过还催"快下单"是错的, 那时该走的是晚上的 confirm。
        hit = [x for x in items
               if x["action"] in ("trade", "cash")
               and x["rel"] == "today" and x["phase"] in ("before", "open")]
        # 被"等确认"卡住的线在执行日同样要提醒: 它不会出新信号, 人却往往
        # 以为"今天没消息就是不用操作", 实际是整条线停摆了。
        stuck = [x for x in items if x["action"] == "await"]
        if not hit and not stuck:
            return None
        lines = [f"**尾盘提醒 · {now.strftime('%m月%d日')}**",
                 "距离下单窗口 14:50–15:00 还有约 15 分钟。", ""]
        for x in hit:
            if x["action"] == "cash":
                lines.append(f"▸ **{x['name']}** 清仓避险, 卖出全部 {x['hold_n']} 只")
                continue
            lines.append(f"▸ **{x['name']}** 卖{len(x['sell'])}买{len(x['buy'])}")
            lines += _rows_text(x["sell"], "卖")
            lines += _rows_text(x["buy"], "买")
        for x in stuck:
            lines.append(f"▸ **{x['name']}** 还卡在等确认, 这条线现在不会出新信号")
        lines += ["",
                  "参考价是信号日收盘价, **不是目标价**。价格涨多了按预算改股数。",
                  SITE_URL]
        return "\n".join(lines)

    if slot == "signal":
        # 盘后新计划已出, 预告下一个交易日。rel=="today" 的不在这里发 ——
        # 那属于当天该做而没做, 由 confirm 兜底, 语气也完全不同。
        hit = [x for x in items
               if x["action"] in ("trade", "cash") and x["rel"] in ("tomorrow", "future")]
        if not hit:
            return None
        when = hit[0]["when_text"] or "下一个交易日 尾盘"
        lines = [f"**新计划 · {when}**", ""]
        for x in hit:
            if x["action"] == "cash":
                lines.append(f"▸ **{x['name']}** 清仓避险, 卖出全部 {x['hold_n']} 只")
                continue
            lines.append(f"▸ **{x['name']}** 卖{len(x['sell'])}买{len(x['buy'])}")
            lines += _rows_text(x["sell"], "卖")
            lines += _rows_text(x["buy"], "买")
        lines += ["", f"到点前会再提醒一次。明细 {SITE_URL}"]
        return "\n".join(lines)

    if slot == "confirm":
        # 只提醒真的在等确认的线。can_confirm 为假时提交也没用(没有执行日
        # 行情, 结算不了), 催了只会让人白跑一趟。
        hit = [x for x in items if x["awaiting"] and x["can_confirm"]]
        if not hit:
            return None
        lines = ["**该回填成交了**",
                 "实盘线在你确认之前**不记账、也不出新信号**。", ""]
        for x in hit:
            ac = x["awaiting"] or {}
            od = x["overdue"]
            tail = f" · 已逾期 {od} 个交易日" if od else ""
            lines.append(f"▸ **{x['name']}** 待确认 {ac.get('exec_date','')}{tail}")
        lines += ["", f"没下单就选「一笔都没成交」。{SITE_URL}"]
        return "\n".join(lines)

    raise ValueError(f"未知时点 {slot!r}")


def is_urgent(slot, items):
    """这条提醒值不值得 @所有人。

    企业微信的 markdown 消息【不支持 @所有人】, 只有 text 类型可以。所以
    "要不要 @" 同时决定了消息用什么格式发, 不是个纯装饰的选择。

    判据是"错过的代价", 不是"事情大小":
      尾盘催单     —— 错过就是这一轮换仓没执行, 必须打断人。
      逾期未确认   —— 整条线已经停摆(不记账也不出新信号), 越拖越歪。
      当天正常待确认/新计划预告 —— 晚上有的是时间, @all 属于滥用。

    天天 @所有人的下场是被人关掉群提醒, 那时连真正紧急的也送不到了。
    """
    if slot == "preclose":
        return True
    if slot == "confirm":
        return any(x["awaiting"] and x["can_confirm"] and x["overdue"] for x in items)
    return False


def admin_alert(items):
    """流水线真的挂了时的告警。与给跟单者的提醒分开 ——

    "系统坏了"是运维信息, 混进操作提醒里会让人以为自己要做什么。
    """
    bad = [x for x in items if x["stale"] and x["stale_reason"] == "pipeline"]
    if not bad:
        return None
    names = ", ".join(x["name"] for x in bad)
    return f"**[运维] 流水线可能未跑或失败**\n影响: {names}\n{SITE_URL}ops"


def main():
    ap = argparse.ArgumentParser(description="跟单提醒")
    ap.add_argument("--slot", required=True, choices=SLOTS + ("all",))
    ap.add_argument("--channel", default=None,
                    help="覆盖配置里的通道 (stdout/queue/wecom)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印文案, 不发送也不记录已发")
    ap.add_argument("--force", action="store_true",
                    help="忽略去重, 强制再发一次")
    ap.add_argument("--ignore-calendar", action="store_true",
                    help="非交易日也跑(仅调试用)")
    args = ap.parse_args()

    now = datetime.now()

    # 非交易日一律不发。14:35 的 timer 每个工作日都会触发, 但节假日的工作日
    # 并不开市 —— 那天发"该下单了"是纯粹的误导。
    if not args.ignore_calendar:
        days, _ = trading_calendar.load()
        if not trading_calendar.is_trading_day(now.date(), days):
            print(f"[notify] {now.date()} 非交易日, 跳过")
            return 0

    # 流水线重建中就退出, 让下一次触发来发。唯一的例外是尾盘催单:
    # 下单窗口只有 14:50-15:00 这 10 分钟, 等不起下一轮; 而那个时点
    # 流水线本来也不该在跑(它只在盘后 17:30 之后启动)。
    if args.slot != "preclose" and pipeline_busy(ROOT, now):
        print("[notify] 流水线正在重建, 本轮跳过(避免发出残缺的计划)")
        return 0

    items = collect(ROOT, now)
    slots = SLOTS if args.slot == "all" else (args.slot,)

    cfg = load_config()
    channel = get_channel(args.channel or ("stdout" if args.dry_run else None), cfg)
    sent = _load_sent()
    today = str(now.date())
    n_sent = 0

    for slot in slots:
        text = compose(slot, items, now)
        if not text:
            print(f"[notify] {slot}: 无待办, 不发送")
            continue
        # 去重按内容哈希而不是"今天这个时点发过没" —— 盘后会跑三轮流水线,
        # 计划可能在第二轮才出来或发生变化, 这时应该重发; 内容没变才跳过。
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        prev = (sent.get(today) or {}).get(slot) or {}
        if prev.get("hash") == h and not args.force:
            print(f"[notify] {slot}: 内容与已发的相同, 跳过")
            continue

        urgent = is_urgent(slot, items)
        ok = channel.send(text, meta={"slot": slot, "date": today, "urgent": urgent})
        print(f"[notify] {slot}: 经 {channel.name} 发送 {'成功' if ok else '失败'}"
              f"{' (@所有人)' if urgent else ''}")
        if ok and not args.dry_run:
            sent.setdefault(today, {})[slot] = {
                "hash": h, "at": now.isoformat(timespec="seconds"),
                "channel": channel.name,
            }
            n_sent += 1

    alert = admin_alert(items)
    if alert:
        print("[notify] 运维告警:")
        print(alert)

    if not args.dry_run and n_sent:
        _save_sent(sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
